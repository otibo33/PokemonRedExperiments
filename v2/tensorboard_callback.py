"""Robust Pokémon Red TensorBoard callback.

Preserves the user's original TensorboardCallback name and histogram/stat
logs. Fixes terminal-episode capture for SB3 VecEnv auto-resets and maintains
cumulative exploration maps/progression across episode resets and process
restarts (provided the same log_dir is reused).

Logs party-board and gameplay-plus-party images through the same SB3
TensorBoard stream as exploration maps. No rewards, policy observations, or
training actions are changed by this callback.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from einops import rearrange, reduce
from PIL import Image as PILImage, ImageDraw, ImageFont
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Image
from torch.utils.tensorboard import SummaryWriter
from gen1_party_metadata import (
    INTERNAL_TO_NATIONAL_DEX,
    LEGACY_TYPE_NAMES_BY_INTERNAL_ID,
    LEGACY_STARTER_INTERNAL_IDS,
    LEGACY_HM_PREFERRED_CARRIERS_BY_INTERNAL_ID,
    merge_missing_party_metadata,
)


def merge_dicts(dicts):
    """Return numeric means and distributions, ignoring non-numeric fields."""
    values_by_key: dict[str, list[float]] = {}
    for item in dicts:
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            if isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)):
                values_by_key.setdefault(key, []).append(float(value))

    means = {key: float(np.mean(values)) for key, values in values_by_key.items()}
    distributions = {
        key: np.asarray(values, dtype=np.float32)
        for key, values in values_by_key.items()
    }
    return means, distributions


def _as_uint8_map(value: Any, expected_shape: tuple[int, int] | None = None):
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim != 2:
        return None
    if expected_shape is not None and tuple(array.shape) != tuple(expected_shape):
        return None
    return np.ascontiguousarray(np.clip(array, 0, 255), dtype=np.uint8)


def _make_map_mosaic(maps: np.ndarray) -> np.ndarray:
    """Arrange N HxW maps in a roughly square grid; supports any N >= 1."""
    maps = np.asarray(maps, dtype=np.uint8)
    if maps.ndim != 3 or maps.shape[0] < 1:
        raise ValueError(f"Expected NxHxW maps, got {maps.shape}")
    count, height, width = maps.shape
    columns = int(math.ceil(math.sqrt(count)))
    rows = int(math.ceil(count / columns))
    canvas = np.zeros((rows * height, columns * width), dtype=np.uint8)
    for index, item in enumerate(maps):
        row, column = divmod(index, columns)
        canvas[
            row * height:(row + 1) * height,
            column * width:(column + 1) * width,
        ] = item
    return canvas


def _nearest_neighbor_scale(image: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return np.ascontiguousarray(image, dtype=np.uint8)
    return np.ascontiguousarray(
        np.repeat(np.repeat(image, scale, axis=0), scale, axis=1),
        dtype=np.uint8,
    )


class TensorboardCallback(BaseCallback):
    """Keep the original callback API while preserving complete map history.

    Existing use remains valid:
        TensorboardCallback(log_dir)
        TensorboardCallback(log_dir, verbose=1)

    Args:
        log_dir: Existing run/session directory. A `_trajectory_history`
            subdirectory is created here to persist map history between runs.
        verbose: Stable-Baselines3 callback verbosity.
        map_image_scale: Nearest-neighbour visual enlargement (default 2).
        scalar_log_every: Aggregate environment timesteps between progress
            scalar updates (default 2560).
        party_image_every: Aggregate timesteps between party dashboard images
            (default 40960); an initial image is also queued at startup.
    """

    def __init__(
        self,
        log_dir,
        verbose=0,
        map_image_scale=2,
        scalar_log_every=2560,
        map_log_every=10000,
        party_image_every=40960,
        party_frame_scale=3,
    ):
        super().__init__(verbose=verbose)
        self.log_dir = Path(log_dir).expanduser()
        self.writer = None
        self.map_image_scale = int(map_image_scale)
        self.scalar_log_every = int(scalar_log_every)
        self.map_log_every = int(map_log_every)
        self.party_image_every = int(party_image_every)
        self.party_frame_scale = int(party_frame_scale)
        if self.map_image_scale < 1:
            raise ValueError("map_image_scale must be a positive integer")
        if self.scalar_log_every < 1:
            raise ValueError("scalar_log_every must be a positive integer")
        if self.map_log_every < 1:
            raise ValueError("map_log_every must be a positive integer")
        if self.party_image_every < 1:
            raise ValueError("party_image_every must be a positive integer")
        if self.party_frame_scale < 1:
            raise ValueError("party_frame_scale must be a positive integer")

        self.history_dir = self.log_dir / "_trajectory_history"
        self.maps_path = self.history_dir / "worker_exploration_maps.npz"
        self.metadata_path = self.history_dir / "progress_metadata.json"
        self.best_party_path = self.history_dir / "most_advanced_party.json"
        self.best_party_record: dict[str, Any] | None = None
        self.best_party_rank: tuple[int, ...] | None = None
        self._best_party_dirty = False
        self.worker_maps: np.ndarray | None = None
        self.map_shape: tuple[int, int] | None = None
        self.best_map_progress = -1
        self.best_map_id = -1
        self.episodes_completed = 0
        self.event_flags: dict[str, str] = {}
        self._last_scalar_log_timestep = 0
        self._last_map_log_timestep = 0
        self._last_party_image_timestep = 0
        self._warned_missing_snapshot = False
        self._warned_party_read = False

    def _on_training_start(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir / "histogram"))

        # Discover the fixed global map dimensions from the environments.
        try:
            existing_maps = self.training_env.get_attr("lifetime_explore_map")
            if existing_maps:
                first = np.asarray(existing_maps[0])
                if first.ndim == 2:
                    self.map_shape = tuple(int(v) for v in first.shape)
        except Exception:
            # We still support legacy environments for statistics. The
            # terminal-map tags require the patched red_gym_env_v2.py.
            self.map_shape = None

        self._load_persisted_history()
        # Track and display only the worker with the strongest story progress.
        # The best party record is also persisted across resumes in this log_dir.
        self._prime_best_party_from_envs()
        self._last_party_image_timestep = int(self.num_timesteps)
        initial_count = self._log_party_images()
        if initial_count:
            print(
                "[TensorboardCallback] Most-advanced party dashboard queued "
                f"(worker {self.best_party_record['worker_index']}, "
                f"story-map progress {self.best_party_record['best_map_progress']}). "
                "It will be written with the next PPO TensorBoard dump."
            )

    def _load_persisted_history(self) -> None:
        n_envs = int(self.training_env.num_envs)
        self.worker_maps = None

        if self.maps_path.exists():
            try:
                with np.load(self.maps_path, allow_pickle=False) as saved:
                    old_maps = np.asarray(saved["worker_maps"], dtype=np.uint8)
                if old_maps.ndim == 3 and old_maps.shape[0] > 0:
                    if self.map_shape is None:
                        self.map_shape = tuple(int(v) for v in old_maps.shape[1:])
                    if tuple(old_maps.shape[1:]) == tuple(self.map_shape):
                        if old_maps.shape[0] == n_envs:
                            self.worker_maps = old_maps.copy()
                        else:
                            # If worker count changes, keep the complete prior
                            # union by seeding each new worker with that union.
                            merged = np.max(old_maps, axis=0)
                            self.worker_maps = np.repeat(merged[None, :, :], n_envs, axis=0)
            except Exception as exc:
                self._warn(f"Could not load saved exploration maps ({exc}); starting map history empty.")

        if self.worker_maps is None and self.map_shape is not None:
            self.worker_maps = np.zeros((n_envs, *self.map_shape), dtype=np.uint8)

        if self.metadata_path.exists():
            try:
                metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
                self.best_map_progress = int(metadata.get("best_map_progress", -1))
                self.best_map_id = int(metadata.get("best_map_id", -1))
                self.episodes_completed = int(metadata.get("episodes_completed", 0))
                flags = metadata.get("event_flags", {})
                if isinstance(flags, dict):
                    self.event_flags = {str(k): str(v) for k, v in flags.items()}
            except Exception as exc:
                self._warn(f"Could not load progress metadata ({exc}); starting metadata fresh.")

        if self.best_party_path.exists():
            try:
                record = json.loads(self.best_party_path.read_text(encoding="utf-8"))
                if isinstance(record, dict) and isinstance(record.get("party_snapshot"), list):
                    # Recompute the rank from named fields so persisted records
                    # remain compatible if tie-breaker order changes.
                    self.best_party_record = record
                    level_metrics = self._party_level_metrics(
                        record.get("party_snapshot", []),
                        record.get("party_level_spread_target", 10),
                    )
                    record.update(level_metrics)
                    self.best_party_rank = self._party_rank(record)
                    record["rank"] = [float(value) for value in self.best_party_rank]
                    self._save_best_party_record()
            except Exception as exc:
                self._warn(f"Could not load saved most-advanced party ({exc}); waiting for a new snapshot.")

    def _save_best_party_record(self) -> None:
        if self.best_party_record is None:
            return
        self.history_dir.mkdir(parents=True, exist_ok=True)
        temp_path = self.best_party_path.with_suffix(".json.tmp")
        temp_path.write_text(
            json.dumps(self.best_party_record, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temp_path, self.best_party_path)

    @staticmethod
    def _party_rank(info: dict[str, Any]) -> tuple[float, ...]:
        """Rank workers by furthest story-map progress, then readiness.

        best_map_progress is the highest essential-story-map milestone ever
        reached and is the primary key. Current map_progress breaks ties before
        badges, HM readiness, Pokédex, captures, and party composition.
        """
        def number(key: str, default: int = 0) -> int:
            try:
                return int(info.get(key, default))
            except (TypeError, ValueError):
                return default

        def decimal(key: str, default: float = 0.0) -> float:
            try:
                return float(info.get(key, default))
            except (TypeError, ValueError):
                return default

        return (
            float(number("best_map_progress", number("map_progress", -1))),
            float(number("map_progress", -1)),
            float(number("badges")),
            float(number("party_size")),
            float(number("hm_critical_ready")),
            float(number("hm_critical_moves_taught")),
            float(number("hm_moves_taught")),
            float(number("pokedex_owned")),
            float(number("captures")),
            float(number("party_species_count")),
            decimal("party_diversity_score"),
            decimal("party_level_balance_score"),
        )

    @staticmethod
    def _party_level_metrics(party, target: int = 10) -> dict[str, float | int]:
        """Recompute level spread/score from the exact party snapshot being logged."""
        levels = []
        for mon in party if isinstance(party, (list, tuple)) else []:
            if not isinstance(mon, dict):
                continue
            try:
                level = int(mon.get("level", 0))
            except (TypeError, ValueError):
                continue
            if level > 0:
                levels.append(level)
        target = max(0, int(target))
        # A one-member party is not evidence of balanced team levels.
        if len(levels) < 2:
            return {
                "party_level_balance_eligible": 0,
                "party_level_balance_score": 0.0,
                "party_level_spread": 0,
                "party_level_spread_target": int(target),
                "party_level_spread_excess": 0,
                "party_level_spread_within_target": 0,
            }
        spread = max(levels) - min(levels)
        score = 1.0 if spread <= target else (float(target) / spread if target else 0.0)
        return {
            "party_level_balance_eligible": 1,
            "party_level_balance_score": float(max(0.0, min(1.0, score))),
            "party_level_spread": int(spread),
            "party_level_spread_target": int(target),
            "party_level_spread_excess": int(max(0, spread - target)),
            "party_level_spread_within_target": int(spread <= target),
        }

    def _consider_party_candidate(
        self,
        worker_index: int,
        info: dict[str, Any],
        step: int,
        *,
        terminal: bool = False,
    ) -> bool:
        """Keep the party snapshot associated with the best worker progression."""
        payload = info.get("episode_snapshot") if terminal else None
        if not isinstance(payload, dict):
            payload = info

        party = payload.get("party_snapshot", info.get("party_snapshot"))
        if not isinstance(party, (list, tuple)):
            return False

        # Terminal episode snapshots are captured before SB3 auto-resets the
        # worker, so their party and progression are the actual episode result.
        merged = dict(info)
        merged.update(payload)
        rank = self._party_rank(merged)
        normalized_party = [dict(mon) if isinstance(mon, dict) else {} for mon in party]

        # Refresh presentation metadata for the historical best worker even
        # when its current progression rank is below the stored best.
        metadata_upgraded = False
        if isinstance(self.best_party_record, dict):
            previous_worker = int(self.best_party_record.get("worker_index", -1))
            if previous_worker == int(worker_index):
                enriched, metadata_upgraded = merge_missing_party_metadata(
                    self.best_party_record.get("party_snapshot", []), normalized_party
                )
                if metadata_upgraded:
                    self.best_party_record["party_snapshot"] = enriched
                    self._save_best_party_record()
                    self._best_party_dirty = True

        level_metrics = self._party_level_metrics(
            normalized_party, merged.get("party_level_spread_target", 10)
        )
        merged.update(level_metrics)
        if self.best_party_rank is not None:
            if rank < self.best_party_rank:
                return metadata_upgraded
            if rank == self.best_party_rank:
                # Keep the same leading worker's party current when its actual
                # team composition or moves change, but ignore routine HP/status
                # fluctuations to avoid writing the JSON file every emulator step.
                previous_worker = int(self.best_party_record.get("worker_index", -1))
                if previous_worker != int(worker_index):
                    return False
                previous_party = self.best_party_record.get("party_snapshot", [])
                def signature(values):
                    return tuple(
                        (
                            mon.get("species"), mon.get("level"),
                            tuple(mon.get("moves", [])),
                            tuple(mon.get("hm_can_learn", [])),
                            tuple(mon.get("hm_knows", [])),
                            mon.get("national_dex"),
                            tuple(mon.get("type_ids", [])),
                            tuple(mon.get("types", [])),
                            bool(mon.get("is_starter", False)),
                            tuple(mon.get("hm_preferred_carrier_for", [])),
                        )
                        for mon in values if isinstance(mon, dict)
                    )
                if signature(previous_party) == signature(normalized_party):
                    return metadata_upgraded

        self.best_party_rank = rank
        self.best_party_record = {
            "worker_index": int(worker_index),
            "timestep": int(step),
            "rank": [float(value) for value in rank],
            "best_map_progress": int(merged.get("best_map_progress", -1)),
            "best_map_id": int(merged.get("best_map_id", merged.get("map_id", -1))),
            "map_progress": int(merged.get("map_progress", -1)),
            "map_id": int(merged.get("map_id", -1)),
            "badges": int(merged.get("badges", 0)),
            "hm_critical_ready": int(merged.get("hm_critical_ready", 0)),
            "hm_critical_moves_taught": int(merged.get("hm_critical_moves_taught", 0)),
            "hm_moves_taught": int(merged.get("hm_moves_taught", 0)),
            "pokedex_owned": int(merged.get("pokedex_owned", 0)),
            "captures": int(merged.get("captures", 0)),
            "party_size": int(merged.get("party_size", len(party))),
            "party_fill_score": float(merged.get("party_fill_score", min(len(party), 6) / 6.0)),
            "party_species_count": int(merged.get("party_species_count", len(party))),
            "party_diversity_score": float(merged.get("party_diversity_score", 0.0)),
            "party_level_balance_eligible": int(merged.get("party_level_balance_eligible", level_metrics.get("party_level_balance_eligible", 0))),
            "party_level_balance_score": float(merged.get("party_level_balance_score", 0.0)),
            "party_level_spread": int(merged.get("party_level_spread", 0)),
            "party_level_spread_target": int(merged.get("party_level_spread_target", 10)),
            "party_level_spread_excess": int(merged.get("party_level_spread_excess", 0)),
            "party_level_spread_within_target": int(merged.get("party_level_spread_within_target", 0)),
            "party_snapshot": normalized_party,
        }
        self._save_best_party_record()
        self._best_party_dirty = True
        return True

    def _prime_best_party_from_envs(self) -> None:
        """Capture an initial baseline from live VecEnv workers at startup."""
        if self.best_party_record is not None or self.training_env is None:
            return
        try:
            snapshots = self.training_env.env_method("read_party_snapshot")
            best_progress = self.training_env.get_attr("best_map_progress")
            best_map_ids = self.training_env.get_attr("best_map_id")
            map_progress = self.training_env.get_attr("max_map_progress")
            map_ids = []  # best_map_id is the known story-progression map
            badges = self.training_env.env_method("get_badges")
            dex = self.training_env.env_method("read_pokedex_owned_count")
            captures = self.training_env.get_attr("capture_count")
            hm_ready = self.training_env.get_attr("current_hm_critical_ready_count")
            hm_moves = self.training_env.get_attr("current_hm_moves_taught_count")
            hm_critical_moves = self.training_env.get_attr("current_hm_critical_moves_taught_count")
            party_counts = self.training_env.env_method("read_unique_party_count")
        except Exception:
            # Startup snapshots are best-effort. The normal per-step info path
            # below will provide a candidate as soon as the first action runs.
            return

        for index, party in enumerate(snapshots):
            if not isinstance(party, (list, tuple)):
                continue
            info = {
                "party_snapshot": party,
                "best_map_progress": best_progress[index],
                "best_map_id": best_map_ids[index],
                "map_progress": map_progress[index],
                "map_id": map_ids[index] if index < len(map_ids) else -1,
                "badges": badges[index],
                "pokedex_owned": dex[index],
                "captures": captures[index],
                "hm_critical_ready": hm_ready[index],
                "hm_moves_taught": hm_moves[index],
                "hm_critical_moves_taught": hm_critical_moves[index],
                "party_species_count": party_counts[index],
            }
            self._consider_party_candidate(index, info, int(self.num_timesteps))

    def _save_persisted_history(self) -> None:
        if self.worker_maps is not None:
            temp_path = self.maps_path.with_suffix(".npz.tmp")
            with temp_path.open("wb") as handle:
                np.savez_compressed(handle, worker_maps=self.worker_maps)
            os.replace(temp_path, self.maps_path)

        metadata = {
            "best_map_progress": int(self.best_map_progress),
            "best_map_id": int(self.best_map_id),
            "episodes_completed": int(self.episodes_completed),
            "event_flags": self.event_flags,
        }
        temp_meta = self.metadata_path.with_suffix(".json.tmp")
        temp_meta.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temp_meta, self.metadata_path)

    def _warn(self, message: str) -> None:
        # Use plain stdout so this remains compatible with SB3 logger versions
        # that do not expose a Logger.warn() method.
        print(f"[TensorboardCallback] WARNING: {message}")

    def _get_infos_and_dones(self):
        infos = self.locals.get("infos", ())
        dones = self.locals.get("dones", ())
        if infos is None:
            infos = ()
        if dones is None:
            dones = ()
        try:
            dones = np.asarray(dones, dtype=bool).reshape(-1)
        except Exception:
            dones = np.zeros(len(infos), dtype=bool)
        return list(infos), dones

    def _update_maps_from_envs(self, infos, dones):
        n_envs = int(self.training_env.num_envs)
        if self.worker_maps is None:
            try:
                raw = self.training_env.get_attr("lifetime_explore_map")
                first = _as_uint8_map(raw[0])
                if first is None:
                    return []
                self.map_shape = tuple(first.shape)
                self.worker_maps = np.zeros((n_envs, *self.map_shape), dtype=np.uint8)
            except Exception as exc:
                self._warn(f"Environment does not expose lifetime_explore_map: {exc}")
                return []

        episode_maps = []
        snapshots_seen = 0
        for index in range(min(n_envs, len(infos), len(dones))):
            info = infos[index] if isinstance(infos[index], dict) else {}
            worker_best = info.get("best_map_progress")
            if isinstance(worker_best, (int, float, np.number)) and int(worker_best) > self.best_map_progress:
                self.best_map_progress = int(worker_best)
                self.best_map_id = int(info.get("best_map_id", info.get("map_id", -1)))
            if not dones[index]:
                continue
            snapshot = info.get("episode_snapshot")
            if not isinstance(snapshot, dict):
                continue
            snapshots_seen += 1

            episode_map = _as_uint8_map(snapshot.get("explore_map"), self.map_shape)
            lifetime_map = _as_uint8_map(snapshot.get("lifetime_explore_map"), self.map_shape)
            if episode_map is not None:
                episode_maps.append(episode_map)
            if lifetime_map is not None:
                self.worker_maps[index] = np.maximum(self.worker_maps[index], lifetime_map)
            elif episode_map is not None:
                self.worker_maps[index] = np.maximum(self.worker_maps[index], episode_map)

            stats = snapshot.get("agent_stats")
            if isinstance(stats, dict) and stats:
                self._completed_episode_stats.append(stats)

            progress = int(snapshot.get("best_map_progress", snapshot.get("map_progress", -1)))
            if progress > self.best_map_progress:
                self.best_map_progress = progress
                self.best_map_id = int(snapshot.get("best_map_id", snapshot.get("map_id", -1)))

            self.episodes_completed += 1
            flags = snapshot.get("lifetime_event_flags", snapshot.get("event_flags", {}))
            if isinstance(flags, dict):
                self.event_flags.update({str(k): str(v) for k, v in flags.items()})

        # Pull current cumulative maps from all workers. A completed worker has
        # already auto-reset, but its lifetime map intentionally survives reset.
        try:
            current_maps = self.training_env.get_attr("lifetime_explore_map")
            for index, raw_map in enumerate(current_maps[:n_envs]):
                current_map = _as_uint8_map(raw_map, self.map_shape)
                if current_map is not None:
                    self.worker_maps[index] = np.maximum(self.worker_maps[index], current_map)
        except Exception as exc:
            self._warn(f"Could not read lifetime maps from workers ({exc}).")

        # Fallback for a partially upgraded environment: at least collect named
        # event flags that are available after the VecEnv step.
        try:
            worker_flags = self.training_env.get_attr("lifetime_event_flags")
            for flags in worker_flags:
                if isinstance(flags, dict):
                    self.event_flags.update({str(k): str(v) for k, v in flags.items()})
        except Exception:
            pass

        if dones.size and np.any(dones) and snapshots_seen == 0 and not self._warned_missing_snapshot:
            self._warn(
                "An episode ended without info['episode_snapshot']. Replace red_gym_env_v2.py "
                "with the paired update; otherwise SB3's auto-reset can erase terminal maps before logging."
            )
            self._warned_missing_snapshot = True

        return episode_maps

    def _log_completed_episode_stats(self) -> None:
        if not self._completed_episode_stats:
            return
        means, distributions = merge_dicts(self._completed_episode_stats)
        for key, value in means.items():
            self.logger.record(f"env_stats/{key}", value)
        if self.writer is not None:
            for key, values in distributions.items():
                self.writer.add_histogram(
                    f"env_stats_distribs/{key}", values, self.num_timesteps
                )
                self.logger.record(f"env_stats_max/{key}", float(np.max(values)))
        self._completed_episode_stats.clear()

    def _log_map_images(self, episode_maps):
        if self.worker_maps is None or self.worker_maps.size == 0:
            return

        cumulative_sum = np.max(self.worker_maps, axis=0)
        cumulative_mosaic = _make_map_mosaic(self.worker_maps)
        cumulative_sum = _nearest_neighbor_scale(cumulative_sum, self.map_image_scale)
        cumulative_mosaic = _nearest_neighbor_scale(cumulative_mosaic, self.map_image_scale)

        # Preserve the original TensorBoard image tags, now showing lifetime
        # exploration rather than appearing to lose progress after each reset.
        self.logger.record(
            "trajectory/explore_sum",
            Image(cumulative_sum, "HW"),
            exclude=("stdout", "log", "json", "csv"),
        )
        self.logger.record(
            "trajectory/explore_map",
            Image(cumulative_mosaic, "HW"),
            exclude=("stdout", "log", "json", "csv"),
        )

        # Also keep a separate view of just the episode(s) that ended on this
        # VecEnv step. This makes a high-water episode easy to distinguish from
        # the all-time cumulative map.
        if episode_maps:
            episode_array = np.stack(episode_maps, axis=0).astype(np.uint8, copy=False)
            episode_sum = np.max(episode_array, axis=0)
            episode_mosaic = _make_map_mosaic(episode_array)
            episode_sum = _nearest_neighbor_scale(episode_sum, self.map_image_scale)
            episode_mosaic = _nearest_neighbor_scale(episode_mosaic, self.map_image_scale)
            self.logger.record(
                "trajectory/episode_explore_sum",
                Image(episode_sum, "HW"),
                exclude=("stdout", "log", "json", "csv"),
            )
            self.logger.record(
                "trajectory/episode_explore_map",
                Image(episode_mosaic, "HW"),
                exclude=("stdout", "log", "json", "csv"),
            )

    def _log_progress_scalars(self, infos):
        if self.num_timesteps - self._last_scalar_log_timestep < self.scalar_log_every:
            return
        self._last_scalar_log_timestep = int(self.num_timesteps)
        progress_values = []
        best_values = []
        seen_values = []
        capture_values = []
        novel_capture_values = []
        duplicate_capture_values = []
        unclassified_capture_values = []
        party_duplicate_values = []
        lead_hp_values = []
        lead_hp_danger_values = []
        lead_hp_critical_values = []
        critical_lead_steps_values = []
        lead_faint_values = []
        center_visit_values = []
        center_low_hp_visit_values = []
        center_heal_values = []
        center_hp_restored_values = []
        novel_capture_reward_values = []
        duplicate_capture_penalty_values = []
        party_duplicate_penalty_values = []
        critical_hp_exposure_penalty_values = []
        lead_faint_penalty_values = []
        center_return_reward_values = []
        center_heal_reward_values = []
        dex_values = []
        party_values = []
        party_size_values = []
        party_fill_values = []
        party_balance_eligible_values = []
        party_diversity_values = []
        party_balance_values = []
        capture_marker_values = []
        capture_pending_values = []
        pending_dex_values = []
        dex_delta_values = []
        party_spread_values = []
        party_spread_target_values = []
        party_spread_excess_values = []
        party_spread_within_target_values = []
        ball_values = []
        new_dex_values = []
        lifetime_dex_values = []
        starter_bonus_values = []
        hm_scalar_keys = (
            "hm_party_compatible", "hm_items_owned", "hm_moves_taught", "hm_field_ready",
            "hm_nonstarter_candidates", "hm_role_candidates", "hm_nonstarter_taught",
            "hm_role_taught", "hm_preferred_carrier_ready", "hm_cut_gate_entries",
            "hm_surf_route_entries", "hm_strength_boulder_moves",
        ) + tuple(
            f"hm_{name}_{suffix}"
            for name in ("cut", "fly", "surf", "strength", "flash")
            for suffix in (
                "party_compatible", "item_owned", "taught", "badge", "usable",
                "nonstarter_candidate", "role_candidate", "nonstarter_taught",
                "role_taught", "preferred_carrier_ready",
            )
        )
        hm_scalar_values = {key: [] for key in hm_scalar_keys}
        for info in infos:
            if not isinstance(info, dict):
                continue
            for target, key in (
                (progress_values, "map_progress"),
                (best_values, "best_map_progress"),
                (seen_values, "seen_tiles"),
            ):
                value = info.get(key)
                if isinstance(value, (int, float, np.number)):
                    target.append(float(value))
            for target, key in (
                (capture_values, "captures"),
                (novel_capture_values, "novel_captures"),
                (duplicate_capture_values, "duplicate_captures"),
                (unclassified_capture_values, "unclassified_captures"),
                (party_duplicate_values, "party_duplicate_count"),
                (lead_hp_values, "lead_hp_fraction"),
                (lead_hp_danger_values, "lead_hp_danger"),
                (lead_hp_critical_values, "lead_hp_critical"),
                (critical_lead_steps_values, "critical_lead_hp_steps"),
                (lead_faint_values, "lead_faints"),
                (center_visit_values, "center_visits"),
                (center_low_hp_visit_values, "center_low_hp_visits"),
                (center_heal_values, "center_heals"),
                (center_hp_restored_values, "center_hp_restored"),
                (novel_capture_reward_values, "novel_capture_reward_score"),
                (duplicate_capture_penalty_values, "duplicate_capture_penalty_score"),
                (party_duplicate_penalty_values, "party_duplicate_penalty_score"),
                (critical_hp_exposure_penalty_values, "critical_hp_exposure_penalty_score"),
                (lead_faint_penalty_values, "lead_faint_penalty_score"),
                (center_return_reward_values, "center_return_reward_score"),
                (center_heal_reward_values, "center_heal_reward_score"),
                (dex_values, "pokedex_owned"),
                (party_values, "party_species_count"),
                (party_size_values, "party_size"),
                (party_fill_values, "party_fill_score"),
                (party_balance_eligible_values, "party_level_balance_eligible"),
                (party_diversity_values, "party_diversity_score"),
                (party_balance_values, "party_level_balance_score"),
                (capture_marker_values, "capture_marker_active"),
                (capture_pending_values, "capture_pending_steps"),
                (pending_dex_values, "pending_dex_entries"),
                (dex_delta_values, "pokedex_owned_delta_current"),
                (party_spread_values, "party_level_spread"),
                (party_spread_target_values, "party_level_spread_target"),
                (party_spread_excess_values, "party_level_spread_excess"),
                (party_spread_within_target_values, "party_level_spread_within_target"),
                (ball_values, "pokeballs"),
                (new_dex_values, "new_dex_entries"),
                (lifetime_dex_values, "lifetime_best_owned_count"),
                (starter_bonus_values, "starter_diversity_reward"),
            ):
                value = info.get(key)
                if isinstance(value, (int, float, np.number)):
                    target.append(float(value))
            for key in hm_scalar_keys:
                value = info.get(key)
                if isinstance(value, (int, float, np.number)):
                    hm_scalar_values[key].append(float(value))
            worker_best = info.get("best_map_progress")
            if isinstance(worker_best, (int, float, np.number)) and int(worker_best) > self.best_map_progress:
                self.best_map_progress = int(worker_best)
                self.best_map_id = int(info.get("best_map_id", info.get("map_id", -1)))
        if progress_values:
            self.logger.record("progress/current_map_progress_mean", float(np.mean(progress_values)))
        if best_values:
            self.logger.record("progress/worker_best_map_progress_mean", float(np.mean(best_values)))
        if seen_values:
            self.logger.record("progress/current_episode_seen_tiles_mean", float(np.mean(seen_values)))
        if self.best_map_progress >= 0:
            self.logger.record("progress/best_ever_map_progress", float(self.best_map_progress))
        self.logger.record("progress/episodes_completed_total", float(self.episodes_completed))
        if capture_values:
            self.logger.record("game/captures_mean", float(np.mean(capture_values)))
        if novel_capture_values:
            self.logger.record("game/novel_captures_mean", float(np.mean(novel_capture_values)))
        if duplicate_capture_values:
            self.logger.record("game/duplicate_captures_mean", float(np.mean(duplicate_capture_values)))
        if unclassified_capture_values:
            self.logger.record("game/unclassified_captures_mean", float(np.mean(unclassified_capture_values)))
        if party_duplicate_values:
            self.logger.record("game/party_duplicate_slots_mean", float(np.mean(party_duplicate_values)))
        if lead_hp_values:
            self.logger.record("healing/lead_hp_fraction_mean", float(np.mean(lead_hp_values)))
        if lead_hp_danger_values:
            self.logger.record("healing/lead_hp_danger_mean", float(np.mean(lead_hp_danger_values)))
        if lead_hp_critical_values:
            self.logger.record("healing/lead_hp_critical_fraction", float(np.mean(lead_hp_critical_values)))
        if critical_lead_steps_values:
            self.logger.record("healing/critical_lead_hp_steps_mean", float(np.mean(critical_lead_steps_values)))
        if lead_faint_values:
            self.logger.record("healing/lead_faints_mean", float(np.mean(lead_faint_values)))
        if center_visit_values:
            self.logger.record("healing/center_visits_mean", float(np.mean(center_visit_values)))
        if center_low_hp_visit_values:
            self.logger.record("healing/center_low_hp_visits_mean", float(np.mean(center_low_hp_visit_values)))
        if center_heal_values:
            self.logger.record("healing/center_heals_mean", float(np.mean(center_heal_values)))
        if center_hp_restored_values:
            self.logger.record("healing/center_hp_restored_mean", float(np.mean(center_hp_restored_values)))
        for values, tag in (
            (novel_capture_reward_values, "rewards/novel_capture_score"),
            (duplicate_capture_penalty_values, "rewards/duplicate_capture_penalty_score"),
            (party_duplicate_penalty_values, "rewards/party_duplicate_penalty_score"),
            (critical_hp_exposure_penalty_values, "rewards/critical_hp_exposure_penalty_score"),
            (lead_faint_penalty_values, "rewards/lead_faint_penalty_score"),
            (center_return_reward_values, "rewards/center_return_score"),
            (center_heal_reward_values, "rewards/center_heal_score"),
        ):
            if values:
                self.logger.record(tag, float(np.mean(values)))
        if dex_values:
            self.logger.record("game/pokedex_owned_mean", float(np.mean(dex_values)))
        if party_values:
            self.logger.record("game/party_species_count_mean", float(np.mean(party_values)))
        if party_size_values:
            self.logger.record("game/party_size_mean", float(np.mean(party_size_values)))
        if party_fill_values:
            self.logger.record("game/party_fill_score_mean", float(np.mean(party_fill_values)))
        if party_balance_eligible_values:
            self.logger.record("game/party_level_balance_eligible_mean", float(np.mean(party_balance_eligible_values)))
        if capture_marker_values:
            self.logger.record("game/capture_marker_active_mean", float(np.mean(capture_marker_values)))
        if capture_pending_values:
            self.logger.record("game/capture_pending_steps_mean", float(np.mean(capture_pending_values)))
        if pending_dex_values:
            self.logger.record("game/pending_dex_entries_mean", float(np.mean(pending_dex_values)))
        if dex_delta_values:
            self.logger.record("game/pokedex_owned_delta_current_mean", float(np.mean(dex_delta_values)))
        if party_diversity_values:
            self.logger.record("game/party_diversity_score_mean", float(np.mean(party_diversity_values)))
        if party_balance_values:
            self.logger.record("game/party_level_balance_score_mean", float(np.mean(party_balance_values)))
        if party_spread_values:
            self.logger.record("game/party_level_spread_mean", float(np.mean(party_spread_values)))
        if party_spread_target_values:
            self.logger.record("game/party_level_spread_target_mean", float(np.mean(party_spread_target_values)))
        if party_spread_excess_values:
            self.logger.record("game/party_level_spread_excess_mean", float(np.mean(party_spread_excess_values)))
        if party_spread_within_target_values:
            self.logger.record("game/party_spread_within_target_mean", float(np.mean(party_spread_within_target_values)))
        if ball_values:
            self.logger.record("game/pokeballs_mean", float(np.mean(ball_values)))
        if new_dex_values:
            self.logger.record("game/new_dex_entries_mean", float(np.mean(new_dex_values)))
        if lifetime_dex_values:
            self.logger.record("game/lifetime_best_owned_count_mean", float(np.mean(lifetime_dex_values)))
        if starter_bonus_values:
            self.logger.record(
                "game/starter_diversity_reward_mean",
                float(np.mean(starter_bonus_values)),
            )
        for key, values in hm_scalar_values.items():
            if values:
                tag = f"game/{key}_mean"
                mean_value = float(np.mean(values))
                # Long tags remain fully available in TensorBoard but are
                # excluded from SB3's narrow stdout table to avoid truncation collisions.
                if len(tag) > 36:
                    self.logger.record(tag, mean_value, exclude=("stdout", "log"))
                else:
                    self.logger.record(tag, mean_value)

        # All subprocess workers share the same SQLite count file. Read it only
        # on the scalar-log interval, not on every emulator step.
        try:
            count_rows = self.training_env.env_method(
                "get_starter_choice_counts", indices=[0]
            )
            if count_rows and isinstance(count_rows[0], dict):
                for species in ("Bulbasaur", "Charmander", "Squirtle"):
                    self.logger.record(
                        f"game/starter_choices/{species.lower()}",
                        float(count_rows[0].get(species, 0)),
                    )
        except Exception:
            # Older environment copies do not expose this optional diagnostic.
            pass

    @staticmethod
    def _party_font(size: int, bold: bool = False):
        candidates = (
            ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
             "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf")
            if bold else
            ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
             "/usr/share/fonts/TTF/DejaVuSans.ttf")
        )
        for candidate in candidates:
            if Path(candidate).is_file():
                try:
                    return ImageFont.truetype(candidate, size=size)
                except OSError:
                    pass
        return ImageFont.load_default()

    @staticmethod
    def _party_hm_text(values):
        order = ("CUT", "FLY", "SURF", "STRENGTH", "FLASH")
        names = {str(value).upper() for value in (values or [])}
        return " / ".join(name for name in order if name in names) or "—"

    @staticmethod
    def _party_hp_fraction(mon):
        try:
            hp = max(0.0, float(mon.get("hp", 0)))
            max_hp = max(1.0, float(mon.get("max_hp", 1)))
            return max(0.0, min(1.0, hp / max_hp))
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _render_party_board(cls, party, env_index: int, step: int, progress_meta=None):
        """Build a visible 6-slot party image; no external sprite files required."""
        width, height = 1240, 760
        image = PILImage.new("RGB", (width, height), (17, 24, 39))
        draw = ImageDraw.Draw(image)
        title_font = cls._party_font(30, bold=True)
        header_font = cls._party_font(19, bold=True)
        body_font = cls._party_font(17)
        small_font = cls._party_font(14)

        heading = (
            f"MOST ADVANCED PARTY  |  WORKER {env_index:02d}"
            if progress_meta is not None else f"POKEMON PARTY  |  ENV {env_index:02d}"
        )
        draw.text((28, 20), heading, font=title_font, fill=(243, 244, 246))
        if progress_meta is None:
            subheading = f"Training timestep: {step:,}    |    {len(party)} / 6 party members"
        else:
            subheading = (
                f"Best story-map progress: {progress_meta.get('best_map_progress', -1)}"
                f"  |  Map: {progress_meta.get('best_map_id', -1)}"
                f"  |  Badges: {progress_meta.get('badges', 0)}"
                f"  |  Dex: {progress_meta.get('pokedex_owned', 0)}"
                f"  |  Captures: {progress_meta.get('captures', 0)}"
                f"  |  Saved at step: {progress_meta.get('timestep', step):,}"
            )
        draw.text((30, 61), subheading[:125], font=small_font, fill=(156, 163, 175))
        if progress_meta is not None:
            team_line = (
                f"Unique species: {progress_meta.get('party_species_count', len(party))}/6"
                f"  |  Level balance: {float(progress_meta.get('party_level_balance_score', 0.0)):.0%}"
                f"  |  Spread: {progress_meta.get('party_level_spread', 0)}"
                f"/{progress_meta.get('party_level_spread_target', 10)} max"
            )
            draw.text((30, 82), team_line, font=small_font, fill=(134, 239, 172))

        card_w, card_h, gap_x, gap_y = 386, 306, 18, 18
        start_x, start_y = 28, 102
        for slot in range(6):
            row, col = divmod(slot, 3)
            x = start_x + col * (card_w + gap_x)
            y = start_y + row * (card_h + gap_y)
            draw.rounded_rectangle(
                (x, y, x + card_w, y + card_h), radius=18,
                fill=(31, 41, 55), outline=(75, 85, 99), width=2,
            )
            draw.text((x + 18, y + 15), f"SLOT {slot + 1}",
                      font=small_font, fill=(156, 163, 175))
            if slot >= len(party):
                draw.rounded_rectangle(
                    (x + 22, y + 58, x + 122, y + 158), radius=16,
                    fill=(24, 32, 45), outline=(75, 85, 99), width=2,
                )
                draw.ellipse((x + 52, y + 88, x + 114, y + 150),
                             outline=(107, 114, 128), width=5)
                draw.line((x + 57, y + 119, x + 109, y + 119),
                          fill=(107, 114, 128), width=5)
                draw.text((x + 143, y + 81), "EMPTY SLOT",
                          font=header_font, fill=(156, 163, 175))
                draw.text((x + 143, y + 113), "Available for a teammate",
                          font=small_font, fill=(107, 114, 128))
                continue

            mon = party[slot] if isinstance(party[slot], dict) else {}
            # Use installed Gen-I sprites when present; fallback to a clear
            # species label if the optional sprite pack has not been downloaded.
            try:
                species_index = int(mon.get("species_index", 0))
            except (TypeError, ValueError):
                species_index = 0
            dex_value = mon.get("national_dex") or INTERNAL_TO_NATIONAL_DEX.get(species_index)
            try:
                dex = int(dex_value) if dex_value is not None else None
            except (TypeError, ValueError):
                dex = None
            if dex is not None and not (1 <= dex <= 151):
                dex = None
            sprite_path = Path(__file__).resolve().parent / "assets" / "gen1_sprites" / f"{dex}.png" if dex else None
            sprite_box = (x + 18, y + 45, x + 132, y + 159)
            draw.rounded_rectangle(sprite_box, radius=14,
                                   fill=(24, 32, 45), outline=(55, 65, 81), width=1)
            sprite = None
            if sprite_path is not None and sprite_path.is_file():
                try:
                    with PILImage.open(sprite_path) as source:
                        sprite = source.convert("RGBA")
                        sprite.thumbnail((106, 106), PILImage.Resampling.NEAREST)
                except (OSError, ValueError):
                    sprite = None
            if sprite is not None:
                px = sprite_box[0] + (114 - sprite.width) // 2
                py = sprite_box[1] + (114 - sprite.height) // 2
                image.paste(sprite, (px, py), sprite)
            else:
                label = f"#{int(dex):03d}" if dex else f"ID {int(mon.get('species_index', 0)):02X}"
                draw.ellipse((x + 47, y + 70, x + 103, y + 126),
                             fill=(220, 228, 236), outline=(148, 163, 184), width=3)
                draw.ellipse((x + 62, y + 85, x + 88, y + 111),
                             fill=(31, 41, 55), outline=(148, 163, 184), width=2)
                draw.text((x + 40, y + 132), label, font=small_font,
                          fill=(203, 213, 225))

            name = str(mon.get("species", "Unknown")).replace("Nidoran♂", "Nidoran-M").replace("Nidoran♀", "Nidoran-F")
            draw.text((x + 146, y + 49), name[:22], font=header_font, fill=(243, 244, 246))
            draw.text((x + 146, y + 80), f"Lv. {mon.get('level', '?')}",
                      font=body_font, fill=(203, 213, 225))
            status = str(mon.get("status", "Unknown"))
            draw.text((x + 146, y + 110), status[:20], font=small_font,
                      fill=(74, 222, 128) if status == "OK" else (251, 191, 36))
            identity = "STARTER" if mon.get("is_starter", species_index in LEGACY_STARTER_INTERNAL_IDS) else "TEAMMATE"
            type_values = mon.get("types") or LEGACY_TYPE_NAMES_BY_INTERNAL_ID.get(species_index, ())
            types = "/".join(str(value) for value in type_values) or "Unknown type"
            draw.text((x + 146, y + 130), f"{identity} | {types}"[:34],
                      font=small_font, fill=(253, 224, 71) if identity == "STARTER" else (147, 197, 253))
            preferred_roles = cls._party_hm_text(
                mon.get("hm_preferred_carrier_for")
                or LEGACY_HM_PREFERRED_CARRIERS_BY_INTERNAL_ID.get(species_index, ())
            )
            draw.text((x + 146, y + 148), f"HM ROLE: {preferred_roles}"[:34],
                      font=small_font, fill=(196, 181, 253))

            hp = mon.get("hp", "?")
            max_hp = mon.get("max_hp", "?")
            draw.text((x + 18, y + 171), f"HP  {hp}/{max_hp}",
                      font=small_font, fill=(226, 232, 240))
            bar = (x + 18, y + 194, x + card_w - 18, y + 209)
            draw.rounded_rectangle(bar, radius=7, fill=(55, 65, 81))
            fraction = cls._party_hp_fraction(mon)
            if fraction > 0:
                color = (34, 197, 94) if fraction > 0.5 else ((234, 179, 8) if fraction > 0.2 else (239, 68, 68))
                draw.rounded_rectangle(
                    (bar[0], bar[1], bar[0] + int((bar[2] - bar[0]) * fraction), bar[3]),
                    radius=7, fill=color,
                )

            moves = [str(move) for move in mon.get("moves", [])[:4]]
            draw.text((x + 18, y + 220), "MOVES", font=small_font, fill=(156, 163, 175))
            draw.text((x + 18, y + 241), (" / ".join(moves) or "No moves")[:47],
                      font=small_font, fill=(226, 232, 240))
            can_learn = cls._party_hm_text(mon.get("hm_can_learn", []))
            knows = cls._party_hm_text(mon.get("hm_knows", []))
            draw.text((x + 18, y + 266), f"CAN LEARN: {can_learn}"[:47],
                      font=small_font, fill=(147, 197, 253))
            draw.text((x + 18, y + 286), f"KNOWS: {knows}"[:47],
                      font=small_font, fill=(134, 239, 172))

        return np.asarray(image, dtype=np.uint8).copy()

    @staticmethod
    def _compose_gameplay_party(game_frame, party_board):
        game = PILImage.fromarray(np.asarray(game_frame, dtype=np.uint8)).convert("RGB")
        board = PILImage.fromarray(np.asarray(party_board, dtype=np.uint8)).convert("RGB")
        target_h = 760
        game_w = max(1, int(game.width * target_h / max(game.height, 1)))
        game = game.resize((game_w, target_h), PILImage.Resampling.NEAREST)
        canvas = PILImage.new("RGB", (game_w + board.width, target_h), (12, 18, 28))
        canvas.paste(game, (0, 0))
        canvas.paste(board, (game_w, 0))
        return np.asarray(canvas, dtype=np.uint8).copy()

    def _log_party_images(self):
        """Log one party image: the party belonging to the most advanced worker."""
        if self.training_env is None or self.logger is None:
            return 0
        if self.best_party_record is None:
            self._prime_best_party_from_envs()
        record = self.best_party_record
        if not isinstance(record, dict):
            return 0

        party = record.get("party_snapshot")
        if not isinstance(party, (list, tuple)):
            return 0

        worker_index = int(record.get("worker_index", -1))
        board = self._render_party_board(
            party,
            worker_index,
            int(record.get("timestep", self.num_timesteps)),
            progress_meta=record,
        )
        # The env collection intentionally contains ONE image: the party
        # snapshot selected by _consider_party_candidate using best_map_progress
        # as its primary ranking key. It is not a grid or one image per worker.
        self.logger.record(
            "game/party_visual/env",
            Image(board, "HWC"),
            exclude=("stdout", "log", "json", "csv"),
        )
        # Keep the earlier tag as an identical compatibility alias. Both tags
        # point to the same single best-progression worker, never all workers.
        self.logger.record(
            "game/party_visual/most_advanced",
            Image(board, "HWC"),
            exclude=("stdout", "log", "json", "csv"),
        )
        # These scalar tags make it possible to track which worker snapshot is
        # currently represented by the single most-advanced party image.
        self.logger.record("progress/most_advanced_worker_index", float(worker_index))
        self.logger.record("progress/most_advanced_map_progress", float(record.get("best_map_progress", -1)))
        self.logger.record("progress/most_advanced_badges", float(record.get("badges", 0)))
        self.logger.record("progress/most_advanced_pokedex_owned", float(record.get("pokedex_owned", 0)))
        self.logger.record("progress/most_advanced_captures", float(record.get("captures", 0)))
        self.logger.record("game/most_advanced_party_size", float(record.get("party_size", len(party))))
        self.logger.record("game/most_advanced_party_fill_score", float(record.get("party_fill_score", min(len(party), 6) / 6.0)))
        self.logger.record("game/most_advanced_party_diversity_score", float(record.get("party_diversity_score", 0.0)))
        self.logger.record("game/most_advanced_party_level_balance_eligible", float(record.get("party_level_balance_eligible", 0)))
        self.logger.record("game/most_advanced_party_level_balance_score", float(record.get("party_level_balance_score", 0.0)))
        self.logger.record("game/most_advanced_party_level_spread", float(record.get("party_level_spread", 0)))
        self.logger.record("game/most_advanced_party_level_spread_excess", float(record.get("party_level_spread_excess", 0)))
        self.logger.record("game/most_advanced_party_spread_within_target", float(record.get("party_level_spread_within_target", 0)))
        self._best_party_dirty = False
        if self.verbose:
            print(
                "[TensorboardCallback] Logged most-advanced party: "
                f"worker={worker_index}, story_progress={record.get('best_map_progress')}, "
                f"party={len(party)}/6, tags=game/party_visual/env + most_advanced"
            )
        return 1

    def _update_best_party_from_infos(self, infos, dones) -> bool:
        changed = False
        for index, raw_info in enumerate(infos):
            if not isinstance(raw_info, dict):
                continue
            terminal = bool(index < len(dones) and dones[index])
            changed = self._consider_party_candidate(
                index, raw_info, int(self.num_timesteps), terminal=terminal
            ) or changed
        return changed

    def _on_step(self) -> bool:
        infos, dones = self._get_infos_and_dones()
        party_improved = self._update_best_party_from_infos(infos, dones)
        self._log_progress_scalars(infos)

        any_done = bool(dones.size and np.any(dones))
        map_due = self.num_timesteps - self._last_map_log_timestep >= self.map_log_every
        if any_done or map_due:
            self._completed_episode_stats = []
            episode_maps = self._update_maps_from_envs(infos, dones)
            if any_done:
                self._log_completed_episode_stats()

            if self.best_map_progress >= 0:
                self.logger.record("progress/best_ever_map_progress", float(self.best_map_progress))
                self.logger.record("progress/best_map_id", float(self.best_map_id))
            self.logger.record("progress/episodes_completed_total", float(self.episodes_completed))
            self.logger.record("trajectory/all_flags", json.dumps(self.event_flags, sort_keys=True))

            self._log_map_images(episode_maps)
            self._save_persisted_history()
            self._last_map_log_timestep = int(self.num_timesteps)

        if party_improved or self.num_timesteps - self._last_party_image_timestep >= self.party_image_every:
            self._log_party_images()
            self._last_party_image_timestep = int(self.num_timesteps)

        return True

    def _on_training_end(self) -> None:
        # Save current lifetime maps even if training stopped before any worker
        # reached its episode limit. This prevents a partial episode from being
        # lost when the run is resumed later.
        try:
            self._completed_episode_stats = []
            self._update_maps_from_envs([], np.zeros(int(self.training_env.num_envs), dtype=bool))
            self._save_persisted_history()
        except Exception as exc:
            self._warn(f"Could not save final exploration history ({exc}).")

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None
