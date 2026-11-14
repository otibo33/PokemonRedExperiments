"""TensorBoard callback for metrics returned by RedGymEnv.step()."""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class GameStatsCallback(BaseCallback):
    """Log averaged game metrics every ``every`` environment timesteps.

    Works with Stable-Baselines3 vector environments. Unlike counting
    ``n_calls``, ``num_timesteps`` accounts for all parallel environments.
    """

    KEYS = (
        "map_progress",
        "seen_tiles",
        "visited_maps",
        "badges",
        "events",
        "levels_sum",
        "deaths",
        "story_milestone_score",
        "captures",
        "novel_captures",
        "duplicate_captures",
        "unclassified_captures",
        "party_duplicate_count",
        "lead_hp_fraction",
        "lead_hp_danger",
        "lead_hp_critical",
        "critical_lead_hp_steps",
        "lead_faints",
        "current_map_is_pokecenter",
        "center_visits",
        "center_low_hp_visits",
        "center_heals",
        "center_hp_restored",
        "novel_capture_reward_score",
        "duplicate_capture_penalty_score",
        "party_duplicate_penalty_score",
        "critical_hp_exposure_penalty_score",
        "lead_faint_penalty_score",
        "center_return_reward_score",
        "center_heal_reward_score",
        "capture_marker_species",
        "capture_marker_active",
        "capture_pending_steps",
        "pending_dex_entries",
        "new_dex_entries",
        "pokedex_owned",
        "pokedex_owned_delta_current",
        "party_size",
        "party_fill_score",
        "party_species_count",
        "party_level_balance_eligible",
        "party_level_balance_score",
        "party_level_spread",
        "party_level_spread_excess",
        "party_level_spread_within_target",
        "pokeballs",
        "starter_choice_recorded",
        "starter_diversity_reward",
    )

    def __init__(self, every: int = 2560, verbose: int = 0):
        if every <= 0:
            raise ValueError("every must be a positive integer")
        super().__init__(verbose=verbose)
        self.every = int(every)
        self._last_log_timestep = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_log_timestep < self.every:
            return True

        self._last_log_timestep = self.num_timesteps
        infos = self.locals.get("infos", ())

        for key in self.KEYS:
            values = []
            for info in infos:
                if not isinstance(info, dict) or key not in info:
                    continue
                value = info[key]
                if isinstance(value, (int, float, np.number)):
                    values.append(float(value))

            if values:
                self.logger.record(f"game/{key}", float(np.mean(values)))

        return True
