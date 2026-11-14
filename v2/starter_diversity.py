"""Persistent population-level novelty rewards for Pokémon Red starters.

The module is intentionally independent of Gym/PyBoy so it can be used from
all SubprocVecEnv workers. SQLite serializes writes across worker processes.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, Tuple

STARTERS = ("Bulbasaur", "Charmander", "Squirtle")


class StarterDiversityTracker:
    """Track starter choice frequencies shared by all workers in one run dir.

    The least-selected starter gets the full configured reward. Repeating a
    starter that is already overrepresented gives a smaller reward:

        reward = weight * (least_used_count + 1) / (chosen_count + 1)

    The write and reward calculation happen in one IMMEDIATE transaction, so
    parallel workers cannot all receive the same stale novelty reward.
    """

    def __init__(self, db_path: str | Path, reward_weight: float = 20.0):
        self.db_path = Path(db_path).expanduser()
        self.reward_weight = float(reward_weight)
        if self.reward_weight < 0:
            raise ValueError("reward_weight must be >= 0")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path), timeout=30.0)
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS starter_counts ("
                "species TEXT PRIMARY KEY, count INTEGER NOT NULL CHECK(count >= 0)"
                ")"
            )
            connection.executemany(
                "INSERT OR IGNORE INTO starter_counts(species, count) VALUES (?, 0)",
                ((name,) for name in STARTERS),
            )

    @staticmethod
    def _read_counts(connection: sqlite3.Connection) -> Dict[str, int]:
        rows = connection.execute(
            "SELECT species, count FROM starter_counts"
        ).fetchall()
        counts = {name: int(count) for name, count in rows}
        return {name: counts.get(name, 0) for name in STARTERS}

    def get_counts(self) -> Dict[str, int]:
        with self._connect() as connection:
            return self._read_counts(connection)

    def record_choice(self, species: str) -> Tuple[float, Dict[str, int]]:
        """Atomically add one selection and return (novelty reward, new counts)."""
        if species not in STARTERS:
            raise ValueError(
                f"Unsupported starter {species!r}; expected one of {STARTERS}"
            )

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            counts_before = self._read_counts(connection)
            chosen_before = counts_before[species]
            least_used_before = min(counts_before.values())
            reward = self.reward_weight * (least_used_before + 1) / (chosen_before + 1)

            connection.execute(
                "UPDATE starter_counts SET count = count + 1 WHERE species = ?",
                (species,),
            )
            counts_after = self._read_counts(connection)
            connection.commit()

        return float(reward), counts_after
