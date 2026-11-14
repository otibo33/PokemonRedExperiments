from __future__ import annotations

import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from starter_diversity import STARTERS, StarterDiversityTracker


def test_underrepresented_starter_gets_larger_reward():
    with tempfile.TemporaryDirectory() as directory:
        tracker = StarterDiversityTracker(Path(directory) / "counts.sqlite3", 20.0)
        first_reward, counts = tracker.record_choice("Charmander")
        assert first_reward == 20.0
        assert counts == {"Bulbasaur": 0, "Charmander": 1, "Squirtle": 0}

        repeated_reward, counts = tracker.record_choice("Charmander")
        assert repeated_reward == 10.0
        assert counts["Charmander"] == 2

        bulbasaur_reward, counts = tracker.record_choice("Bulbasaur")
        assert bulbasaur_reward == 20.0
        assert counts["Bulbasaur"] == 1

        squirtle_reward, counts = tracker.record_choice("Squirtle")
        assert squirtle_reward == 20.0
        assert counts == {"Bulbasaur": 1, "Charmander": 2, "Squirtle": 1}


def test_counts_persist_when_tracker_is_recreated():
    with tempfile.TemporaryDirectory() as directory:
        db = Path(directory) / "counts.sqlite3"
        StarterDiversityTracker(db).record_choice("Squirtle")
        tracker = StarterDiversityTracker(db)
        assert tracker.get_counts() == {
            "Bulbasaur": 0,
            "Charmander": 0,
            "Squirtle": 1,
        }


def test_parallel_writes_are_not_lost():
    with tempfile.TemporaryDirectory() as directory:
        db = Path(directory) / "counts.sqlite3"
        tracker = StarterDiversityTracker(db)
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [
                pool.submit(StarterDiversityTracker(db).record_choice, STARTERS[i % 3])
                for i in range(60)
            ]
            for future in futures:
                future.result()
        counts = tracker.get_counts()
        assert sum(counts.values()) == 60
        assert counts == {"Bulbasaur": 20, "Charmander": 20, "Squirtle": 20}


def test_unknown_species_is_rejected():
    with tempfile.TemporaryDirectory() as directory:
        tracker = StarterDiversityTracker(Path(directory) / "counts.sqlite3")
        try:
            tracker.record_choice("Pikachu")
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid starter species should be rejected")


if __name__ == "__main__":
    tests = [
        test_underrepresented_starter_gets_larger_reward,
        test_counts_persist_when_tracker_is_recreated,
        test_parallel_writes_are_not_lost,
        test_unknown_species_is_rejected,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} starter-diversity tests passed")
