# Pokémon Red v2 — latest tested environment snapshot

This branch layers the cumulative, locally developed v2 environment and telemetry updates onto the repository's existing `master` baseline. It does **not** replace `master` and it does not touch run data or checkpoints.

## What's included

- `red_gym_env_v2.py`: current cumulative environment update, including starter-choice diversity, party completeness and level-balance signals, capture/Pokédex accounting, HM awareness and field-readiness metrics, duplicate-capture/duplicate-party penalties, lead-Pokémon survival shaping, and Pokémon Center recovery signals.
- `tensorboard_callback.py`: cumulative TensorBoard metrics, persistent exploration/progression history, and the most-advanced-party dashboard with Gen-I sprite, type, starter, and HM-role metadata.
- `starter_diversity.py` and `gen1_party_metadata.py`: required helper modules used by the environment and callback.
- `game_stats_callback.py`: optional callback for direct `game/...` scalar logging. The richer `TensorboardCallback` already logs the primary environment, healing, reward, and party metrics.
- `migrate_most_advanced_party.py`: safe metadata backfill utility for an existing `_trajectory_history/most_advanced_party.json` file.
- Six focused test files covering capture/reward accounting, party balance, starter diversity, callback metrics, HM awareness, and party metadata/visualization.

The upstream `v2/baseline_fast_v2.py`, `global_map.py`, map data, requirements, and other original project files are left unchanged. The repository's training launcher is not replaced by this branch; use the launcher/configuration that matches your existing run.

## Validation performed for this snapshot

- `python -m py_compile *.py` — passed for all Python files in `v2/`.
- `test_capture_rewards.py` — passed.
- `test_party_balance.py` — passed.
- `test_starter_diversity.py` — passed.
- `test_callback_party_metrics.py` — passed.
- `test_hm_awareness_patch.py` — passed.
- `test_party_metadata_visual_fix.py` — passed.

The six test suites passed 52 test cases in total. These tests use isolated RAM/reward fixtures and callback stubs. A live PyBoy + ROM smoke test after checking out this branch has **not** been performed here; treat this as a unit-tested snapshot pending that local run.

## Important checkpoint note

The action space and observation dictionary keys, shapes, and dtypes are preserved by the environment update, so existing PPO checkpoints remain structurally loadable. The existing `health` observation retains its `(1,)` float32 shape but now represents the lead Pokémon's HP fraction instead of aggregate party HP. Resuming an old policy therefore means fine-tuning with a changed interpretation of that input, not behavior-identical continuation.

Keep your existing `init.state`, `runs/`, checkpoints, TensorBoard event files, and starter-choice database. Do not delete or reset them just to install this code snapshot.

## Run the tests

From the `v2/` directory, in the project's virtual environment:

```fish
python -m py_compile red_gym_env_v2.py tensorboard_callback.py game_stats_callback.py gen1_party_metadata.py starter_diversity.py migrate_most_advanced_party.py
python test_capture_rewards.py
python test_party_balance.py
python test_starter_diversity.py
python test_callback_party_metrics.py
python test_hm_awareness_patch.py
python test_party_metadata_visual_fix.py
```

For a live run, first stop the previous trainer normally, back up the two active Python modules and apply the branch. Reuse the existing session and checkpoint only after the local smoke test succeeds.
