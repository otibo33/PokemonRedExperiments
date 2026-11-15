# v0.1.0 — Goal-conditioned humanlike training experiment

## Changes
- Adds a state-based objective supervisor for meaningful tasks such as obtaining a starter, healing, restocking, building a party, training, and pursuing the next badge.
- Adds semantic world-state inputs alongside the existing visual/map observations.
- Adds a per-worker stagnation watchdog that truncates only workers that stop making meaningful progress.
- Adds a separate PPO trainer and run directory so the original trainer, checkpoints, and training history remain intact.

## Compatibility and validation
This experiment adds two observation keys (objective and world_state), so checkpoints from the original policy are not compatible with this new policy. Bundled fake-environment tests cover objective selection and stuck-worker handling; this is not a live PyBoy/ROM validation.

## Map progress from subsequent feedback
This alternative path was not adopted as the continuing training line. Development returned to the original PPO v2 run. No completed in-game map milestone was reported for this experiment.
