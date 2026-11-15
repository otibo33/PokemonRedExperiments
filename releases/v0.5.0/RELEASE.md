# v0.5.0 — Vermilion Gym progression and per-worker stagnation handling

## Changes
- Extends the story-map objective ladder toward Vermilion City, the S.S. Anne, and Vermilion Gym (map ID 92).
- Tracks directed map transitions per worker across episode resets.
- Rewards first-time directed transitions and forward story progression while discouraging repeated edge oscillation.
- Adds a one-time Vermilion Gym arrival signal.
- Resets only an individual worker after 12,288 steps without meaningful progress; battle time is excluded.
- Adds TensorBoard measures for unique/repeated transitions, forward transitions, steps since progress, stagnation resets, and Gym arrival.
- Keeps the original PPO network and action/observation schema structurally compatible with existing checkpoints.

## Map progress from feedback
At approximately **58.49 million aggregate timesteps**, best story progress was still **rank 5 — Viridian Forest**; **Vermilion Gym had not been reached**. Reported means were **4.07 unique maps**, **4.27 unique directed transitions**, **1.07 forward transitions**, and **126 repeated transitions**. Mean steps since progress were 128, below the 12,288-step threshold; stagnation resets remained zero. The run had not broken through to Pewter or Vermilion.
