# v0.2.0 — Lifetime exploration reward increase

## Changes
- Changes tile novelty from episode-local to worker-lifetime accounting.
- Tracks each worker's best story-map rank across episode resets.
- Increases the explicit exploration weight from 0.25 to 2.0 (8×), while keeping reward scale and PPO/action/observation structure unchanged.

## Map progress from feedback
At approximately **47.33 million aggregate timesteps**, best reported story progress remained **rank 5 — Viridian Forest**. The run reported around **150 tiles per episode**, but had not advanced the best milestone to Pewter City. Exploration pressure increased without a demonstrated story-progression breakthrough.
