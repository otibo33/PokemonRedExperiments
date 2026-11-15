# v0.3.0 — Directional lifetime exploration tracking

## Changes
- Makes lifetime unique-tile novelty persistent across ordinary worker episode resets.
- Tracks unique maps visited over a worker's lifetime and rewards first-time map discovery.
- Retains the existing map-progression reward rather than replacing it.
- Adds tests for the new exploration accounting.

## Map progress from feedback
At approximately **54.07 million aggregate timesteps**, mean lifetime coverage was about **175 unique tiles** and **4.33 unique maps per worker**. Best story-map progress was still **rank 5 — Viridian Forest**. Exploration tracking broadened, but no new story milestone was reported.
