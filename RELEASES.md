# Pokémon Red RL — iteration archive

This branch preserves five development iterations as separate, ordered snapshots. It is an archive only; it does not replace the project's main branch or modify existing training branches, checkpoints, or run data.

## Version history

### v0.1.0 — Goal-conditioned humanlike training experiment
Added objective selection from the current world state, semantic inputs, and per-worker stagnation handling. This was an alternative training path. Feedback then returned development to the original PPO v2 run; no completed map milestone was reported for this experiment.

### v0.2.0 — Lifetime exploration reward increase
Changed tile novelty to worker-lifetime accounting, tracked each worker's best story-map rank, and increased the explicit exploration weight from 0.25 to 2.0. At approximately **47.33 million aggregate timesteps**, best story progress remained **rank 5 — Viridian Forest**. The run reported roughly **150 tiles per episode**; Pewter City had not yet been reached.

### v0.3.0 — Directional lifetime exploration tracking
Added persistent lifetime tile/map novelty and first-visit map discovery rewards. At approximately **54.07 million aggregate timesteps**, mean lifetime coverage was about **175 unique tiles** and **4.33 unique maps per worker**. Best story progress remained **rank 5 — Viridian Forest**; no new story-map milestone was reported.

### v0.4.0 — Live party dashboard and navigation telemetry
Improved live party visualization, atomic party image/metadata updates, and exploration telemetry without changing PPO, the action space, or observation schema. This was an observability change, not a new gameplay policy. No new map milestone was attributed to it; best reported story progress remained **rank 5 — Viridian Forest**.

### v0.5.0 — Vermilion Gym progression and per-worker stagnation handling
Added directed map-transition tracking, forward-progression rewards, anti-oscillation signals, a Vermilion Gym arrival metric, and per-worker stagnation recovery. At approximately **58.49 million aggregate timesteps**, best story progress was still **rank 5 — Viridian Forest** and **Vermilion Gym had not been reached**. Reported means: **4.07 unique maps**, **4.27 unique directed transitions**, **1.07 forward transitions**, and **126 repeated transitions**. Mean steps since progress were 128, below the 12,288-step reset threshold; stagnation resets remained zero.

## Privacy and branch safety
Installation examples use generic project-relative paths. The patch archives were scanned for the local username, media-volume label, known mount path, and previously observed personal email pattern; no matches were found. This archive is on a separate branch created from the existing master head. The master branch was not moved or edited.
