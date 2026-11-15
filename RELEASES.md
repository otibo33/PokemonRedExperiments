# Pokémon Red RL — iteration archive

This branch preserves five development iterations as separate, ordered snapshots. It is an archive only; it does not replace the project's main branch or modify existing training branches, checkpoints, or run data.

## Version history

### v0.1.0 — Goal-conditioned humanlike training experiment
Added objective selection from the current world state, semantic inputs, and per-worker stagnation handling. This was an alternative training path. Feedback then returned development to the original PPO v2 run; **no completed map milestone was recorded**.

### v0.2.0 — Lifetime exploration reward increase
Changed tile novelty to worker-lifetime accounting, tracked each worker's best story-map rank, and increased the explicit exploration weight from 0.25 to 2.0. At approximately **47.33 million aggregate timesteps**, best story progress was **rank 5 — Viridian Forest (map ID 51)**. The run reported roughly **150 tiles per episode**; Pewter City had not yet been reached.

### v0.3.0 — Directional lifetime exploration tracking
Added persistent lifetime tile/map novelty and first-visit map discovery rewards. At approximately **54.07 million aggregate timesteps**, mean lifetime coverage was about **175 unique tiles** and **4.33 unique maps per worker**. Best story progress remained **rank 5 — Viridian Forest (map ID 51)**; no new story-map milestone was reported.

### v0.4.0 — Live party dashboard and navigation telemetry
Improved live party visualization, atomic party image/metadata updates, and exploration telemetry without changing PPO, the action space, or observation schema. This was an observability change, not a new gameplay policy. The surrounding telemetry still showed **Viridian Forest (map ID 51)** as the best reported story milestone; the dashboard itself did not advance the agent.

### v0.5.0 — Vermilion Gym progression and per-worker stagnation handling
Added directed map-transition tracking, forward-progression rewards, anti-oscillation signals, a Vermilion Gym arrival metric, and per-worker stagnation recovery. At approximately **58.49 million aggregate timesteps**, best story progress was still **rank 5 — Viridian Forest (map ID 51)** and **Vermilion Gym (map ID 92) had not been reached**. Reported means: **4.07 unique maps**, **4.27 unique directed transitions**, **1.07 forward transitions**, and **126 repeated transitions**. Mean steps since progress were 128, below the 12,288-step reset threshold; stagnation resets remained zero.

## Full map-region comparison

The complete table covers all **34 dated patch/source iterations** and **10 standalone snapshots/helpers** recovered from this workstream. It records the furthest region only when feedback or telemetry supports it; otherwise it says "Not reported".

[Open history/MAP_PROGRESS.md](history/MAP_PROGRESS.md).

Evidence note: the earlier direct location feedback said **Mt. Moon before the fossils**. The later November 15 rank-based training telemetry reported **rank 5 / map ID 51 — Viridian Forest**. The archive keeps these as separate evidence contexts rather than pretending they are one continuous run. The repository's `v2/map_data.json` identifies map ID 51 as Viridian Forest and IDs 59–61 as Mt. Moon areas.

## Privacy and branch safety

Installation examples use generic project-relative paths. The patch archives were scanned for the local username, media-volume label, known mount path, and previously observed personal email pattern; no matches were found. This archive is on a separate branch created from the existing master head. The master branch was not moved or edited.
