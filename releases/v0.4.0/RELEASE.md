# v0.4.0 — Live party dashboard and navigation telemetry

## Changes
- Refreshes the live active-worker party view every **8,192 aggregate timesteps** rather than every 40,960.
- Keeps the current active-party image separate from the historical most-advanced-party image.
- Writes party PNGs and metadata atomically while training.
- Adds/retains TensorBoard lifetime map/tile and map-novelty telemetry.
- Does not alter PPO, observation keys, action space, ROM, or checkpoints.

## Map progress from feedback
This was an observability update, not a new gameplay policy. It improved visibility into current and best historical parties, but no separate map-progression milestone was attributed to the dashboard. Best reported story progress remained **rank 5 — Viridian Forest**.
