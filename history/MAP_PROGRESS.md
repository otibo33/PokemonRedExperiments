# Map progress by recovered version

This index adds a map-region result to every dated source iteration and standalone snapshot in the recovered archive. “Not reported” means the retained feedback does not establish a map reached by that specific code revision; it is not a claim that the agent failed to move.

## How to read the map results

- **Furthest direct location explicitly reported in the conversation:** **Mt. Moon, before the fossil encounter** (the early baseline report, carried into the Mt. Moon-focused patch discussion).
- **Later rank-based telemetry series:** best story milestone was **rank 5 / map ID 51 — Viridian Forest** at approximately 47.33M, 54.07M, and 58.49M aggregate timesteps for the exploration/progression versions. The repository map data identifies map ID 51 as Viridian Forest, while map IDs 59–61 are Mt. Moon areas.
- These are retained as separate evidence records. The older direct Mt. Moon observation and later rank-5 telemetry are not forced into a single continuous run timeline, because the saved feedback does not prove they belong to the same uninterrupted training run or identical progression configuration.

## Archived patch/source iterations

| Date | Version folder | Furthest map region / outcome | Evidence note |
|---|---|---|---|
| 2026-11-08 | `01_pokemon_red_training_patch` | **Mt. Moon — before the fossils** | Direct user feedback about the baseline run: the agent was stuck in Mt. Moon before reaching the fossils. No exact map ID or timestep was attached. |
| 2026-11-09 | `01_training_patch_moon` | **Mt. Moon — before the fossils (last directly reported location)** | This Mt. Moon-focused patch follows the direct baseline report. No separate post-patch report confirms that the fossil encounter was completed. |
| 2026-11-09 | `02_tensorboard_image_update` | **Not reported for this iteration** | Image/visualization update; no distinct gameplay run result or new map location was reported. |
| 2026-11-09 | `03_full_rewards_current_run` | **Not reported for this iteration** | Reward update; the retained artifact/feedback does not tie a new furthest-map result specifically to this build. |
| 2026-11-09 | `04_tensorboard_safe_update` | **Not reported for this iteration** | Safe callback update; no distinct gameplay location result was reported. |
| 2026-11-09 | `05_tensorboard_callback_safe_update` | **Not reported for this iteration** | Callback-only update; no gameplay movement result was reported. |
| 2026-11-09 | `06_map_resolution_update` | **Not reported for this iteration** | Map image-resolution update; it changes visualization rather than game navigation. |
| 2026-11-09 | `07_tensorboard_maps_only` | **Not reported for this iteration** | TensorBoard map-image logging update; no gameplay movement result was reported. |
| 2026-11-09 | `08_tensorboard_map_only` | **Not reported for this iteration** | TensorBoard map-image logging update; no gameplay movement result was reported. |
| 2026-11-10 | `01_progress_history_fix` | **Not reported for this iteration** | Progress-history instrumentation change; it improves tracking but no new in-game region was reported for this code version. |
| 2026-11-12 | `01_capture_aware` | **Not reported for this iteration** | Capture/Dex reward logic; no version-specific training map result was retained. |
| 2026-11-12 | `02_capture_hm_aware` | **Not reported for this iteration** | Capture and HM awareness reward logic; no version-specific training map result was retained. |
| 2026-11-12 | `03_capture_hm_party_tensorboard` | **Not reported for this iteration** | Capture/HM/party telemetry update; no version-specific training map result was retained. |
| 2026-11-12 | `04_party_visual` | **Not reported for this iteration** | Party visualization and related metrics; no distinct map-progression result was reported. |
| 2026-11-12 | `05_party_tensorboard_fix` | **Not reported for this iteration** | Party dashboard callback integration fix; no distinct map-progression result was reported. |
| 2026-11-12 | `06_tensorboard_callback_fix` | **Not reported for this iteration** | Moves party-image logging into the active training callback; no gameplay map result was reported. |
| 2026-11-12 | `07_most_advanced_party` | **Not reported for this iteration** | Tracks the party associated with the leading worker; no actual furthest-map location was reported for this snapshot. |
| 2026-11-12 | `08_balanced_diverse_party` | **Not reported for this iteration** | Party size/diversity and balance reward update; no version-specific map result was reported. |
| 2026-11-12 | `09_party_visual_env` | **Not reported for this iteration** | Party visualization environment update; no gameplay map result was reported. |
| 2026-11-12 | `10_starter_diversity` | **Not reported for this iteration** | Starter-selection diversity reward update; no version-specific map result was reported. |
| 2026-11-12 | `11_party_level_spread` | **Not reported for this iteration** | Party level-spread reward correction; no version-specific map result was reported. |
| 2026-11-13 | `01_collection_tuning` | **Not reported for this iteration** | Collection reward tuning; no version-specific furthest-map result was retained. |
| 2026-11-13 | `02_logger_key_fix` | **Not reported for this iteration** | TensorBoard logger-key collision fix; no gameplay movement change or map result was reported. |
| 2026-11-13 | `03_hm_awareness` | **Not reported for this iteration** | HM carrier/readiness awareness patch; no version-specific training map result was retained. |
| 2026-11-13 | `04_hm_field_awareness` | **Not reported for this iteration** | Adds HM field-use tracking and reward signals; no version-specific furthest-map result was retained. |
| 2026-11-13 | `05_capture_and_hm_awareness` | **Not reported for this iteration** | Combines collection and HM-awareness signals; no version-specific furthest-map result was retained. |
| 2026-11-14 | `01_party_sprite_type_fix` | **Not reported for this iteration** | Party sprite/type metadata correction; no gameplay map result was reported. |
| 2026-11-14 | `02_duplicate_species_fix` | **Not reported for this iteration** | Duplicate species/capture reward correction; no version-specific map result was reported. |
| 2026-11-14 | `03_duplicate_capture_healing` | **Not reported for this iteration** | Duplicate-capture and lead-survival/healing reward update; no version-specific furthest-map result was reported. |
| 2026-11-15 | `01_humanlike_progression` | **No completed map milestone recorded** | This alternative goal-conditioned experiment was not adopted as the continuing training line; development returned to the original PPO v2 run. |
| 2026-11-15 | `02_exploration_speed` | **Viridian Forest — rank 5, map ID 51** | At approximately 47.33M aggregate timesteps, the best reported story-map milestone was rank 5/map ID 51. Repository map data identifies map ID 51 as Viridian Forest. |
| 2026-11-15 | `03_directional_exploration` | **Viridian Forest — rank 5, map ID 51** | At approximately 54.07M aggregate timesteps, best story progress remained rank 5/map ID 51 (Viridian Forest). Mean lifetime coverage was about 175 unique tiles and 4.33 unique maps per worker; those coverage values do not mean the story milestone advanced. |
| 2026-11-15 | `04_live_party_visual_update` | **Viridian Forest — rank 5, map ID 51 (last surrounding telemetry)** | This was an observability update, not a new gameplay policy. The surrounding training feedback still reported rank 5/map ID 51; no new region was attributed to the dashboard. |
| 2026-11-15 | `05_vermilion_gym_progression` | **Viridian Forest — rank 5, map ID 51; Vermilion Gym not reached** | At approximately 58.49M aggregate timesteps, best story progress remained rank 5/map ID 51 (Viridian Forest). Vermilion Gym (map ID 92) had not been reached. |

## Standalone source snapshots and helpers

| Date | Snapshot folder | Furthest map region / outcome | Evidence note |
|---|---|---|---|
| 2026-11-08 | `01_initial_training` | **Mt. Moon — before the fossils** | The standalone initial environment/trainer snapshot belongs to the baseline run for which the user reported being stuck in Mt. Moon before the fossils. |
| 2026-11-09 | `01_tensorboard_image` | **Not reported for this snapshot** | Visualization helper; no new gameplay map result was reported. |
| 2026-11-09 | `02_tensorboard_minimal` | **Not reported for this snapshot** | Minimal callback snapshot; no gameplay location result was reported. |
| 2026-11-09 | `03_tensorboard_safe` | **Not reported for this snapshot** | Safe callback snapshot; no gameplay location result was reported. |
| 2026-11-10 | `01_progress_history` | **Not reported for this snapshot** | Standalone progress-history environment/callback snapshot; no new map region was reported with this file snapshot. |
| 2026-11-10 | `02_optimization_helpers` | **Not reported for this snapshot** | Patch helpers for emulator tick optimization; no new gameplay map result was reported. |
| 2026-11-12 | `01_party_visual_versions` | **Not reported for these snapshots** | Standalone callback versions for party visualization; no version-specific map result was reported. |
| 2026-11-13 | `01_level_spread_fix` | **Not reported for this snapshot** | Standalone environment/callback level-spread correction; no gameplay map result was reported. |
| 2026-11-13 | `02_logger_hm_fixes` | **Not reported for this snapshot** | Logger/HM test helper snapshots; no gameplay map result was reported. |
| 2026-11-13 | `03_tensorboard_trim` | **Not reported for this snapshot** | TensorBoard image trimming utility; it does not change game navigation. |

## Map-ID reference

- `v2/map_data.json`: map ID `51` is **Viridian Forest**; IDs `59`, `60`, and `61` are Mt. Moon areas.
- The progress logs supplied with the November 15 feedback recorded `best_map_id = 51` and `best_ever_map_progress = 5`.
- The earlier Mt. Moon-before-fossils observation is transcribed from the user’s feedback, rather than inferred from a map ID.