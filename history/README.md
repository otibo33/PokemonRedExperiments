# Complete PokémonRedExperiments source history

This is a chronological source archive for the PokémonRedExperiments/v2 workstream. It preserves the recovered code artifacts rather than only release notes.

## Contents

- `PokemonRedExperiments-full-python-history-2026-11-08-to-2026-11-15.zip`
  - **141 Python files**, including environment versions, callbacks, trainers, patch application scripts, utility scripts, and tests.
  - **35 patch/source archives** from November 8 through November 15, 2026.
  - **17 standalone Python snapshots/helpers** saved during the same workstream.
  - Patch-format diffs and the READMEs included with the source artifacts.
  - A detailed `MANIFEST.json` with archive/date/file paths and SHA-256 hashes.
  - `AUDIT.json` with recovery counts and privacy scan status.
  - `MAP_PROGRESS.md` and `MAP_PROGRESS.json`, indexing the furthest reported map region for every dated source iteration and standalone snapshot.

## Map progress at a glance

The furthest direct location explicitly reported in the earlier feedback was **Mt. Moon, before the fossil encounter**. Later rank-based telemetry for the November 15 exploration/progression versions reported **rank 5 / map ID 51 — Viridian Forest**. The map index preserves both observations and flags unreported iterations instead of guessing.

[Open the full per-version map progress table](MAP_PROGRESS.md).

## Chronology

- **2026-11-08:** initial trainer/environment patch.
- **2026-11-09:** emulator tick optimization, TensorBoard image and map rendering, reward and callback updates.
- **2026-11-10:** progress-history persistence and RL optimization helpers.
- **2026-11-12:** capture awareness, HM awareness, party display, starter diversity, party balance and level-spread iterations.
- **2026-11-13:** collection tuning, callback logger fixes, HM field-use tracking and related tests.
- **2026-11-14:** party sprite metadata, duplicate-capture fixes, lead survival/healing updates.
- **2026-11-15:** goal-conditioned humanlike experiment, exploration reward, directional exploration, live party visualization, and Vermilion Gym progression updates.

Each item is kept in its own dated folder inside the ZIP. Repeated source filenames are intentionally retained under their original iteration folders so that older code is not overwritten by a later revision.

## Scope

The independent from-scratch `Pokémon Red Lab` project is excluded. It was developed as a separate project and is not part of this `PokemonRedExperiments/v2` history.

This history branch is an archive only. It does not replace `master`, and the main branch and existing working branches were not moved by this source-history commit.
