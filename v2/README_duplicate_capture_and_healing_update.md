# Pokémon Red v2 — Duplicate Capture + Lead Survival Update

This package is for the existing `PokemonRedExperiments/v2` project only. It is
not for the separate Pokémon Red RL Lab project.

## What caused the repeated Geodudes

The saved environment counted every successful capture marker as a full capture
reward, whether or not the Pokédex already owned that species. It also rewarded
an additional active-party slot without penalising a duplicate species. As a
result, catching a second Geodude could still earn the full capture and party
size rewards even though it added no new species.

The update separates these cases:

- A first capture of a species not already owned earns the novelty capture reward.
- A repeat capture increments the total capture counter, but earns no novelty
  reward and receives a duplicate-capture penalty.
- Every redundant species slot in the active party receives a separate
  current-state penalty. For example, two Geodudes count as one redundant slot.
- Duplicate species in the PC do not affect active-party diversity. Capture
  duplicates are still tracked separately, including when the new Pokémon is
  automatically sent to the PC because the party is full.
- Geodude is classified using its Gen-I internal species ID and National Dex
  mapping; no species-name guesswork is used.

Default weights at `reward_scale=0.5`:

| Event | Score contribution |
|---|---:|
| First-time species capture | +16 capture score, plus the normal Dex progress score |
| Repeat capture of an already-owned species | -5 duplicate-capture score |
| One extra duplicate slot in the active party | -2 party-duplicate score |

These are state-score deltas in the environment's reward calculation; they are
not a direct instruction to catch or release any particular Pokémon.

## Lead Pokémon survival and Pokémon Centers

The old `health` observation was the combined party HP fraction. Healthy reserve
Pokémon could therefore hide a nearly fainted lead Pokémon. The update changes
the value carried by the existing `health` observation key to the first party
member's HP fraction, while keeping the key, shape `(1,)`, and `float32` dtype
unchanged. The existing aggregate-party HP reader remains in place for the
original healing metric.

Additional reward shaping:

- The lead receives a danger-state penalty below 30% HP.
- Remaining below that threshold accumulates a small penalty each environment
  step, so staying in danger becomes progressively costly.
- A lead faint receives an additional penalty.
- Entering a verified Pokémon Center while the lead is already at critical HP
  earns a one-time arrival reward for that Center per episode.
- A meaningful increase in lead HP while inside a Center is recorded as a
  successful Center heal and earns a recovery reward.

Center IDs are based on the English Red/Blue map constants in pret/pokered.
This includes Viridian, Pewter, Cerulean, Mt. Moon, Rock Tunnel, Vermilion,
Celadon, Lavender, Fuchsia, Cinnabar, Indigo Plateau lobby, and Saffron.

The Center rewards are navigation shaping, not a scripted route planner. The
policy still has to learn the route and the nurse interaction from the existing
screen/map observations and action set.

## TensorBoard metrics

The environment `info` payload and callbacks now expose:

- `novel_captures`, `duplicate_captures`, `unclassified_captures`
- `party_duplicate_count` and `party_duplicate_species`
- `lead_hp_fraction`, `lead_hp_danger`, `lead_hp_critical`,
  `critical_lead_hp_steps`, `lead_faints`
- `center_visits`, `center_low_hp_visits`, `center_heals`,
  `center_hp_restored`
- separate current reward-score components for capture novelty, duplicate
  capture penalty, duplicate party penalty, critical-HP exposure, fainting,
  Center return, and Center healing

`GameStatsCallback` logs the numeric values under `game/...`. The richer
`TensorboardCallback` adds `healing/...` and `rewards/...` scalar tags when that
callback is included in the trainer's callback list.

## Map-image tags

The map images have two different meanings and should not be confused:

- `trajectory/episode_explore_map` and `trajectory/episode_explore_sum` are
  intentionally per-episode images. They restart from the episode's starting
  state and therefore look like the map is being rediscovered in each episode.
- `trajectory/explore_map` and `trajectory/explore_sum` are the cumulative
  lifetime maps. The callback merges terminal episode snapshots and reads each
  worker's `lifetime_explore_map` after SB3 auto-reset, so these should not
  restart at every episode.

The update preserves that separation. If the non-episode `trajectory/explore_*`
tags themselves reset, that points to an old callback/environment being used
or a worker/session data issue rather than the intended episode-map behavior.

## Checkpoint compatibility and caveat

The action space and observation dictionary keys, shapes, and dtypes are
unchanged. Existing PPO checkpoints remain structurally loadable. However, the
meaning of the existing `health` input has changed from whole-party average HP
to lead-Pokémon HP. A resumed policy will initially interpret that scalar using
its old learned meaning, so continued training should be treated as fine-tuning
under a revised reward/health signal, not as behaviorally identical inference.

The training launcher does not need a new PPO architecture. No `runs/`,
checkpoint, TensorBoard event, `init.state`, or starter-choice database files
are included in this package or need to be deleted.

## Install

Stop the trainer normally with `Ctrl+C` first. From the v2 project directory,
make backups and then extract the ZIP so the three updated Python modules and
helper/test files land beside the originals:

```fish
set stamp (date +%Y%m%d_%H%M%S)
cp red_gym_env_v2.py "red_gym_env_v2.py.pre_duplicate_healing_$stamp"
cp tensorboard_callback.py "tensorboard_callback.py.pre_duplicate_healing_$stamp"
cp game_stats_callback.py "game_stats_callback.py.pre_duplicate_healing_$stamp"
unzip -o ~/Downloads/pokemon_red_duplicate_capture_healing_patch.zip -d .
```

Run validation in the project's existing virtual environment:

```fish
python -m py_compile red_gym_env_v2.py tensorboard_callback.py game_stats_callback.py gen1_party_metadata.py
python test_capture_rewards.py
python test_party_balance.py
python test_starter_diversity.py
python test_callback_party_metrics.py
python test_hm_awareness_patch.py
python test_party_metadata_visual_fix.py
```

Only resume training after all checks pass. Use the same existing checkpoint
and run directory you normally use. Keep the existing `init.state`, `runs/`,
TensorBoard logs, and `starter_choice_counts.sqlite3` intact.

## Validation performed while preparing this package

The source files passed Python bytecode compilation. All bundled unit tests
listed above passed in the build environment. The tests use isolated RAM/reward
fixtures and callback stubs; a live PyBoy/ROM training smoke test still needs to
be run in the user's local v2 environment.
