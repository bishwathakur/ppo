# SDD ledger — plan: H:\ml\kaggriculture_ppo\docs\superpowers\specs\2026-11-04-kaggriculture-ppo-plan.md

Spec: docs/superpowers/specs/2026-11-04-kaggriculture-day-level-ppo-design.md

## Setup

- Branch: master (no remote, local-only repo, single author).
  Ruling: no worktree — repo has no remote and no other branches; isolation
  buys nothing. Cost if wrong: none, history is linear local commits.

## Pre-flight scan

- Task 3 features.py FEAT_DIM consumed by Task 7 day_env, Task 9 model,
  Task 4 test: spec says "~70 dims" but plan fixes 64. Plan wins (explicit);
  model takes feat_dim from FEAT_DIM constant, so no hardcoded 70 anywhere.
  Ruling: FEAT_DIM is the single source of truth. Cost if wrong: one constant.
- Task 5 controller emits market/action dicts consumed by Task 7 day_env:
  interface `list[dict]` of 24, keys farmer/hands/market. Consistent.
- Task 10 ppo consumes Task 9 model.make_dist -> .sample/.log_prob/.entropy.
  Consistent.
- Task 12 train consumes Task 7 day_env gym API. Consistent.

## Progress

- Task 1 requirements.txt — done. Ruling: CPU torch (`2.14.1+cpu`, 124 MB) not the
  cu121 wheel. The cu121 download died 3× at 1.4/2.4 GB, and a 64→128→128 MLP with
  batch 64 is faster on CPU. Spec §17 + plan Task 1 + README updated. Cost if wrong:
  add `--index-url .../cu121` back, one line.
- Task 2 scripts/bench.py — done. **17.8 day-steps/s** (426 engine steps/s) → gate
  passes, no multiprocessing. Finding: the engine accepts **719** `step()` calls, not
  720; `episodeSteps=720` counts the reset state, so the terminal action is never
  applied and the last day has 23 actionable turns and no end-of-day. Bench loops on
  `env.done`; plan Task 7 updated.
- Task 3/4 features.py + tests — done, 4 passed. Schema probed live: `private` is
  already per-seat (own only, no player index needed); `day` is 0-based; tiles are
  `None | "LOCKED" | dict(kind=PLANT|WEED|COOP|PASTURE)`; `hands` are `[x, y]`;
  hire cost is `fib(hires_today)`; `BASE_PRICE` is taken from the engine's
  `MARKET_PARAMS` so price ratios cannot drift.
- Task 5/6 day_controller.py + tests — done, 8 passed. Rulings:
  1. `per_turn_action(macro, obs)` is the single source of truth and `day_env` calls
     it with the live obs each turn. `controller_actions` returns the spec's 24-entry
     plan by replaying it against a local state copy. A precomputed 24-turn plan is
     wrong: tiles and unit positions change every turn.
  2. `BUY_SEED` added to the turn-0 market list. The spec names SELL/BUY_LAND/HIRE
     only; without seeds PLANT is a silent no-op (engine line 423).
  3. Engine caps market orders at 10/turn (`maxMarketOrdersPerTurn`), silently
     dropping extras → order list clamped, hires get the remaining slots.
  4. **Engine bug found and guarded:** if the turn's total PLANT requests for a crop
     exceed seeds held, `interpreter` drops *all* of them (line 926). With 5 units
     and 3 seeds every unit re-asks forever — a livelock that produced 1489 no-op
     PLANTs and killed the carrot run ($2,738). Fixed with a shared per-turn seed
     budget; regression test `test_plant_requests_never_exceed_seeds`.
  5. Executor benchmark (fixed macro, vs starter): MELON/4 hires/dump **$28,957**,
     WHEAT/2 hires $9,525, TOMATO/4 $6,763, CARROT/4 $6,145, MELON+buy_land $17,502.
     Land purchase costs more than it returns at 4 hands — good, PPO has something
     to learn (buy_land=0).
- Task 7/8 day_env.py + tests — done, 5 passed. Bug found during the test: the day
  loop `while hour != 0` never ran a single step because `reset()` starts at hour 0
  → infinite test hang. Fixed to "step at least once, break when hour wraps".
  Ruling: callables are invoked every turn; a frozen self-play snapshot must cache
  its own day decision (phase 2). Ledger cost: 6 lines when self-play lands.
- Task 9 model.py + tests — done, 3 passed. `torch.distributions.Independent` does
  **not** accept a list of Categoricals (torch 2.14: `'list' object has no
  attribute 'batch_shape'`), so `MultiCategorical` is a 12-line wrapper. Test
  verifies log_prob/entropy equal the sum of the four factor terms, which is the
  real risk (a wrong slice silently misroutes actions).
- Task 10/11 ppo.py + tests — done, 3 passed incl. a hand-computed 2-step GAE and
  an episode-boundary case. **Reward scale fix:** raw dollar rewards squared gave
  `value_loss ≈ 2.7e6` and swamped the shared trunk's policy gradient. Rewards are
  now multiplied by `REWARD_SCALE = 1e-4` inside `collect_rollout` (the env still
  returns dollars; the reward-sum test is unchanged) and value_loss is O(0.01).
  Also: `torch.load` on 2.14 needs `weights_only=False` for our checkpoints (they
  carry numpy RNG state).
- Task 12 train.py — done. Smoke: 2→3 iterations, metrics.json + latest.pth +
  best.pth written, eval at iter 2 vs starter $7,783 / 100%.
- Task 13 play.py — done, prints the day table, schedule buckets and `--ascii` grid.
- Task 14 analyze.py — done on runs/smoke: learning_curve.png, winrate.png,
  summary.md. Deviation: the plan says the summary includes the learned schedule,
  but metrics.json has no per-day macros; the summary points at
  `play.py --checkpoint ... --ascii` instead. Cost if wrong: one plot panel.
- Task 15 .gitignore — done (runs/, analysis/, *.pth).

