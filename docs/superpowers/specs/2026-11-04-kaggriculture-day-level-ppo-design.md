# Kaggriculture Day-Level PPO — Design Spec

**Date:** 2026-11-04
**Status:** approved, implementation-ready
**Location:** `H:\ml\kaggriculture_ppo`

---

## 1. Purpose

Two goals, in order:

1. **Learn PPO.** Produce a small, readable, hand-rolled PPO implementation whose
   every line maps to the algorithm, so the user can study the math in working code.
2. **Train a competent Kaggriculture agent.** Reach a policy that beats the
   built-in `starter` agent consistently, as a foundation for a later, stronger
   system (behavior cloning, self-play league, animals).

Success for this phase:

- Short instrumented runs produce a rising reward curve and interpretable
  day-by-day plans within ~15 minutes.
- An overnight run produces analysis outputs (learning curves, win rates,
  money trajectories, macro-action distributions) showing sustained improvement
  against `starter`.
- The user can read `ppo.py` and explain the clipped surrogate, GAE, and the
  value-loss clip back to us.

## 2. Constraints

| Constraint | Value |
|---|---|
| Machine | Windows, Python 3.10, RTX 3050 Laptop 4 GB |
| Installed | `kaggle-environments 1.25.9`, `gymnasium 1.3.0`, `numpy 1.26.4` |
| Missing | `torch` (CPU wheel ≈ 124 MB — installed, `2.14.1+cpu`) |
| No Kaggle auth | `~/.kaggle/kaggle.json` absent → no replay download, no leaderboard |
| Competition state | closed 2026-10-14 → no submissions possible |
| Episode | 720 engine steps (30 days × 24 turns), ≈10 s wall-clock |
| GPU budget | small MLP only; no Transformer |

**Version risk:** installed `kaggle-environments` is `1.25.9`; the competition
pinned `1.32.7`. Obs schema must be asserted by test (§13), not assumed.

## 3. Research basis — why day-level hierarchical PPO

Sources: competition discussion threads
[#741258](https://www.kaggle.com/competitions/kaggriculture/discussion/741258),
[#741320](https://www.kaggle.com/competitions/kaggriculture/discussion/741320),
[#740969](https://www.kaggle.com/competitions/kaggriculture/discussion/740969);
public solution writeups `msdsm/kaggriculture-1st-solution` and
`LucachuTW/kaggriculture` (`rl/README.md`, `rl/RESEARCH.md`).

1. **Per-turn PPO from scratch does not learn.** Across 100k–800k timesteps it
   collapses to "do nothing" ($3,000 flat): a `random` opponent bankrupts itself,
   so standing still already wins, and 720-step credit assignment never surfaces
   the *buy→walk→plant→water→harvest→sell* chain.
2. **Shrinking the decision horizon fixes it.** Forum user BillDoser (#741258):
   the policy decides *the day's plan* and a deterministic process executes it —
   720 decisions → 30. Reported result from scratch: 100% win vs random at
   $20,448, 95% vs heuristic at $15,464. Even a uniformly random day-policy earns
   $4,600–$6,000, which per-turn random exploration never did.
3. **The ladder rewards macro-schedules.** Thread #741320 argues the competition
   is shared-market economics, not spatial play. The trained day-agent learned
   exactly a macro schedule (melon days 0–11 → wheat days 12–29, sell=dump,
   never buy land).
4. **1st place is not a learning vehicle.** `msdsm`: 12-block 10.23M Transformer,
   BC→PPO, conditional action masks, action-repair rules, C++ final-day search.
   Rejected for this phase as too heavy to learn from.
5. **Two documented engineering traps, avoided by construction:**
   - SB3's actor and critic are independent MLPs. After imitation the critic is
     pure noise and PPO pushes a good actor in random directions. → we use a
     **shared trunk with two heads**.
   - Checkpoint selection against `random` saturates at win-rate 1.0 immediately
     and freezes `best_model` forever. → we select **against `starter`**.

**Why not single-head flat menus (options A/C):** one op per turn cannot match
`starter`, which issues market *and* farmer ops in the same turn.
**Why not per-turn two-head (option B):** documented as unable to learn from
scratch; would need the full BC pipeline and Kaggle replays we cannot download.

## 4. Non-goals (this phase)

- Animals (Goose/Cow/Sheep) — largest remaining lever, roughly doubles the
  controller; deferred to phase 2.
- `FERTILIZE` — measured −6% upstream and reverted; do not add.
- Behavior cloning from replays — requires Kaggle auth we do not have.
- Final-day search planner, Transformer, Net2Net growth.
- Kaggle submission (competition closed).
- Multiprocessing / `SubprocVecEnv` — only if measured collection throughput
  falls below 3 day-steps/s (§12).

## 5. Architecture

```
                 ┌──────────────────────────────────────────┐
   obs (dict) ──►│ features.py   obs dict → ~70 float32     │
                 └───────────────┬──────────────────────────┘
                                 ▼
                 ┌──────────────────────────────────────────┐
                 │ model.py      shared MLP trunk           │
                 │               ├── actor head → 15 logits (4 factors)│
                 │               └── critic head → 1 value  │
                 └───────────────┬──────────────────────────┘
                                 ▼
   ppo.py ──► day_env.step(macro)   # ppo.py samples MultiDiscrete([5,2,5,3]); day_env runs the controller over 24 turns
        │
        ├── day_controller.py  macro → 24 per-turn action dicts
        │        └── kaggle_environments Env.step(actions) × 24
        ▼
   reward = money Δ over the day
```

One `day_env.step()` = one in-game day = 24 engine steps = 30 steps per episode.

## 6. Components

### 6.1 `day_controller.py`
Deterministic executor. Given the macro action and current obs, emits the 24
per-turn action dicts `{"farmer": [...], "hands": [...], "market": [...]}`.

Responsibilities:
- Choose the target tile for `crop_focus`: plant empty tile, else water unwatered
  plant of that crop, else harvest ripe plant, else walk toward target.
- Maintain existing plants of *any* crop regardless of `crop_focus` (water +
  harvest always) — invariant the research found missing and bug-causing.
- Queue `HIRE` orders only for the daily shortfall (`n_hires − hires_today`),
  once per day — upstream hit a bug queuing them on all 24 turns.
- Queue `SELL` orders **before** purchases in the same turn's market list; the
  engine processes the list in order, so a purchase queued first can fail on
  insufficient funds.
- `sell_mode`: `hold` (no sales), `normal` (sell when price ≥ base),
  `dump` (sell all shed inventory regardless of price).
- `buy_land`: issue `BUY_LAND` when cash covers price + reserve.
- Illegal actions are silent no-ops in the engine (verified, `kaggriculture.py:311`),
  so the controller never needs to prove legality — but it must not emit
  malformed dicts.

Interface: `controller_actions(macro, obs, cfg) -> list[dict]` (24 entries).

### 6.2 `day_env.py`
`gymnasium.Env`. Uses `kaggle_environments.make("kaggriculture")` with
`env.reset()` / `env.step(actions)` for per-step control (both seats supplied;
seat 0 = learner, seat 1 = opponent).

- `observation_space`: `Box(-inf, inf, (FEAT_DIM,))`
- `action_space`: `MultiDiscrete([5, 2, 5, 3])`
- `step(macro)`: run 24 engine turns, accumulate reward, return
  `(features, reward, terminated, truncated, info)`
- `terminated` at engine step 720; `info` carries `money`, `opponent_money`
- Opponent is pluggable: `starter`, `random`, or a frozen policy snapshot
- Frozen snapshot interface: callable `obs_dict → action_dict` that caches its
  day decision once and re-emits it for turns 1–23 of the day (the learner
  decides once per day; a real opponent therefore effectively decides once per
  day too).
- Reward is computed from engine money only — no shaping bonuses (the research
  documents a `PLANT`/`DIG` shaping exploit: plant, let it die, dig, replant).

### 6.3 `features.py`
`encode(obs) -> np.float32[FEAT_DIM]`, ~70 dims, all bounded:

- own farm: money, unlocked quadrants, `hires_today`, farmer position, hands count
- tile census: counts of empty / plant-by-crop / weed / structure, plus the
  current farmer tile block (kind, crop, age, watered)
- private: seed counts (5 crops), shed counts (9 products)
- opponent census (public only): money, quadrants, hands, animal/tile counts
- market: 9 prices, 9 inventories, price relative to base
- town: `len(unlocked_shops)`
- economy: `day/30`, next-hire-cost, bounded money advantage
  `(mine − theirs) / (|mine| + |theirs| + 1)`

`FEAT_DIM` is a module constant; a test asserts the vector length and finiteness.

### 6.4 `model.py`
Shared-trunk actor-critic (deliberately not SB3):

```
trunk: Linear(FEAT, 128) → Tanh → Linear(128, 128) → Tanh
actor: Linear(128, 15)           # outputs 5+2+5+3 = 15 logits; sliced into four categoricals
critic: Linear(128, 1)
```

Returns `(logits, value)`. The 15 actor outputs are sliced into four heads:
actor_crop_focus = logits[:5], actor_buy_land = logits[5:7], actor_n_hires = logits[7:12],
actor_sell_mode = logits[12:15]. Distribution built as
`Independent([Categorical(5), Categorical(2), Categorical(5), Categorical(3)], 1)`
so `log_prob` is the **sum** of the four factor log-probs — PPO's math is
unchanged from the single-head reference. Four independent heads; no joint
150-way softmax.

### 6.5 `ppo.py`
Structurally mirrors `H:\ml\ppo\Proximal_Policy_Optimization_Algorithms\ppo.py`
so the user can diff the two: rollout buffer → GAE → normalize advantages →
`K` epochs of minibatch updates with clipped policy loss, clipped value loss,
entropy bonus, grad clipping.

Differences from the reference (each needs a one-line comment explaining why):
- `gamma = 0.99` over **30** steps, not 720 (reference's `0.99^720 ≈ 0.0007`
  would erase all signal here; at 30 steps `0.99^30 ≈ 0.74` is healthy)
- reward is money Δ, **not** clipped to `sign()` — money magnitude is the metric
- four factor heads instead of one
- no `MaxAndSkipEnv` / frame stack — tabular-ish vector input

### 6.6 `train.py`
- Seeds everything (`random`, `numpy`, `torch`).
- Periodic eval: **10 episodes vs `starter` and 10 vs `random`** every 10
  iterations, recording mean final money and win rate. Fixed seeds so eval
  episodes are comparable across runs.
- **Checkpoint selection by performance vs `starter`**, mean money as tiebreak.
- Writes `runs/<name>/metrics.json` (per-iteration losses, entropy, clip frac,
  eval money, win rates) and saves `best.pth` / `latest.pth`.
- CLI: `--iterations`, `--envs`, `--timesteps`, `--opponent`, `--out`, `--resume`.

### 6.7 `play.py`
Visualization/learning tool. Runs a trained (or random) agent for one episode
and prints a day-by-day table: `day | macro | crop | land | hires | sell | money |
opp money`, plus a summary of the learned schedule. Optional `--ascii` renders
the 10×10 farm at the end.

CLI: `--checkpoint PATH` (omit for random policy), `--opponent starter|random` (default `starter`), `--seed INT` (default 0), `--ascii`.

### 6.8 `analyze.py`
Consumes `metrics.json` (+ optional eval dumps) from an overnight run and emits:
- learning curve (episode return, value loss, policy loss, entropy)
- win rate and mean money vs `starter` over time
- money trajectory comparison (trained vs `starter` vs `random`)
- macro-action distribution heatmap (which days → which crop/sell mode)

Output: `analysis/` PNGs + a short `analysis/summary.md`.

## 7. Action space (final)

`MultiDiscrete([5, 2, 5, 3])` = **150 combinations**

| Dim | Values | Meaning |
|---|---|---|
| `crop_focus` | WHEAT, CARROT, TOMATO, STRAWBERRY, MELON | which crop to plant/water/harvest |
| `buy_land` | 0/1 | attempt a land purchase today |
| `n_hires` | 0,1,2,3,4 | hands to hire today (Fibonacci cost) |
| `sell_mode` | hold / normal / dump | market aggressiveness |

`n_hires` is 0–4 rather than a boolean on purpose: the research measured that
single change at **+43% money vs random**.

## 8. Reward

`reward = money_end_of_day − money_start_of_day` (learner seat only).

No win bonus, no shaping bonuses:
- a `WIN_BONUS` rewards "do nothing wins" identically to a skilled win;
- `PLANT`/`DIG` shaping was exploited (plant → die → dig → replant loop).

Because the opponent is `starter` rather than `random`, "do nothing" loses
every game — the degenerate optimum is removed at the opponent level instead
of by reward hacking.

Sum of 30 daily rewards = terminal money − $3,000, so the return *is* the
competition metric.

## 9. PPO hyperparameters (initial)

| Param | Value | Note |
|---|---|---|
| `gamma` | 0.99 | 30-step horizon |
| `gae_lambda` | 0.95 | |
| `clip_range` | 0.2 | |
| `entropy_coef` | 0.01 | decay to 0 over run |
| `value_coef` | 0.5 | |
| rollout | 8 envs × 30 days = 240 day-steps / iteration | one full episode per env; day_env resets after each episode before the next rollout begins |
| `K` epochs | 4 | |
| minibatch | 64 | |
| optimizer | Adam, lr `3e-4` | linear decay to 0 |
| grad clip | 0.5 | same as reference |
| net | 128×128 tanh | fits 4 GB GPU trivially |

Measured after implementation: **~12 day-steps/s** single-process (the bench gate
requires >= 3). 8 envs × 30 days = 240 day-steps ≈ 20 s per rollout, so 10
iterations ≈ 3.5 min and 200 iterations ≈ 70 min of collection plus ~20 min of
evaluation.

## 10. Training stages

1. **Smoke** — 2 iterations (≈1 min), verify losses finite, curve writes, `play.py` runs.
2. **Short instrumented** — ~15 min. Watch the reward curve and day plans with
   the user. Iterate on code here.
3. **Overnight** — long run vs fixed `starter` opponent (~200 iterations at
   8 envs × 30 days; see §9 throughput math). Self-play league deferred to phase 2 (§15).
4. **Analysis** — `analyze.py` produces the outputs.

## 11. Error handling

- **Schema drift (1.25.9 vs 1.32.7):** `features.py` raises `KeyError` with the
  missing path; test in §13 fails loudly rather than silently zero-filling.
- **Non-finite loss:** abort the run, save `latest.pth`, print the offending
  tensor — never continue on NaN.
- **Engine exceptions mid-episode:** `day_env.step` marks the episode terminated
  with the money seen so far, logs the traceback once, and resets. A single bad
  episode must not kill an overnight run.
- **Checkpoint resume:** `--resume latest.pth` restores model + optimizer + RNG.
- **Missing torch:** fail at import with the exact install command.

## 12. Performance

Measured: one 720-step episode ≈ 10 s (≈72 engine steps/s) through
`kaggle_environments`' agent machinery. Direct `env.step()` should be faster —
**measure it first**.

Gate: if collection throughput < 3 day-steps/s (120 s/iter; at that rate stage-2’s 10-iteration target takes 20 min, over the 15-min goal), add per-process env workers
(only then — it is the one piece of infrastructure explicitly deferred).
Deliverable: `scripts/bench.py` measures env throughput and prints the verdict.

## 13. Testing

Small assert-based checks, run via `python -m pytest`:

1. `test_features` — schema from a live reset; vector length == `FEAT_DIM`;
   all finite; works for both seats.
2. `test_controller_legal` — for 150 macros × 3 seeds: action dicts are well
   formed (correct keys, ops in the engine vocabulary); after stepping, the
   engine step counter advanced by the expected amount and the episode did not
   terminate early. Illegal actions are silent no-ops, so "no exception" is not
   a valid legality check.
3. `test_hire_shortfall` — controller queues exactly `n_hires − hires_today`
   HIRE orders, not on all 24 turns (regression for the upstream bug).
4. `test_sell_before_buy` — same-turn market list orders SELL before BUY.
5. `test_day_env_roundtrip` — one `step()` advances engine by exactly 24,
   episode ends at day 30, reward sums to terminal money − 3000.
6. `test_ppo_one_iteration` — one iteration on a tiny net: all losses finite,
   parameters change, entropy > 0.

## 14. Deliverables

```
H:\ml\kaggriculture_ppo\
  docs/superpowers/specs/2026-11-04-kaggriculture-day-level-ppo-design.md
  features.py  day_controller.py  day_env.py  model.py  ppo.py
  train.py  play.py  analyze.py  scripts/  tests/
  requirements.txt  README.md
  runs/            # metrics + checkpoints (gitignored)
  analysis/        # overnight outputs
```

## 15. Phase 2 roadmap (out of scope now, recorded so it isn't lost)

1. Animals — `animal_focus` dim; only uncapped compounding income source.
2. Self-play league — phase 2, once win rate vs `starter` exceeds 0.8 in phase 1.
3. Behavior cloning — only if Kaggle auth is added; needs `pretrain_critic()`
   first (documented failure otherwise).
4. Per-turn reactive control for `FERTILIZE` / opponent-aware play — only with
   a finer-than-day, coarser-than-turn granularity.
5. KL penalty vs reference policy during fine-tuning.

## 16. Open questions for implementation

- Exact `FEAT_DIM` (settled by building `features.py` + test 1).
- Whether direct `env.step()` beats `env.run()` enough to skip multiprocessing
  (settled by the §12 measurement).

## 17. Install commands

**`requirements.txt`** (place this file in the project root):

```
torch
numpy
gymnasium
kaggle-environments
pytest
matplotlib
tqdm
```

One-liner equivalent:
```
pip install -r requirements.txt
```

- **CPU torch, deliberately.** Implemented and measured: the CUDA cu121 wheel is a
  2.4 GB download that failed repeatedly on this connection, and the network is a
  64→128→128 MLP with batch 64 — CPU beats GPU at this size (no transfer overhead).
  Installed: `torch 2.14.1+cpu` (124 MB). `device` stays a CLI flag, so a GPU run
  is a one-line change if phase 2 grows the network.
