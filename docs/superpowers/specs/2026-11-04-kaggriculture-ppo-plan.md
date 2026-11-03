# Kaggriculture Day-Level PPO — Implementation Plan

**Spec:** `2026-11-04-kaggriculture-day-level-ppo-design.md`
**Order:** strict dependency chain — each task's done-when is a gate for the next.

---

### Task 1: `requirements.txt`

**What:** dependency file. CPU torch — see design spec §17 for the measurement.

**Key details:**
- No `--extra-index-url`: the default PyPI wheel is CPU-only and 124 MB
- Dependencies: `torch`, `numpy`, `gymnasium`, `kaggle-environments`, `pytest`, `matplotlib`, `tqdm`
- No version pins at first (let pip resolve); pin after smoke run passes

**Done when:** `pip install -r requirements.txt` completes and `python -c "import torch; print(torch.__version__)"` prints a version.

---

### Task 2: `scripts/bench.py`

**What:** one-shot throughput measurement — decides whether multiprocessing is needed before writing anything else.

**Key details:**
- Use `kaggle_environments.make("kaggriculture")` with direct `env.reset()` / `env.step([a0, a1])` (both seats, no `env.run()`)
- `a0` and `a1` are both `{"farmer": ["PASS"], "hands": [], "market": []}` for speed
- Measure 3 full episodes (720 steps each), compute day-steps/s (= engine steps / 24 / elapsed)
- Print: `Throughput: X.X day-steps/s — OK` if ≥ 3, else `SLOW — add SubprocVecEnv`
- Also confirm `env.step()` returns per-seat state objects (schema check)

**Done when:** prints throughput without error; number makes sense (expect 3–15 day-steps/s).

---

### Task 3: `features.py`

**What:** `encode(obs, player) -> np.float32[FEAT_DIM]` — converts the raw obs dict to a fixed-length float vector.

**Key details:**
- `CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]` (5)
- `PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]` (9)
- Sections (in order, all clipped/normalised to ~[-3, 3]):

| Section | Features | Count |
|---|---|---|
| Own farm | `money/10000`, `len(unlocked_quadrants)/4`, `hires_today/10`, `farmer_x/10`, `farmer_y/10`, `len(hands)/10` | 6 |
| Tile census (own 10×10) | count None, count weed, count structure, count-per-crop × 5 = 8 counts / 100 | 8 |
| Current farmer tile | one-hot kind (None/plant/weed/structure = 4), crop one-hot (5), `age/30`, `watered_today` | 11 |
| Private seeds | `seeds[crop]/10` × 5 | 5 |
| Private shed | `shed[product]/50` × 9 | 9 |
| Opponent census | `opp_money/10000`, `len(opp_quadrants)/4`, `len(opp_hands)/10` | 3 |
| Market prices | `price[p]/base_price[p]` × 9 | 9 |
| Market inventory | `inventory[p]/100` × 9 | 9 |
| Town | `len(unlocked_shops)/8` | 1 |
| Economy | `day/30`, `next_hire_cost/1000`, `(own-opp)/(abs(own)+abs(opp)+1)` | 3 |
| **Total** | | **64** |

- `FEAT_DIM = 64` (module constant)
- Raises `KeyError` with descriptive message on missing obs key — no silent zero-filling
- No `hour/24` (always 23 at day boundary — zero information)

**Done when:** Task 4 tests pass.

---

### Task 4: `tests/test_features.py`

**What:** pytest suite for `features.py`.

**Key details:**
- `test_features_shape`: live `env.reset()`, encode both seats, assert `len == FEAT_DIM`
- `test_features_finite`: assert `np.all(np.isfinite(vec))` for both seats
- `test_features_no_nan`: assert `not np.any(np.isnan(vec))`
- `test_features_schema_error`: mock obs with a missing key, assert `KeyError` raised (not silently zero-filled)

**Done when:** `python -m pytest tests/test_features.py -v` — all 4 pass.

---

### Task 5: `day_controller.py`

**What:** `controller_actions(macro, obs, cfg=None) -> list[dict]` — deterministic expansion of one macro into 24 per-turn action dicts.

**Key details:**

`macro` is a 4-element array: `[crop_focus, buy_land, n_hires, sell_mode]`

```
crop_focus: 0=WHEAT 1=CARROT 2=TOMATO 3=STRAWBERRY 4=MELON
buy_land:   0=no    1=yes (if money > land_price + 500 reserve)
n_hires:    0..4   hands to have hired by end of day
sell_mode:  0=hold  1=normal (price>=base)  2=dump (all)
```

Rules (in priority order per turn):
1. **Market list** (build once, emit on turn 0 only; engine cap is 10 orders/turn):
   - SELL items first, BUY items after (engine processes list in order; BUY before SELL can fail if cash is short)
   - `sell_mode=dump`: SELL all non-zero shed items regardless of price
   - `sell_mode=normal`: SELL items where `price >= BASE_PRICE[item]`
   - `sell_mode=hold`: no SELL orders
   - `BUY_SEED crop_focus` for the number of empty tiles not already covered by held seeds, capped at cash minus a 500 reserve. **The design spec omits this order; without it PLANT is a silent no-op and the farm never starts.**
   - Add `BUY_LAND` if `buy_land=1` and `money > next_land_price + 500`
   - Add `HIRE` orders for shortfall: `max(0, n_hires - hires_today)` HIRE entries, but only on turn 0
   - Turns 1–23: empty market list `[]`

2. **Farmer action** (pick the most useful one):
   - Any existing plant of *any* crop that needs water → `WATER` (maintains all crops, not just crop_focus)
   - Any existing plant of *any* crop that is ripe → `HARVEST`
   - Current tile is empty and a seed of `crop_focus` is available → `PLANT crop_focus`
   - Otherwise: `PASS` or move 1 step toward nearest actionable tile

3. **Hand actions**: list of one action per hand, same logic applied to each hand's position

4. Result: list of 24 dicts `{"farmer": [op, ...], "hands": [[op], ...], "market": [...]}`

Upstream bugs explicitly avoided:
- Queue HIRE shortfall once (turn 0), not on all 24 turns
- SELL before BUY in market list
- Maintain *all* crops' water/harvest needs regardless of `crop_focus`

**Done when:** Task 6 tests pass.

---

### Task 6: `tests/test_controller.py`

**What:** pytest for `day_controller.py`.

**Key details:**
- `test_hire_shortfall`: `n_hires=3, hires_today=1` → exactly 2 HIRE entries in turn 0's market list, 0 HIRE in turns 1–23
- `test_sell_before_buy`: in the turn 0 market list, every SELL entry appears before every BUY entry
- `test_controller_legal`: for 6 random macros × 3 seeds (18 configs) — step 24 engine turns, assert `env.steps` count advanced by 24, episode not terminated early (`terminated=False` after 24 steps in a fresh env)
- `test_all_crops_watered`: plant a CARROT manually on a tile, set crop_focus=MELON, call controller — assert WATER action appears for the CARROT tile

**Done when:** `python -m pytest tests/test_controller.py -v` — all 4 pass.

---

### Task 7: `day_env.py`

**What:** `gymnasium.Env` wrapping the Kaggriculture game at day resolution.

**Key details:**

```python
class DayEnv(gymnasium.Env):
    observation_space = Box(-np.inf, np.inf, (FEAT_DIM,), dtype=np.float32)
    action_space      = MultiDiscrete([5, 2, 5, 3])
```

- `__init__(opponent="starter", seed=0)`: opponent is `"starter"`, `"random"`, or a callable `obs_dict -> action_dict`
- `reset(seed=None) -> (obs, info)`: resets the engine env, returns encoded obs for seat 0
- `step(macro) -> (obs, reward, terminated, truncated, info)`:
  - Run engine turns until the day ends: the learner calls `controller.per_turn_action(macro, obs)` with the **live obs each turn** (not a precomputed 24-turn plan — tile and unit state change every turn)
  - The engine accepts 719 `step()` calls, not 720: `episodeSteps=720` counts the reset state, so the last day has 23 actionable turns and never reaches end-of-day
  - `reward = money_after_day - money_before_day`
  - `terminated = (engine_step == 720)`
  - `truncated = False`
  - `info = {"money": float, "opponent_money": float, "day": int}`
- Engine exception during step: mark terminated, log traceback once, return last known money
- Frozen opponent caching: callable is invoked once per day (turn 0); result cached and re-emitted for turns 1–23

**Done when:** Task 8 tests pass.

---

### Task 8: `tests/test_day_env.py`

**What:** pytest for `day_env.py`.

**Key details:**
- `test_episode_length`: run full episode vs `"starter"`, assert exactly 30 `step()` calls before `terminated=True`
- `test_reward_sum`: sum of 30 daily rewards ≈ `final_money - 3000` within $1 tolerance
- `test_obs_shape`: each obs returned by `reset()` and `step()` has shape `(FEAT_DIM,)` and is finite
- `test_reset_twice`: two consecutive `reset()` calls produce valid obs without engine error

**Done when:** `python -m pytest tests/test_day_env.py -v` — all 4 pass.

---

### Task 9: `model.py`

**What:** shared-trunk actor-critic network.

**Key details:**

```python
class ActorCritic(nn.Module):
    def __init__(self, feat_dim=FEAT_DIM, hidden=128):
        self.trunk  = nn.Sequential(Linear(feat_dim,hidden), Tanh, Linear(hidden,hidden), Tanh)
        self.actor  = nn.Linear(hidden, 15)   # 5+2+5+3, NOT 150
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.actor(h), self.critic(h).squeeze(-1)

    def make_dist(self, logits):
        # slice into 4 independent categoricals — NOT a joint 150-way softmax
        return Independent([
            Categorical(logits=logits[..., :5]),    # actor_crop_focus
            Categorical(logits=logits[..., 5:7]),   # actor_buy_land
            Categorical(logits=logits[..., 7:12]),  # actor_n_hires
            Categorical(logits=logits[..., 12:]),   # actor_sell_mode
        ])
```

- `make_dist(logits)` returns an object with `.sample()`, `.log_prob(action)`, `.entropy()`
- `log_prob` = sum of four factor log-probs (independent factorisation)
- Weight init: default PyTorch (Kaiming uniform); no custom init needed at this scale

**Done when:** `model.py` imports cleanly; forward pass on random input produces `(logits, value)` with shapes `(..., 15)` and `(...)` respectively; `make_dist(logits).sample()` returns shape `(..., 4)`.

---

### Task 10: `ppo.py`

**What:** hand-rolled PPO — the learning centrepiece. Mirror structure of `H:\ml\ppo\Proximal_Policy_Optimization_Algorithms\ppo.py` so the user can diff them.

**Key details — four functions:**

**`collect_rollout(envs, model, device, gamma, lam) -> buffers`**
- `envs`: list of `DayEnv` instances (already reset)
- For each env, run 30 day-steps, store: `states, actions, log_probs, rewards, values, dones`
- `torch.no_grad()` throughout — no gradient tracking during collection
- Bootstrap final value: `V(s_T)` from model if not done, else 0
- Returns dict of tensors, shape `(n_envs, 30, ...)`

**`compute_gae(rewards, values, dones, gamma=0.99, lam=0.95) -> (advantages, returns)`**
- GAE backwards loop with `(1 - dones[t])` mask at each step
- `returns = advantages + values[:, :T]`
- Comment above the loop: the formula `δ_t = r_t + γV(s_{t+1}) - V(s_t)`

**`ppo_update(model, optimizer, buffers, K, batch_size, clip_eps, vf_coef, ent_coef, clip_range_decay) -> loss_dict`**
- Flatten buffers to `(n_envs * 30, ...)`
- Normalise advantages (mean 0, std 1)
- K epochs of random minibatches:
  - Recompute `new_log_prob` and `value` for sampled states
  - `ratio = exp(new_log_prob - old_log_prob)`
  - Clipped policy loss (eq. 7 of PPO paper)
  - Clipped value loss
  - Entropy bonus
  - Total loss = `policy_loss + vf_coef * value_loss + ent_coef * entropy_loss`
  - `loss.backward(); clip_grad_norm_(0.5); optimizer.step()`
- Returns `{"policy_loss", "value_loss", "entropy", "clip_frac"}`

**Key differences from reference `ppo.py` (each gets a one-line comment):**
- `gamma = 0.99` over 30 steps (not 720; `0.99^30 ≈ 0.74`, healthy signal)
- Reward = money Δ, not `sign(reward)` — magnitude is the competition metric
- 4 factor heads — `log_prob = sum of factor log_probs` (factorised, not joint 150-way)
- No frame stack, no `MaxAndSkipEnv` — vector input, not pixels

**Done when:** Task 11 test passes.

---

### Task 11: `tests/test_ppo.py`

**What:** smoke test for one PPO iteration.

**Key details:**
- Use 2 envs, 1 PPO iteration, tiny model (hidden=32)
- Assert: all returned losses are finite (`not nan, not inf`)
- Assert: model parameters changed after the update
- Assert: entropy > 0 (policy is not deterministic)
- Assert: `clip_frac` is in [0, 1]

**Done when:** `python -m pytest tests/test_ppo.py -v` — passes.

---

### Task 12: `train.py`

**What:** CLI training loop — the entry point for all runs.

**Key details:**

```
python train.py [--iterations 200] [--envs 8] [--opponent starter]
                [--out runs/default] [--resume PATH] [--eval-every 10]
                [--lr 3e-4] [--ent-coef 0.01] [--gamma 0.99]
```

Loop structure:
```
seed everything (random, numpy, torch)
create N DayEnv instances
create ActorCritic, Adam optimizer
if --resume: load model + optimizer + RNG state from checkpoint

for iteration in range(iterations):
    buffers  = collect_rollout(envs, model, device)
    adv, ret = compute_gae(buffers)
    losses   = ppo_update(model, optimizer, buffers, adv, ret)
    
    anneal ent_coef linearly to 0 over all iterations
    
    append to metrics.json: {iter, losses..., timestamp}
    save latest.pth (model + optimizer + rng + iter)
    
    if iteration % eval_every == 0:
        eval_vs_starter = eval_episodes(model, opponent="starter", n=10, seeds=range(100,110))
        eval_vs_random  = eval_episodes(model, opponent="random",  n=10, seeds=range(200,210))
        append eval results to metrics.json
        if eval_vs_starter.mean_money > best_money:
            best_money = eval_vs_starter.mean_money
            save best.pth
        print progress line
```

- Non-finite loss: abort, save `latest.pth`, print offending tensor, exit 1
- `eval_episodes`: runs N full episodes with `model.eval()` + `torch.no_grad()`, returns `{mean_money, win_rate}`
- Progress print: `iter N | policy_loss X | value_loss X | entropy X | vs_starter $X (WR X%)`

**Done when:** `python train.py --iterations 2 --envs 2 --out runs/smoke` completes without error and writes `runs/smoke/metrics.json` and `runs/smoke/latest.pth`.

---

### Task 13: `play.py`

**What:** single-episode visualiser — the primary learning and debugging tool.

**Key details:**

```
python play.py [--checkpoint PATH] [--opponent starter|random] [--seed 0] [--ascii]
```

- Loads checkpoint if given, else uses random policy
- Runs one episode, printing after each `day_env.step()`:

```
Day  1 | crop=MELON  land=no  hires=2 sell=dump | money=$3,840 | opp=$4,100
Day  2 | crop=MELON  land=no  hires=3 sell=dump | money=$5,210 | opp=$4,890
...
Day 30 | crop=WHEAT  land=yes hires=4 sell=dump | money=$22,400 | opp=$12,300

Learned schedule:
  Days 1-12:  MELON, hires=3, dump (capital build)
  Days 13-30: WHEAT, hires=4, dump (volume)
  Land: bought day 9
  Final money: $22,400 | Opponent: $12,300 | WIN
```

- `--ascii`: after episode, print 10×10 grid where each cell is first char of tile type
  (`_`=empty, `W`=weed, `S`=structure, crop initial)

**Done when:** `python play.py --seed 0` prints a full day table without error.

---

### Task 14: `analyze.py`

**What:** post-run analysis — reads `metrics.json`, writes plots and summary.

**Key details:**

```
python analyze.py --run runs/overnight [--out analysis]
```

Outputs:
1. `analysis/learning_curve.png` — 4-panel: episode return, policy loss, value loss, entropy vs iteration
2. `analysis/winrate.png` — win rate and mean money vs `starter` over time
3. `analysis/summary.md` — text:
   - best `vs_starter` money and iteration achieved
   - final win rate
   - learned macro schedule (most common crop and sell_mode per day bucket: days 1–10, 11–20, 21–30)

**Done when:** `python analyze.py --run runs/smoke` completes (even with only 2 iterations of data) and writes the three files.

---

### Task 15: `.gitignore`

**What:** keep runs and checkpoints out of git.

**Contents:**
```
runs/
analysis/
__pycache__/
*.pth
*.pyc
*.egg-info/
.venv/
```

**Done when:** `git status` after a training run shows no `runs/` or `*.pth` entries.

---

## Execution order and dependencies

```
Task 1 (requirements)
    └─ Task 2 (bench)            ← gate: confirms env.step() works directly
        └─ Task 3 (features)
            └─ Task 4 (test_features)    ← gate: schema confirmed against 1.25.9
                └─ Task 5 (controller)
                    └─ Task 6 (test_controller)   ← gate: bugs verified absent
                        └─ Task 7 (day_env)
                            └─ Task 8 (test_day_env)  ← gate: 30-step episode works
                                └─ Task 9 (model)
                                    └─ Task 10 (ppo)
                                        └─ Task 11 (test_ppo)  ← gate: losses finite
                                            └─ Task 12 (train)
                                                ├─ Task 13 (play)
                                                ├─ Task 14 (analyze)
                                                └─ Task 15 (.gitignore)
```

Each gate test must pass before proceeding. Do not skip tasks.

---

## Running the full test suite

```powershell
python -m pytest tests/ -v
```

Expected output after all tasks complete:

```
tests/test_features.py::test_features_shape     PASSED
tests/test_features.py::test_features_finite    PASSED
tests/test_features.py::test_features_no_nan    PASSED
tests/test_features.py::test_features_schema_error PASSED
tests/test_controller.py::test_hire_shortfall   PASSED
tests/test_controller.py::test_sell_before_buy  PASSED
tests/test_controller.py::test_controller_legal PASSED
tests/test_controller.py::test_all_crops_watered PASSED
tests/test_day_env.py::test_episode_length      PASSED
tests/test_day_env.py::test_reward_sum          PASSED
tests/test_day_env.py::test_obs_shape           PASSED
tests/test_day_env.py::test_reset_twice         PASSED
tests/test_ppo.py::test_ppo_one_iteration       PASSED
13 passed in Xs
```
