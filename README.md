# Kaggriculture PPO — Learning Walkthrough

A from-scratch PPO agent for the [Kaggriculture](https://www.kaggle.com/competitions/kaggriculture)
farming simulation. Two goals in order: **understand PPO**, then **train an agent that beats the
built-in `starter`**.

---

## What you'll understand by the end

```
PPO algorithm (clipped surrogate + GAE + value clip + entropy)
 └─ applied to a structured action space (MultiDiscrete, 4 independent heads)
     └─ on a day-level hierarchical environment
         └─ that wraps the Kaggriculture farming game
```

---

## Quick start

```powershell
# 1. Install (CPU torch: the network is a 64→128→128 MLP, CPU is faster than GPU
#    here and the CUDA wheel is a 2.4 GB download)
pip install -r requirements.txt

# 2. Measure environment speed (decides if you need multiprocessing)
python scripts/bench.py

# 3. Run all tests
python -m pytest tests/ -v

# 4. Smoke run (2 iterations, ~1 min) — verify everything works
python train.py --iterations 2 --envs 2 --out runs/smoke

# 5. Watch what the agent does (random policy)
python play.py --seed 0

# 6. Short learning run (~15 min, 10 iterations)
python train.py --iterations 10 --envs 8 --out runs/short

# 7. Watch the trained agent
python play.py --checkpoint runs/short/best.pth

# 8. Overnight run (~4.4 h, 200 iterations)
python train.py --iterations 200 --envs 8 --out runs/overnight

# 9. Analyze results
python analyze.py --run runs/overnight
```

---

## How to learn PPO — step by step

### Stage 1: Understand the game (30 minutes)

**Goal:** know what the agent must decide, before touching any ML code.

1. Read `docs/superpowers/specs/2026-11-04-kaggriculture-day-level-ppo-design.md` §1–4.
2. Run one episode in a Python shell and look at the raw observation:

```python
from kaggle_environments import make
env = make("kaggriculture", debug=True)
env.run(["starter", "random"])

# Look at what the agent sees each turn
obs = env.steps[0][0].observation
print(sorted(obs.keys()))          # top-level keys
print(sorted(obs["farms"][0].keys()))   # own farm
print(obs["market"])               # prices + inventory
print(obs["private"])              # seeds, shed, inventories
```

3. Read `day_controller.py`. This is the **deterministic part** — it translates a
   macro decision (which crop? how many hires? sell mode?) into 24 concrete per-turn
   actions. You don't need to understand PPO to read this. The only question it answers
   is: *given a plan for today, what exact button do I press each turn?*

4. Run `play.py` with no checkpoint (random policy) and read the day table:
```powershell
python play.py --seed 0
```
   An untrained policy wanders around the starting balance (~$3–4k). That's the
   floor. The ceiling with a *hand-written* day plan is much higher — measured
   with the deterministic executor against `starter`:

| Fixed macro | Final money |
|---|---|
| MELON, 4 hires, dump | **$28,957** |
| WHEAT, 2 hires, dump | $9,525 |
| TOMATO, 4 hires, dump | $6,763 |
| CARROT, 4 hires, dump | $6,145 |
| MELON + buy land, 4 hires, dump | $17,502 |

   So the job is not "learn to farm from scratch" — it is "discover which of the
   150 macro combinations is worth $29k". Buying land *loses* money at 4 hands,
   which is why `buy_land` is a separate head rather than a default.

---

### Stage 2: Understand the observation encoding (30 minutes)

**Goal:** know how game state becomes numbers a neural network can read.

Open `features.py` and read `encode(obs, player)`. Every group is labelled:

```python
# ── own farm ──────────────────────────────────────
vec.append(obs["farms"][player]["money"] / 10_000)    # scale to ~0-3
vec.append(len(obs["farms"][player]["unlocked_quadrants"]) / 4)
# ...

# ── market ────────────────────────────────────────
for product in PRODUCTS:
    price = obs["market"]["prices"][product]
    vec.append(price / BASE_PRICE[product])           # ratio to base
# ...
```

Questions to answer as you read:
- Why divide money by 10,000? (neural nets train badly with raw dollar values)
- Why `hour/24` was removed? (always 23 at day-level — zero information)
- What is the "bounded money advantage" feature? (`(mine−theirs)/(|mine|+|theirs|+1)`) Why bound it?

Run `test_features.py` to confirm the vector is correct length and fully finite:
```powershell
python -m pytest tests/test_features.py -v
```

---

### Stage 3: Understand the model (20 minutes)

**Goal:** know what goes in and what comes out of the neural network.

Open `model.py`. The full network is ~20 lines:

```
observation vector (FEAT_DIM floats)
        ↓
Linear(FEAT_DIM, 128) → Tanh → Linear(128, 128) → Tanh     ← shared trunk
        ↓                                         ↓
Linear(128, 15)    ← actor                  Linear(128, 1)  ← critic
        ↓                                         ↓
15 logits split                            one number:
into 4 groups:                             "how good is this state?"
  [:5]  → which crop (5 options)           → value V(s)
  [5:7] → buy land? (2 options)
  [7:12]→ how many hires (5 options)
  [12:] → sell mode (3 options)
```

**Two heads from one trunk** — this is the key architectural fix vs SB3's default.
SB3 builds actor and critic as completely separate MLPs. That means after behavior
cloning only the actor has learned anything; the critic is random noise, and PPO's
advantage estimates are garbage. One shared trunk means both heads see the same
learned features.

Run a forward pass in a shell:

```python
import torch, numpy as np
from features import encode, FEAT_DIM
from model import ActorCritic
from kaggle_environments import make

env = make("kaggriculture")
obs = env.reset()[0].observation   # first agent's obs

model = ActorCritic()
x = torch.as_tensor(encode(obs, player=0)).unsqueeze(0)  # (1, FEAT_DIM)
logits, value = model(x)

print("logits shape:", logits.shape)   # (1, 15)
print("value:", value.item())          # a single number
dist = model.make_dist(logits)
action = dist.sample()
print("sampled action:", action)       # tensor of shape (4,): [crop, land, hires, sell]
print("log_prob:", dist.log_prob(action).item())
```

---

### Stage 4: Understand PPO (the core — 1–2 hours)

**Goal:** read `ppo.py` and be able to explain every section.

Open `ppo.py` alongside the reference implementation at
`H:\ml\ppo\Proximal_Policy_Optimization_Algorithms\ppo.py`.
The structure is identical; only the environment interface differs.

#### Section 1: Rollout collection

```python
# Collect T=30 day-steps from each environment.
# For each step: observe → model forward → sample action → step env → store.
# Nothing is learned here; we're just gathering experience.
```

What to notice:
- `buffer_states`, `buffer_actions`, `buffer_logprobs`, `buffer_rewards`, `buffer_values`
- We store `old_log_probs` from the policy *at collection time*. PPO needs these later.

#### Section 2: Generalized Advantage Estimation (GAE)

GAE answers: "how much better was this action than average?"

```
δ_t  = r_t + γ · V(s_{t+1}) - V(s_t)         ← TD error
A_t  = δ_t + γλ · δ_{t+1} + (γλ)² · δ_{t+2} + ...  ← GAE
```

In code (read backwards through time):
```python
for t in reversed(range(T)):
    delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
    gae   = delta + gamma * lam * gae * (1 - dones[t])
    advantages[t] = gae
```

Why backwards? Because each advantage depends on the *next* one.
Why `(1 - dones[t])`? To zero out the bootstrap at episode boundaries.

γ=0.99 here over 30 steps. `0.99^30 ≈ 0.74` — day 1 rewards still count.
The reference `ppo.py` uses `0.99^720 ≈ 0.0007` — day 1 rewards are invisible.

#### Section 3: Advantage normalisation

```python
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

Pure numerical stability. Makes policy gradient updates scale-independent.

#### Section 4: The PPO loss (3 parts)

**Part A — Clipped policy loss** (the PPO paper's eq. 7):

```python
ratio       = exp(new_log_prob - old_log_prob)   # how much policy changed
surr1       = ratio * advantage
surr2       = clamp(ratio, 1-ε, 1+ε) * advantage
policy_loss = -mean(min(surr1, surr2))
```

`ratio` = how much more (or less) likely the action is under the *new* policy vs
the *old* one. If ratio drifts far from 1.0, `surr2` clips it. This prevents
a single large update from destroying a good policy.

**Part B — Clipped value loss:**

```python
v_clipped   = old_values + clamp(values - old_values, -ε, +ε)
value_loss  = mean(max(MSE(returns, values), MSE(returns, v_clipped)))
```

Same idea: don't let the value estimate jump too far per update.

**Part C — Entropy bonus:**

```python
entropy_loss = -mean(dist.entropy())   # negative because we add it to loss
total_loss   = policy_loss + vf_coef*value_loss + ent_coef*entropy_loss
```

Keeps the policy from going deterministic too early. As training progresses,
`ent_coef` decays toward 0 — exploration naturally reduces.

#### Self-check

After reading, close the file and answer these:
1. What is GAE and why does it use `λ`?
2. Why does PPO clip the ratio instead of just computing policy gradient directly?
3. What happens if `entropy_coef = 0` too early?
4. Why does the farmer head use `Categorical(5)` instead of `Categorical(150)`?

---

### Stage 5: Run the short instrumented run (15 minutes)

```powershell
python train.py --iterations 10 --envs 8 --out runs/short
```

While it runs, watch the output. You should see:
- `policy_loss` start high and fall
- `entropy` start high (random) and gradually fall
- `value_loss` high at first (critic learning), then stabilise
- After iter 10: `eval_money_vs_starter` rising above 0 (from $0 to maybe $2k–$8k)

```powershell
# After it finishes, watch the learned agent play:
python play.py --checkpoint runs/short/best.pth
```

Read the day table. Did it discover anything sensible?
- Does it plant MELONs early? (highest base price)
- Does it use `sell_mode=dump`? (the research found that wins)

---

### Stage 6: Overnight run + analysis

```powershell
# Start before bed, ~4.4 h:
python train.py --iterations 200 --envs 8 --out runs/overnight

# Morning: generate analysis outputs
python analyze.py --run runs/overnight
```

Open `analysis/`:
- `learning_curve.png` — did reward rise, then plateau?
- `winrate.png` — does it beat `starter` consistently after ~50 iterations?
- `summary.md` — what macro schedule did it learn?

---

## How Kaggriculture PPO works — the full picture

```
┌──────────────────────────────────────────────────────────────┐
│                    TRAINING LOOP (ppo.py)                    │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              ROLLOUT PHASE (no gradient)             │    │
│  │                                                      │    │
│  │  for each of 8 envs, for 30 day-steps:               │    │
│  │    obs → features.encode() → model.forward()         │    │
│  │        → logits, value                               │    │
│  │        → sample 4-dim action [crop,land,hires,sell]  │    │
│  │        → day_controller.controller_actions()         │    │  
│  │            → 24 per-turn action dicts                │    │
│  │            → env.step() × 24                        │    │
│  │        → store (state, action, log_prob, reward,     │    │
│  │                 value, done)                         │    │
│  │                                                      │    │
│  │  reward = money Δ that day (no shaping; no clipping) │    │
│  └─────────────────────────────────────────────────────┘    │
│                           ↓                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              GAE PHASE (no gradient)                 │    │
│  │                                                      │    │
│  │  compute δ_t = r_t + γV(s_{t+1}) - V(s_t)           │    │
│  │  compute A_t backwards through time with λ=0.95      │    │
│  │  normalise advantages                                │    │
│  └─────────────────────────────────────────────────────┘    │
│                           ↓                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │             UPDATE PHASE (K=4 epochs)                │    │
│  │                                                      │    │
│  │  shuffle 240 day-steps into minibatches of 64        │    │
│  │  for each minibatch:                                 │    │
│  │    recompute log_prob and value under current policy │    │
│  │    ratio = exp(new_log_prob - old_log_prob)          │    │
│  │    policy_loss = -mean(min(ratio·A, clip(ratio)·A))  │    │
│  │    value_loss  = mean(max(MSE, MSE_clipped))         │    │
│  │    entropy_loss= -mean(dist.entropy())               │    │
│  │    loss = policy_loss + 0.5·value_loss + 0.01·ent    │    │
│  │    loss.backward(); clip grad 0.5; Adam.step()       │    │
│  └─────────────────────────────────────────────────────┘    │
│                           ↓                                  │
│         every 10 iters: eval 10 ep vs starter                │
│         save best.pth if mean money improved                 │
└──────────────────────────────────────────────────────────────┘
```

### Why day-level, not per-turn

Per-turn PPO from scratch collapses to "do nothing" on this game:
- A random opponent bankrupts itself → standing still wins → $3,000 flat with zero effort
- 720-step credit assignment: random exploration never stumbles the full chain
  *buy→walk→plant→water→water→harvest→sell* by luck in 32k steps
- Competition forum thread #741258: confirmed experimentally at 100k–800k steps

Day-level (30 decisions, each executed by the deterministic controller):
- Even a **random** day-level policy earns $4,600–$6,000 — just picking a coherent
  crop for 24 turns at a time is enough to stumble into some farming
- PPO only has to learn *which* crop and *when* to sell — not how to walk and water
- Confirmed: 100% win rate vs random from scratch in 200k timesteps

### Why `starter` as opponent, not `random`

`random` opponent bankrupts itself — "do nothing" stays a stable optimum.
`starter` (the built-in carrot loop) earns ~$3,498 and doesn't self-destruct.
Beating `starter` requires actual farming. The degenerate optimum disappears.

### Why shared trunk, two heads

SB3's default builds actor and critic as two independent MLPs. After any imitation
learning or warm-starting, only the actor has learned features; the critic is random
noise. PPO's advantage = `r + γV(s') − V(s)` — if V is noise, every advantage is
noise, and gradient steps are random. Result: the actor degrades. One shared trunk
means both heads share the same learned state representation from day one.

### Why four independent heads instead of one joint 150-way head

`crop_focus × buy_land × n_hires × sell_mode` = 150 combinations. A joint 150-way
softmax assigns independent probability mass to every combo — the model must learn
"melon + land + 4 hires + dump" as a single atom. With four heads, crop and sell mode
are learned independently, which is the correct inductive bias: which crop you grow is
largely independent of how aggressively you sell. Four heads also have 15 parameters
vs 150 — more data-efficient at the start of training.

---

## File map

| File | What it does | Read it to learn |
|---|---|---|
| `features.py` | obs dict → float32 vector | feature engineering, normalisation |
| `day_controller.py` | macro → 24 action dicts | game mechanics, deterministic planning |
| `day_env.py` | gymnasium wrapper | RL environment interface |
| `model.py` | actor-critic MLP | neural network architecture, shared trunk |
| `ppo.py` | PPO algorithm | GAE, clipped surrogate, value clip, entropy |
| `train.py` | training loop | rollout → GAE → update → eval → checkpoint |
| `play.py` | visualise agent | what the policy actually learned |
| `analyze.py` | plot training run | learning curves, win rates, schedules |
| `scripts/bench.py` | env speed check | throughput gate before adding multiprocessing |
| `tests/` | pytest suite | correctness checks for every component |

---

## Common questions

**"The reward isn't going up."**
Check `play.py` output — is the agent farming at all, or standing still?
If standing still: the opponent is probably `random` (set `--opponent starter`).
If farming but not improving: run for more iterations (100+); day-level PPO takes
50–100 iterations to show clear improvement.

**"Loss is NaN."**
Training aborts and saves `latest.pth`. Usually caused by a very large reward spike
(e.g. 10x hire all at once). Lower `lr` to `1e-4` and resume: `--resume runs/.../latest.pth`.

**"Value loss is huge."**
Normal for the first 5–10 iterations. The critic starts at random initialisation and
must learn V(s) from scratch. It converges once it has enough rollout data.

**"entropy goes to 0 immediately."**
The policy went deterministic too fast. Increase `--ent-coef 0.05` for the next run.

---

## Phase 2 (after overnight run succeeds)

1. **Animals** — add `animal_focus` dim to macro; only uncapped compounding income source
2. **Self-play league** — freeze snapshots every N iters, sample opponents from pool
3. **Behavior cloning** — needs Kaggle auth + replay downloads; then `pretrain_critic()` before PPO

---

*Competition closed 2026-10-14. No submissions possible. This is for learning and local research.*
