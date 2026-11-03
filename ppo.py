"""Hand-rolled PPO.

Structure mirrors `H:\\ml\\ppo\\Proximal_Policy_Optimization_Algorithms\\ppo.py`
(Breakout). Diffs, each marked in place:
  * gamma = 0.99 over 30 day-steps, not 720 frames (0.99^30 = 0.74 vs 0.0007).
  * Reward is the money delta, not `sign(reward)` — magnitude *is* the metric.
  * Four factorised action heads, so log_prob is a sum of four log-probs.
  * No frame stack / MaxAndSkip — the state is a 64-float vector.
"""

import numpy as np
import torch

from features import FEAT_DIM

# Rewards arrive in dollars (~thousands per day). Squaring them makes the value
# loss ~1e7 and swamps the shared trunk's policy gradient, so the critic is
# trained on thousands-of-dollars — the same scale `features.py` uses for money.
REWARD_SCALE = 1e-4


# ============================================================
# ROLLOUT COLLECTION
# ============================================================

def collect_rollout(envs, model, device, T=30):
    """Collect T day-steps from every env. No gradients here — just experience.

    Each DayEnv episode is exactly 30 days, so one call is one full episode per
    environment. Buffers stay on CPU (the model is tiny; moving minibatches is
    cheaper than keeping 8x30 copies on a device).
    """
    n = len(envs)
    buffers = {
        "obs": torch.zeros(n, T, FEAT_DIM),
        "actions": torch.zeros(n, T, 4, dtype=torch.long),
        "logp": torch.zeros(n, T),
        "rewards": torch.zeros(n, T),
        "values": torch.zeros(n, T + 1),
        "dones": torch.zeros(n, T),
    }
    episode_money = []

    obs = np.stack([env.reset()[0] for env in envs])
    model.eval()
    with torch.no_grad():
        for t in range(T):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            logits, values = model(obs_t)
            dist = model.make_dist(logits)
            actions = dist.sample()

            buffers["obs"][:, t] = obs_t.cpu()
            buffers["actions"][:, t] = actions.cpu()
            buffers["logp"][:, t] = dist.log_prob(actions).cpu()
            buffers["values"][:, t] = values.cpu()

            for i, env in enumerate(envs):
                next_obs, reward, terminated, truncated, info = env.step(actions[i].cpu().numpy())
                obs[i] = next_obs
                buffers["rewards"][i, t] = reward * REWARD_SCALE
                buffers["dones"][i, t] = float(terminated or truncated)
                if terminated or truncated:
                    episode_money.append(info["money"])
                    obs[i] = env.reset()[0]

        # Bootstrap value for the step after the horizon (masked by `dones`).
        final_obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        buffers["values"][:, T] = model(final_obs)[1].cpu()

    return buffers, episode_money


# ============================================================
# GENERALIZED ADVANTAGE ESTIMATION
# ============================================================
#
#   delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
#   A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
#
# lambda trades bias for variance: lambda=0 is one-step TD, lambda=1 is Monte
# Carlo. (1 - done) zeroes the bootstrap across episode boundaries.

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    n, T = rewards.shape
    advantages = torch.zeros(n, T)
    last_gae = torch.zeros(n)

    for t in reversed(range(T)):
        next_non_terminal = 1.0 - dones[:, t]
        delta = rewards[:, t] + gamma * values[:, t + 1] * next_non_terminal - values[:, t]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        advantages[:, t] = last_gae

    returns = advantages + values[:, :T]
    return advantages, returns


# ============================================================
# PPO UPDATE
# ============================================================
#
# policy_loss = -mean(min(ratio * A, clip(ratio, 1-eps, 1+eps) * A))
#   ratio = exp(new_log_prob - old_log_prob). Clipping stops a single update
#   from moving the policy so far that the old data stops being representative.
#
# value_loss  = 0.5 * mean(max((R - V)^2, (R - V_clipped)^2))
#   Same idea for the critic.
#
# loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
#   The entropy bonus keeps the policy from collapsing to deterministic early.

def ppo_update(model, optimizer, buffers, advantages, returns,
               K=4, batch_size=64, clip_range=0.2, vf_coef=0.5,
               ent_coef=0.01, max_grad_norm=0.5, device="cpu"):
    T = buffers["rewards"].shape[1]
    n = buffers["rewards"].shape[0]
    states = buffers["obs"].reshape(n * T, FEAT_DIM)
    actions = buffers["actions"].reshape(n * T, 4)
    old_logp = buffers["logp"].reshape(-1)
    old_values = buffers["values"][:, :T].reshape(-1)
    adv = advantages.reshape(-1)
    ret = returns.reshape(-1)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    total = states.shape[0]
    batch_size = min(batch_size, total)
    stats = {"policy_loss": [], "value_loss": [], "entropy": [], "clip_frac": []}

    model.train()
    for _ in range(K):
        perm = torch.randperm(total)
        for start in range(0, total, batch_size):
            idx = perm[start:start + batch_size]
            logits, values = model(states[idx].to(device))
            dist = model.make_dist(logits)

            new_logp = dist.log_prob(actions[idx].to(device))
            entropy = dist.entropy().mean()
            ratio = torch.exp(new_logp - old_logp[idx].to(device))

            policy_loss = -torch.min(
                ratio * adv[idx].to(device),
                torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv[idx].to(device),
            ).mean()

            value_clipped = old_values[idx].to(device) + torch.clamp(
                values - old_values[idx].to(device), -clip_range, clip_range
            )
            value_loss = 0.5 * torch.max(
                (ret[idx].to(device) - values).pow(2),
                (ret[idx].to(device) - value_clipped).pow(2),
            ).mean()

            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            stats["policy_loss"].append(policy_loss.item())
            stats["value_loss"].append(value_loss.item())
            stats["entropy"].append(entropy.item())
            stats["clip_frac"].append(
                ((ratio - 1.0).abs() > clip_range).float().mean().item()
            )

    return {key: float(np.mean(values)) for key, values in stats.items()}
