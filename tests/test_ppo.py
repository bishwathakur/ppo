import math

import pytest
import torch

from day_env import DayEnv
from model import ActorCritic
from ppo import collect_rollout, compute_gae, ppo_update


def test_gae_matches_hand_computation():
    rewards = torch.tensor([[1.0, 1.0]])
    values = torch.tensor([[0.5, 0.4, 0.3]])
    dones = torch.tensor([[0.0, 0.0]])

    adv, ret = compute_gae(rewards, values, dones, gamma=0.5, lam=0.5)

    # t=1: delta = 1 + 0.5*0.3 - 0.4 = 0.75
    # t=0: delta = 1 + 0.5*0.4 - 0.5 = 0.70, A = 0.70 + 0.5*0.5*0.75 = 0.8875
    assert torch.allclose(adv, torch.tensor([[0.8875, 0.75]]))
    assert torch.allclose(ret, adv + values[:, :2])


def test_gae_stops_at_episode_boundary():
    adv, _ = compute_gae(
        torch.tensor([[1.0, 1.0]]),
        torch.tensor([[0.5, 0.4, 0.3]]),
        torch.tensor([[1.0, 0.0]]),
        gamma=0.5,
        lam=0.5,
    )
    # t=0 sees done=1, so the next state contributes nothing.
    assert torch.allclose(adv, torch.tensor([[0.5, 0.75]]))


def test_ppo_one_iteration():
    envs = [DayEnv(opponent="starter", seed=i) for i in range(2)]
    model = ActorCritic(hidden=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    before = [p.detach().clone() for p in model.parameters()]

    buffers, episode_money = collect_rollout(envs, model, torch.device("cpu"), T=30)
    assert len(episode_money) == 2
    advantages, returns = compute_gae(buffers["rewards"], buffers["values"], buffers["dones"])
    stats = ppo_update(model, optimizer, buffers, advantages, returns, K=2, batch_size=16)

    for key, value in stats.items():
        assert math.isfinite(value), f"{key} = {value}"
    assert 0.0 <= stats["clip_frac"] <= 1.0
    assert stats["entropy"] > 0.0
    assert any(not torch.equal(p, q) for p, q in zip(model.parameters(), before))
