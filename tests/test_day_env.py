import numpy as np
import pytest

import features
from day_env import DayEnv

MACRO = np.array([4, 0, 4, 2])  # MELON, no land, 4 hires, dump


@pytest.fixture(scope="module")
def episode():
    env = DayEnv(opponent="starter", seed=0)
    obs, info = env.reset()
    rewards, steps, done = [], 0, False
    first_obs = obs
    while not done:
        obs, reward, terminated, truncated, info = env.step(MACRO)
        rewards.append(reward)
        steps += 1
        done = terminated or truncated
    return env, steps, rewards, first_obs, info


def test_episode_length(episode):
    _, steps, _, _, _ = episode
    assert steps == 30


def test_reward_sum(episode):
    _, _, rewards, _, info = episode
    assert abs(sum(rewards) - (info["money"] - 3000)) < 1.0
    assert info["money"] > 4000, f"executor should farm, got ${info['money']}"


def test_obs_shape(episode):
    _, _, _, first_obs, _ = episode
    assert first_obs.shape == (features.FEAT_DIM,)
    assert np.all(np.isfinite(first_obs))


def test_reset_twice():
    env = DayEnv(opponent="starter", seed=3)
    for _ in range(2):
        obs, info = env.reset()
        assert obs.shape == (features.FEAT_DIM,)
        assert np.all(np.isfinite(obs))
        assert info["money"] == 3000.0
        assert info["day"] == 0


def test_action_space_matches_macro():
    env = DayEnv()
    assert list(env.action_space.nvec) == [5, 2, 5, 3]
