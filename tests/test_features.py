import numpy as np
import pytest
from kaggle_environments import make

import features


@pytest.fixture(scope="module")
def reset_obs():
    env = make("kaggriculture", debug=False)
    env.reset()
    return [s.observation for s in env.state]


@pytest.fixture(scope="module")
def episode():
    env = make("kaggriculture", debug=False)
    env.run(["starter", "starter"])
    return env


def test_features_shape(reset_obs):
    for player in (0, 1):
        vec = features.encode(reset_obs[player], player)
        assert isinstance(vec, np.ndarray)
        assert vec.dtype == np.float32
        assert vec.shape == (features.FEAT_DIM,)


def test_features_finite(reset_obs):
    for player in (0, 1):
        vec = features.encode(reset_obs[player], player)
        assert np.all(np.isfinite(vec))
        assert not np.any(np.isnan(vec))
        assert np.all(np.abs(vec) <= 10.0), f"unbounded feature: max={np.abs(vec).max()}"

def _step_with_plant_and_weed(episode):
    for i, step in enumerate(episode.steps):
        tiles = step[0].observation["farms"][0]["tiles"]
        kinds = {t.get("kind") if isinstance(t, dict) else t for row in tiles for t in row}
        if "PLANT" in kinds and "WEED" in kinds:
            return i
    return None

def test_features_mid_episode(episode):
    # Day 3: starter has planted, watered and harvested, and weeds exist.
    idx = _step_with_plant_and_weed(episode)
    assert idx is not None, "no step in this episode has both PLANT and WEED"
    obs = [s.observation for s in episode.steps[idx]]
    # obs = [s.observation for s in episode.steps[80]]
    tiles = obs[0]["farms"][0]["tiles"]
    kinds = {tile.get("kind") if isinstance(tile, dict) else tile for row in tiles for tile in row}
    assert "PLANT" in kinds and "WEED" in kinds, f"test does not cover plant/weed branches: {kinds}"
    for player in (0, 1):
        vec = features.encode(obs[player], player)
        assert vec.shape == (features.FEAT_DIM,)
        assert np.all(np.isfinite(vec))


def test_features_schema_error(reset_obs):
    broken = dict(reset_obs[0])
    del broken["market"]
    with pytest.raises(KeyError):
        features.encode(broken, 0)
