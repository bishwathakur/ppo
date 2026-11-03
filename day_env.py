"""Gymnasium wrapper: one `step()` = one in-game day = 24 engine turns.

Seat 0 is the learner (macro action), seat 1 the opponent. The learner's macro is
re-expanded into concrete per-turn actions by `day_controller.per_turn_action`
against the live observation each turn.
"""

import traceback

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from kaggle_environments import make
from kaggle_environments.envs.kaggriculture.kaggriculture import random_agent, starter_agent

import importlib.util, sys
from pathlib import Path

from day_controller import per_turn_action
from features import FEAT_DIM, encode

OPPONENTS = {"starter": starter_agent, "random": random_agent}


def resolve_opponent(opponent):
    """'starter'/'random' stay as-is; a .py path becomes the agent callable."""
    if not isinstance(opponent, str) or not opponent.endswith(".py"):
        return opponent
    path = Path(opponent).resolve()
    sys.path.insert(0, str(path.parent))          # so sibling files import
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agent


class DayEnv(gym.Env):
    def __init__(self, opponent="starter", seed=0):
        self.observation_space = spaces.Box(-np.inf, np.inf, (FEAT_DIM,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([5, 2, 5, 3])
        # shortcut: callables are invoked every turn; a frozen self-play snapshot
        # must cache its own day decision (phase 2).
        # self._opponent = OPPONENTS[opponent] if isinstance(opponent, str) else opponent
        if isinstance(opponent, str) and opponent in OPPONENTS:
            self._opponent = OPPONENTS[opponent]
        else:
            self._opponent = resolve_opponent(opponent)
        self._seed = seed
        self._env = make("kaggriculture", debug=False)
        self._money = 0.0
        self._crashed = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
        self._env.seed = self._seed
        self._env.reset()
        self._crashed = False
        self._money = self._learner_money()
        return self._obs(), {"money": self._money, "day": 0}

    def _opp_obs(self):
        obs = dict(self._env.state[1].observation)
        obs["step"] = self._env.state[0].observation["step"]
        return obs

    def step(self, macro):
        terminated = False
        try:
            while not self._env.done:
                obs = self._env.state[0].observation
                # self._env.step([per_turn_action(macro, obs), self._opponent(self._env.state[1].observation)])
                self._env.step([per_turn_action(macro, obs), self._opponent(self._opp_obs())])
                if int(self._env.state[0].observation["hour"]) == 0:
                    break
            terminated = bool(self._env.done)
        except Exception:
            # One bad episode must not kill a multi-hour training run.
            if not self._crashed:
                traceback.print_exc()
                self._crashed = True
            terminated = True

        money = self._learner_money()
        reward = money - self._money
        self._money = money
        info = {
            "money": money,
            "opponent_money": float(self._env.state[1].observation["farms"][1]["money"]),
            "day": int(self._env.state[0].observation["day"]),
        }
        return self._obs(), float(reward), terminated, False, info

    def _learner_money(self):
        return float(self._env.state[0].observation["farms"][0]["money"])

    def _obs(self):
        return encode(self._env.state[0].observation, 0)
