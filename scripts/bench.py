"""Measure direct kaggriculture env throughput (no env.run()).

The spec gate: >= 3 day-steps/s is OK, slower means add multiprocessing.
Run: python scripts/bench.py
"""

import time

from kaggle_environments import make

PASS_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}
EPISODES = 3
STEPS_PER_DAY = 24


def measure():
    env = make("kaggriculture", debug=False)
    actions = [PASS_ACTION, PASS_ACTION]

    t0 = time.perf_counter()
    total_steps = 0
    for _ in range(EPISODES):
        env.reset()
        # 719 calls per episode: state `step` 0..719 inclusive, and the terminal
        # state's action is never applied (episodeSteps=720 counts the reset state).
        while not env.done:
            env.step(actions)
            total_steps += 1
    elapsed = time.perf_counter() - t0

    rate = (total_steps / STEPS_PER_DAY) / elapsed if elapsed else 0.0
    print(f"{EPISODES} episodes / {total_steps} engine steps in {elapsed:.1f}s")
    print(f"Throughput: {rate:.1f} day-steps/s ({total_steps / elapsed:.0f} engine steps/s)")
    print(f"Verdict: {'OK' if rate >= 3 else 'SLOW - add SubprocVecEnv'}")
    return rate


if __name__ == "__main__":
    measure()
