"""Training entry point.

    python train.py --iterations 200 --envs 8 --out runs/default

Rollout -> GAE -> PPO update every iteration; evaluate against `starter` and
`random` every `--eval-every` iterations; keep `best.pth` by mean money vs
`starter` (win rate vs `random` saturates at 1.0 immediately and would freeze
the best checkpoint at iteration 0).
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch

from day_env import DayEnv
from features import FEAT_DIM
from model import ActorCritic
from ppo import collect_rollout, compute_gae, ppo_update

EVAL_SEEDS = list(range(100, 110))
RANDOM_EVAL_SEEDS = list(range(200, 210))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--opponent", default="starter", choices=["starter", "random"])
    parser.add_argument("--out", default="runs/default")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def eval_episodes(model, opponent, seeds, device):
    env = DayEnv(opponent=opponent)
    moneys, wins = [], []
    model.eval()
    with torch.no_grad():
        for seed in seeds:
            obs, info = env.reset(seed=seed)
            done = False
            while not done:
                logits, _ = model(torch.as_tensor(obs, device=device).unsqueeze(0))
                macro = model.make_dist(logits).mode()[0].cpu().numpy()
                obs, _, terminated, truncated, info = env.step(macro)
                done = terminated or truncated
            moneys.append(info["money"])
            wins.append(float(info["money"] > info["opponent_money"]))
    return float(np.mean(moneys)), float(np.mean(wins))


def save_checkpoint(path, model, optimizer, iteration, best_money):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        "best_money": best_money,
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }, path)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    envs = [DayEnv(opponent=args.opponent, seed=args.seed + i) for i in range(args.envs)]
    model = ActorCritic(feat_dim=FEAT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    start_iter, best_money = 0, -math.inf
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_iter = checkpoint["iteration"] + 1
        best_money = checkpoint["best_money"]
        torch.set_rng_state(checkpoint["torch_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
        print(f"resumed {args.resume} at iteration {start_iter}")

    metrics = []
    metrics_path = os.path.join(args.out, "metrics.json")
    if start_iter and os.path.exists(metrics_path):
        with open(metrics_path) as fh:
            metrics = json.load(fh)

    print(f"device={device} envs={args.envs} opponent={args.opponent} out={args.out}")

    for iteration in range(start_iter, args.iterations):
        # Entropy bonus decays so the policy can sharpen late in training.
        ent_coef = args.ent_coef * max(0.0, 1.0 - iteration / args.iterations)
        t0 = time.perf_counter()

        buffers, episode_money = collect_rollout(envs, model, device, T=30)
        advantages, returns = compute_gae(
            buffers["rewards"], buffers["values"], buffers["dones"], gamma=args.gamma
        )
        stats = ppo_update(model, optimizer, buffers, advantages, returns,
                           ent_coef=ent_coef, device=device)
        elapsed = time.perf_counter() - t0

        record = {
            "iteration": iteration,
            "ent_coef": ent_coef,
            "train_money": float(np.mean(episode_money)),
            "day_steps_per_s": round(args.envs * 30 / elapsed, 1),
            **stats,
        }

        if any(not math.isfinite(value) for value in stats.values()):
            save_checkpoint(os.path.join(args.out, "latest.pth"), model, optimizer, iteration, best_money)
            print(f"ABORT: non-finite loss at iteration {iteration}: {stats}")
            raise SystemExit(1)

        if iteration % args.eval_every == 0 or iteration == args.iterations - 1:
            money, win_rate = eval_episodes(model, "starter", EVAL_SEEDS, device)
            record["eval_money_starter"] = money
            record["eval_winrate_starter"] = win_rate
            if iteration % (args.eval_every * 5) == 0:
                random_money, random_win = eval_episodes(model, "random", RANDOM_EVAL_SEEDS, device)
                record["eval_money_random"] = random_money
                record["eval_winrate_random"] = random_win
            if money > best_money:
                best_money = money
                save_checkpoint(os.path.join(args.out, "best.pth"), model, optimizer, iteration, best_money)

        metrics.append(record)
        with open(metrics_path, "w") as fh:
            json.dump(metrics, fh, indent=1)
        save_checkpoint(os.path.join(args.out, "latest.pth"), model, optimizer, iteration, best_money)

        eval_text = ""
        if "eval_money_starter" in record:
            eval_text = (f" | vs_starter ${record['eval_money_starter']:.0f}"
                         f" ({record['eval_winrate_starter'] * 100:.0f}%)")
        print(
            f"iter {iteration:4d} | policy {stats['policy_loss']:+.3f} | value {stats['value_loss']:.3f}"
            f" | entropy {stats['entropy']:.3f} | clip {stats['clip_frac']:.2f}"
            f" | train ${record['train_money']:.0f}{eval_text}"
            f" | {record['day_steps_per_s']:.1f} d/s"
        )

    print(f"done. best vs starter: ${best_money:.0f}. write-up: python analyze.py --run {args.out}")


if __name__ == "__main__":
    main()
