"""Post-run analysis: learning curves, win rate, text summary.

    python analyze.py --run runs/overnight
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="runs/default")
    parser.add_argument("--out", default="analysis")
    return parser.parse_args()


def eval_points(metrics, key):
    xs = [m["iteration"] for m in metrics if key in m]
    ys = [m[key] for m in metrics if key in m]
    return xs, ys


def main():
    args = parse_args()
    with open(os.path.join(args.run, "metrics.json")) as fh:
        metrics = json.load(fh)
    os.makedirs(args.out, exist_ok=True)
    iters = [m["iteration"] for m in metrics]

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [
        (axes[0][0], "train_money", "Train money (mean of episode)", "$"),
        (axes[0][1], "policy_loss", "Policy loss", ""),
        (axes[1][0], "value_loss", "Value loss", ""),
        (axes[1][1], "entropy", "Entropy", ""),
    ]
    for axis, key, title, unit in panels:
        values = [m.get(key) for m in metrics]
        axis.plot(iters, values, color="#2a6f97")
        axis.set_title(title)
        axis.set_xlabel("iteration")
        if unit:
            axis.set_ylabel(unit)
        axis.grid(alpha=0.3)
    figure.tight_layout()
    curve_path = os.path.join(args.out, "learning_curve.png")
    figure.savefig(curve_path, dpi=120)
    plt.close(figure)

    xs, money = eval_points(metrics, "eval_money_starter")
    _, win_rate = eval_points(metrics, "eval_winrate_starter")
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(xs, money, marker="o", color="#2a6f97", label="mean money vs starter")
    axis.axhline(3000, color="grey", linestyle="--", label="start money")
    axis.set_xlabel("iteration")
    axis.set_ylabel("final money ($)")
    axis.grid(alpha=0.3)
    if win_rate:
        twin = axis.twinx()
        twin.plot(xs, [w * 100 for w in win_rate], marker="s", color="#e07a5f", label="win rate")
        twin.set_ylabel("win rate (%)")
        twin.set_ylim(0, 105)
    axis.set_title("Evaluation vs starter")
    figure.legend(loc="upper left")
    figure.tight_layout()
    winrate_path = os.path.join(args.out, "winrate.png")
    figure.savefig(winrate_path, dpi=120)
    plt.close(figure)

    best = max(metrics, key=lambda m: m.get("eval_money_starter", -float("inf")))
    last = metrics[-1]
    lines = [
        f"# Run summary: {args.run}",
        "",
        f"- Iterations: {len(metrics)}",
        f"- Mean collection speed: {last.get('day_steps_per_s', 0):.1f} day-steps/s",
        f"- Best eval vs starter: **${best.get('eval_money_starter', 0):.0f}** at iteration "
        f"{best['iteration']} (win rate {best.get('eval_winrate_starter', 0) * 100:.0f}%)",
        f"- Last eval vs starter: ${last.get('eval_money_starter', 0):.0f} "
        f"(win rate {last.get('eval_winrate_starter', 0) * 100:.0f}%)",
        f"- Last train money: ${last.get('train_money', 0):.0f}",
        f"- Final entropy: {last.get('entropy', 0):.3f}",
        "",
        f"Learned day-by-day schedule: `python play.py --checkpoint {args.run}/best.pth --ascii`",
        "",
        f"![learning curve](learning_curve.png)",
        f"![win rate](winrate.png)",
    ]
    summary_path = os.path.join(args.out, "summary.md")
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print("\n".join(lines[:8]))
    print(f"\nwrote {curve_path}, {winrate_path}, {summary_path}")


if __name__ == "__main__":
    main()
