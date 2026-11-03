"""Watch one episode: the primary learning and debugging tool.

    python play.py                          # untrained (random-ish) policy
    python play.py --checkpoint runs/short/best.pth
    python play.py --checkpoint runs/short/best.pth --ascii
"""

import argparse
from collections import Counter

import torch

from day_env import DayEnv
from model import ActorCritic

CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
LANDS = ["no", "yes"]
SELLS = ["hold", "normal", "dump"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    # parser.add_argument("--opponent", default="starter", choices=["starter", "random"])
    parser.add_argument("--opponent", default="starter",
                    help="'starter', 'random', or a path to an agent .py file")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ascii", action="store_true")
    return parser.parse_args()


def tile_char(tile):
    if tile is None:
        return "."
    if tile == "LOCKED":
        return "#"
    kind = tile.get("kind")
    if kind == "WEED":
        return "x"
    if kind == "PLANT":
        return tile["crop"][0].lower()
    if "animal" in tile:
        return tile["animal"][0]
    return "S"


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActorCritic().to(device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint.get("model", checkpoint))
    model.eval()

    env = DayEnv(opponent=args.opponent, seed=args.seed)
    obs, info = env.reset(seed=args.seed)
    rows = []
    done = False
    with torch.no_grad():
        while not done:
            day = info["day"] + 1
            logits, _ = model(torch.as_tensor(obs, device=device).unsqueeze(0))
            macro = model.make_dist(logits).mode()[0].cpu().numpy()
            obs, _, terminated, truncated, info = env.step(macro)
            rows.append((day, [int(v) for v in macro], info["money"], info["opponent_money"]))
            done = terminated or truncated

    print("day | crop       land hires sell   |      money |  opponent")
    for day, (crop, land, hires, sell), money, opp in rows:
        print(f"{day:3d} | {CROPS[crop]:<10s} {LANDS[land]:<4s} {hires:5d} {SELLS[sell]:<6s} |"
              f" ${money:9.0f} | ${opp:9.0f}")

    print("\nLearned schedule")
    for label, lo, hi in [("days  1-10", 0, 10), ("days 11-20", 10, 20), ("days 21-30", 20, 30)]:
        bucket = [rows[i] for i in range(lo, min(hi, len(rows)))]
        crop = Counter(CROPS[m[0]] for _, m, _, _ in bucket).most_common(1)[0][0]
        sell = Counter(SELLS[m[3]] for _, m, _, _ in bucket).most_common(1)[0][0]
        hires = Counter(m[2] for _, m, _, _ in bucket).most_common(1)[0][0]
        print(f"  {label}: {crop}, hires={hires}, sell={sell}")

    money, opp = rows[-1][2], rows[-1][3]
    print(f"  Land bought: {'yes' if any(m[1] for _, m, _, _ in rows) else 'no'}")
    print(f"  Final money: ${money:.0f} | Opponent: ${opp:.0f} | {'WIN' if money > opp else 'LOSS'}")

    if args.ascii:
        tiles = env._env.state[0].observation["farms"][0]["tiles"]
        print("\nFarm ( .=empty x=weed #=locked letter=crop S=structure)")
        for row in tiles:
            print("  " + " ".join(tile_char(tile) for tile in row))


if __name__ == "__main__":
    main()
