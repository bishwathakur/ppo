"""Observation dict -> fixed-length float32 vector.

FEAT_DIM is the single source of truth for the observation size. Every value is
normalised to a small range and clipped; missing keys raise KeyError rather than
being silently zero-filled (the competition engine version can drift).
"""

import numpy as np

from kaggle_environments.envs.kaggriculture.kaggriculture import MARKET_PARAMS

CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
BASE_PRICE = {p: MARKET_PARAMS[p]["base"] for p in PRODUCTS}
FEAT_DIM = 64

# n-th hire of the day costs fib(n): 1, 1, 2, 3, 5, 8, 13, ...
_HIRE_COST = [1, 1, 2, 3, 5, 8, 13]


def encode(obs, player):
    farm = obs["farms"][player]
    opp = obs["farms"][1 - player]
    private = obs["private"]
    market = obs["market"]
    town = obs["town"]
    day = int(obs["day"])
    tiles = farm["tiles"]
    money = float(farm["money"])
    opp_money = float(opp["money"])

    v = []
    # own farm
    v += [
        money / 10_000.0,
        len(farm["unlocked_quadrants"]) / 4.0,
        farm["hires_today"] / 10.0,
        farm["farmer"][0] / 10.0,
        farm["farmer"][1] / 10.0,
        len(farm["hands"]) / 10.0,
    ]

    # tile census over the full 10x10 board (LOCKED tiles counted in none)
    counts = dict.fromkeys(["NONE", "WEED", "STRUCT", *CROPS], 0)
    for row in tiles:
        for tile in row:
            if tile is None:
                counts["NONE"] += 1
            elif isinstance(tile, dict):
                kind = tile.get("kind")
                if kind == "PLANT":
                    counts[tile["crop"]] += 1
                elif kind == "WEED":
                    counts["WEED"] += 1
                else:
                    counts["STRUCT"] += 1
    v += [counts[k] / 100.0 for k in ("NONE", "WEED", "STRUCT", *CROPS)]

    # tile under the farmer
    fx, fy = farm["farmer"]
    tile = tiles[fy][fx]
    is_plant = isinstance(tile, dict) and tile.get("kind") == "PLANT"
    kind = tile.get("kind") if isinstance(tile, dict) else None
    v += [
        float(tile is None),
        float(is_plant),
        float(kind == "WEED"),
        float(isinstance(tile, dict) and kind not in ("PLANT", "WEED")),
    ]
    v += [float(is_plant and tile["crop"] == c) for c in CROPS]
    v += [
        min(max(day - tile["planted_day"], 0), 30) / 30.0 if is_plant else 0.0,
        float(tile["watered_today"]) if is_plant else 0.0,
    ]

    # private state (the observation already contains only this player's private data)
    seeds = private["seeds"]
    v += [seeds[c] / 10.0 for c in CROPS]
    shed = private["shed"]
    v += [shed[p] / 50.0 for p in PRODUCTS]

    # opponent public census
    v += [opp_money / 10_000.0, len(opp["unlocked_quadrants"]) / 4.0, len(opp["hands"]) / 10.0]

    # market: price relative to base, and inventory level
    prices = market["prices"]
    v += [prices[p] / BASE_PRICE[p] for p in PRODUCTS]
    inventory = market["inventory"]
    v += [inventory[p] / 100.0 for p in PRODUCTS]

    # town
    v += [len(town["unlocked_shops"]) / 8.0]

    # economy
    n_hired = int(farm["hires_today"])
    v += [
        day / 30.0,
        _HIRE_COST[min(n_hired, len(_HIRE_COST) - 1)] / 1000.0,
        (money - opp_money) / (abs(money) + abs(opp_money) + 1.0),
    ]

    out = np.asarray(v, dtype=np.float32)
    assert out.shape == (FEAT_DIM,), f"feature layout drift: got {out.shape}"
    return np.clip(out, -10.0, 10.0)
