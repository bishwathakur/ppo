import random
import copy

import pytest
from kaggle_environments import make
from kaggle_environments.envs.kaggriculture.kaggriculture import starter_agent

import day_controller


def _empty_tiles():
    return [
        [None if (x < 5 and y < 5) else "LOCKED" for x in range(10)]
        for y in range(10)
    ]


def _plant(crop, planted_day, watered_today=False, yield_units=1):
    return {
        "kind": "PLANT",
        "crop": crop,
        "planted_day": planted_day,
        "watered_today": watered_today,
        "yield_units": yield_units,
    }


def _obs(money=3000, seeds=None, shed=None, tiles=None, farmer=(4, 4), hands=(),
         hires_today=0, quadrants=("NW",), day=0, hour=0):
    farm = {
        "money": money,
        "tiles": tiles if tiles is not None else _empty_tiles(),
        "farmer": list(farmer),
        "hands": [list(h) for h in hands],
        "hires_today": hires_today,
        "unlocked_quadrants": list(quadrants),
    }
    opp = copy.deepcopy(farm)
    private = {
        "seeds": {c: (seeds or {}).get(c, 0) for c in
                  ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]},
        "shed": {p: (shed or {}).get(p, 0) for p in
                 ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
                  "EGG", "MILK", "WOOL", "FERTILIZER"]},
        "inventories": [{}],
    }
    prices = {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120,
              "MELON": 250, "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100}
    return {
        "player": 0,
        "day": day,
        "hour": hour,
        "farms": [farm, opp],
        "private": private,
        "market": {"prices": prices, "inventory": {p: 10000 for p in prices}},
        "town": {"unlocked_shops": []},
    }


def test_plan_length_and_market_only_on_first_turn():
    plan = day_controller.controller_actions([1, 0, 0, 1], _obs())
    assert len(plan) == 24
    assert all(turn["market"] == [] for turn in plan[1:])


def test_hire_shortfall():
    plan = day_controller.controller_actions([1, 0, 3, 1], _obs(hires_today=1))
    turn0 = plan[0]["market"]
    assert sum(1 for order in turn0 if order[0] == "HIRE") == 2
    assert all(order[0] != "HIRE" for turn in plan[1:] for order in turn["market"])


def test_sell_before_buy():
    obs = _obs(shed={"MELON": 10, "CARROT": 5})
    market = day_controller.controller_actions([4, 0, 2, 2], obs)[0]["market"]
    sells = [i for i, o in enumerate(market) if o[0] == "SELL"]
    buys = [i for i, o in enumerate(market) if o[0].startswith("BUY") or o[0] == "HIRE"]
    assert sells and buys
    assert max(sells) < min(buys)


def test_market_order_cap():
    shed = {p: 5 for p in ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
                           "EGG", "MILK", "WOOL", "FERTILIZER"]}
    market = day_controller.controller_actions([4, 0, 4, 2], _obs(shed=shed))[0]["market"]
    assert len(market) <= day_controller.MAX_ORDERS


def test_all_crops_watered():
    tiles = _empty_tiles()
    tiles[4][4] = _plant("CARROT", planted_day=0, watered_today=False, yield_units=2)
    obs = _obs(tiles=tiles)
    action = day_controller.per_turn_action([4, 0, 0, 0], obs)  # crop_focus=MELON
    assert action["farmer"] == ["WATER"]


def test_seeds_bought_for_focus_crop():
    market = day_controller.controller_actions([4, 0, 0, 1], _obs())[0]["market"]
    buys = [o for o in market if o[0] == "BUY_SEED"]
    assert buys and buys[0][1] == "MELON" and buys[0][2] > 0


def test_plant_requests_never_exceed_seeds():
    # The engine drops *all* PLANT requests for a crop when the turn's total
    # exceeds seeds held, so over-asking livelocks the farm forever.
    hands = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
    obs = _obs(seeds={"CARROT": 3}, hands=hands)
    action = day_controller.per_turn_action([1, 0, 0, 0], obs)
    plants = sum(1 for unit in [action["farmer"], *action["hands"]] if unit[0] == "PLANT")
    assert 0 < plants <= 3


def test_controller_legal():
    rng = random.Random(0)
    for trial in range(6):
        macro = [rng.randrange(5), rng.randrange(2), rng.randrange(5), rng.randrange(3)]
        env = make("kaggriculture", debug=False)
        env.reset()
        for _ in range(24):
            obs = env.state[0].observation
            action = day_controller.per_turn_action(macro, obs)
            env.step([action, starter_agent(env.state[1].observation)])
        assert len(env.steps) == 25, f"macro {macro}: engine advanced {len(env.steps) - 1} turns"
        assert not env.done
