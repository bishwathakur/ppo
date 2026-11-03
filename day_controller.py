"""Deterministic executor: one day-level macro decision -> concrete per-turn actions.

`per_turn_action` is the real interface used by `day_env` (fresh observation every
turn). `controller_actions` returns the 24-entry plan named in the design spec by
replaying that function against a local copy of the mutable farm state.

Macro = [crop_focus, buy_land, n_hires, sell_mode]:
    crop_focus 0..4  WHEAT CARROT TOMATO STRAWBERRY MELON
    buy_land   0..1  attempt BUY_LAND when cash covers price + RESERVE
    n_hires    0..4  hands hired by the end of the day
    sell_mode  0..2  hold / normal (price >= base) / dump (sell everything)
"""

import copy

from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS as CROP_DATA

from features import BASE_PRICE, CROPS, PRODUCTS

RESERVE = 500
MAX_ORDERS = 10
LAND_PRICES = [1000, 2000, 4000]


def per_turn_action(macro, obs, cfg=None):
    """Action dict for the current turn, computed from the live observation."""
    cfg = cfg or {}
    crop_focus, buy_land, n_hires, sell_mode = (int(v) for v in macro)
    player = int(obs["player"])
    farm = obs["farms"][player]
    private = obs["private"]
    day = int(obs["day"])
    crop = CROPS[crop_focus]

    market = []
    if int(obs["hour"]) == 0:
        market = _market_orders(obs, farm, private, crop, buy_land, n_hires, sell_mode)

    units = [list(farm["farmer"])] + [list(h) for h in farm["hands"]]
    claimed = set()
    # The engine drops every PLANT order for a crop when the turn's total exceeds
    # seeds held, so units must share one per-turn budget.
    seed_budget = dict(private["seeds"])
    unit_actions = []
    for pos in units:
        action, target = _unit_action(pos, farm, seed_budget, crop, day, claimed)
        if target is not None:
            claimed.add(target)
        unit_actions.append(action)

    return {"farmer": unit_actions[0], "hands": unit_actions[1:], "market": market}


def controller_actions(macro, obs, cfg=None):
    """The 24-entry plan for one day, per the design spec."""
    cfg = cfg or {}
    turns = int(cfg.get("turnsPerDay", 24))
    sim = _sim_state(obs)
    plan = []
    for _ in range(turns):
        action = per_turn_action(macro, sim, cfg)
        plan.append(action)
        _sim_apply(action, sim)
    return plan


# ---------------------------------------------------------------- market

def _market_orders(obs, farm, private, crop, buy_land, n_hires, sell_mode):
    orders = []
    shed = private["shed"]
    prices = obs["market"]["prices"]

    # SELL first: the engine processes the list in order, so cash from sales is
    # available to the purchases that follow.
    if sell_mode != 0:
        sells = []
        for item in PRODUCTS:
            qty = shed.get(item, 0)
            if qty <= 0:
                continue
            if sell_mode == 2 or prices[item] >= BASE_PRICE[item]:
                sells.append((qty * prices[item], item, qty))
        sells.sort(reverse=True)
        orders += [["SELL", item, qty] for _, item, qty in sells]

    # Seeds are required for PLANT; the design spec omits this order, but without
    # it PLANT is a silent no-op and the farm never starts.
    seeds = private["seeds"]
    empty = sum(1 for row in farm["tiles"] for tile in row if tile is None)
    want = max(0, empty - seeds.get(crop, 0))
    cost = CROP_DATA[crop]["seed"]
    affordable = max(0, int((farm["money"] - RESERVE) // cost))
    seed_orders = []
    if want and affordable:
        seed_orders = [["BUY_SEED", crop, min(want, affordable)]]

    extra = len(farm["unlocked_quadrants"]) - 1
    land_orders = []
    if buy_land and extra < len(LAND_PRICES) and farm["money"] >= LAND_PRICES[extra] + RESERVE:
        land_orders = [["BUY_LAND"]]

    # HIRE only for the daily shortfall: upstream queued it on all 24 turns.
    shortfall = max(0, n_hires - farm["hires_today"])
    room = max(0, MAX_ORDERS - len(orders) - len(seed_orders) - len(land_orders))
    hire_orders = [["HIRE"]] * min(shortfall, room)

    return (orders + seed_orders + land_orders + hire_orders)[:MAX_ORDERS]


# ---------------------------------------------------------------- units

def _unit_action(pos, farm, seed_budget, crop, day, claimed):
    x, y = pos
    tile = farm["tiles"][y][x]
    action = _act(tile, seed_budget, crop, day)
    if action is not None:
        return action, None

    target = _nearest_task(farm, seed_budget, crop, day, claimed, x, y)
    if target is None:
        return ["PASS"], None
    tx, ty = target
    return _step_toward(x, y, tx, ty), target


def _act(tile, seed_budget, crop, day):
    if tile is None:
        if seed_budget.get(crop, 0) > 0:
            seed_budget[crop] -= 1
            return ["PLANT", crop]
        return None
    if not isinstance(tile, dict):
        return None  # LOCKED
    kind = tile.get("kind")
    if kind == "PLANT":
        if _ripe(tile, day):
            return ["HARVEST"]
        if not tile["watered_today"]:
            return ["WATER"]
        return None
    if kind == "WEED":
        return ["DIG"]
    return None  # animal structure: phase 2


def _ripe(tile, day):
    if tile.get("yield_units", 0) <= 0:
        return False
    crop_data = CROP_DATA[tile["crop"]]
    age = day - tile["planted_day"]
    if crop_data["ongoing"]:
        return age >= crop_data["first_yield_day"]
    return age >= crop_data["max_yield_day"]


def _priority(tile, seed_budget, crop, day):
    if tile is None:
        return 2 if seed_budget.get(crop, 0) > 0 else None
    if not isinstance(tile, dict):
        return None
    kind = tile.get("kind")
    if kind == "PLANT":
        if _ripe(tile, day):
            return 0
        return 1 if not tile["watered_today"] else None
    if kind == "WEED":
        return 3
    return None


def _nearest_task(farm, seed_budget, crop, day, claimed, x, y):
    """Strict priority (harvest > water > plant > dig), nearest tile within it."""
    best = None
    tiles = farm["tiles"]
    for ty, row in enumerate(tiles):
        for tx, tile in enumerate(row):
            if (tx, ty) in claimed:
                continue
            priority = _priority(tile, seed_budget, crop, day)
            if priority is None:
                continue
            key = (priority, abs(tx - x) + abs(ty - y), tx, ty)
            if best is None or key < best:
                best = key
    return None if best is None else (best[2], best[3])


def _step_toward(x, y, tx, ty):
    if tx > x:
        return ["EAST"]
    if tx < x:
        return ["WEST"]
    if ty > y:
        return ["SOUTH"]
    if ty < y:
        return ["NORTH"]
    return ["PASS"]


# ---------------------------------------------------------------- local replay

def _sim_state(obs):
    player = int(obs["player"])
    return {
        "player": player,
        "day": int(obs["day"]),
        "hour": int(obs["hour"]),
        "farms": copy.deepcopy(obs["farms"]),
        "private": copy.deepcopy(obs["private"]),
        "market": copy.deepcopy(obs["market"]),
        "town": copy.deepcopy(obs["town"]),
    }


def _sim_apply(action, sim):
    """Advance the local copy enough that the next turn's decision is sane."""
    crop_data_day = sim["day"]
    for unit_index, uni in enumerate([action["farmer"], *action["hands"]]):
        farm = sim["farms"][sim["player"]]
        pos = farm["farmer"] if unit_index == 0 else farm["hands"][unit_index - 1]
        x, y = pos
        if uni[0] in ("NORTH", "SOUTH", "EAST", "WEST"):
            dx, dy = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}[uni[0]]
            pos[0], pos[1] = x + dx, y + dy
            continue

        tile = farm["tiles"][y][x]
        if uni[0] == "PLANT" and tile is None and sim["private"]["seeds"].get(uni[1], 0) > 0:
            sim["private"]["seeds"][uni[1]] -= 1
            farm["tiles"][y][x] = {
                "kind": "PLANT", "crop": uni[1], "planted_day": crop_data_day,
                "watered_today": False, "yield_units": 1,
            }
        elif uni[0] == "WATER" and isinstance(tile, dict) and tile.get("kind") == "PLANT":
            tile["watered_today"] = True
            tile["yield_units"] = min(CROP_DATA[tile["crop"]]["max_yield"], tile["yield_units"] + 1)
        elif uni[0] == "HARVEST" and isinstance(tile, dict) and tile.get("kind") == "PLANT":
            if CROP_DATA[tile["crop"]]["ongoing"]:
                tile["yield_units"] = 0
            else:
                farm["tiles"][y][x] = None
        elif uni[0] == "DIG" and isinstance(tile, dict) and "animal" not in tile:
            farm["tiles"][y][x] = None

    farm = sim["farms"][sim["player"]]
    shed = sim["private"]["shed"]
    seeds = sim["private"]["seeds"]
    for order in action["market"]:
        op = order[0]
        if op == "SELL":
            qty = min(order[2], shed.get(order[1], 0))
            shed[order[1]] -= qty
            farm["money"] += qty * sim["market"]["prices"][order[1]]
        elif op == "BUY_SEED":
            qty = min(order[2], int(farm["money"] // CROP_DATA[order[1]]["seed"]))
            farm["money"] -= qty * CROP_DATA[order[1]]["seed"]
            seeds[order[1]] += qty
        elif op == "BUY_LAND":
            extra = len(farm["unlocked_quadrants"]) - 1
            if extra < len(LAND_PRICES) and farm["money"] >= LAND_PRICES[extra]:
                farm["money"] -= LAND_PRICES[extra]
                quad = ["NE", "SW", "SE"][extra]
                farm["unlocked_quadrants"].append(quad)
                half = len(farm["tiles"]) // 2
                for y in range(len(farm["tiles"])):
                    for x in range(len(farm["tiles"])):
                        if _quadrant(x, y, half) == quad and farm["tiles"][y][x] == "LOCKED":
                            farm["tiles"][y][x] = None
        elif op == "HIRE":
            farm["hires_today"] += 1

    sim["hour"] = (sim["hour"] + 1) % 24
    if sim["hour"] == 0:
        sim["day"] += 1


def _quadrant(x, y, half):
    return ("N" if y < half else "S") + ("W" if x < half else "E")
