import importlib.util
from pathlib import Path


_BASE_PATH = Path(__file__).with_name("main_v17.py")
_SPEC = importlib.util.spec_from_file_location("main_v17_base", _BASE_PATH)
_base = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_base)


def _wheat_stock(obs):
    seat = _base._seat(obs)
    farm = _base._farm(obs, seat)
    private = _base._get(obs, "private", {}) or {}
    shed = _base._get(private, "shed", {}) or {}
    inventories = _base._get(private, "inventories", []) or []
    animals = 0
    risk = 0
    for row in _base._get(farm, "tiles", []) or []:
        for tile in row if isinstance(row, list) else []:
            if not isinstance(tile, dict) or "animal" not in tile:
                continue
            animals += 1
            risk += int(not tile.get("fed_today") and int(tile.get("consecutive_unfed", 0) or 0) >= 1)
    carried = sum(int(_base._get(inv, "WHEAT", 0) or 0) for inv in inventories)
    return animals, risk, int(_base._get(shed, "WHEAT", 0) or 0), carried


def _feed_market_guard(obs, action):
    action = _base._copy_action(action)
    step = int(_base._get(obs, "step", 0) or 0)
    day = int(_base._get(obs, "day", step // 24) or 0)
    if day not in (23, 24, 25, 26):
        return action
    market = [list(order) for order in action.get("market", []) or []]
    if not any(len(order) >= 3 and order[0] == "SELL" and order[1] == "WHEAT" for order in market):
        return action
    animals, risk, shed_wheat, carried_wheat = _wheat_stock(obs)
    reserve = animals + 8 + risk * 2
    stock = shed_wheat + carried_wheat
    guarded = []
    for order in market:
        if len(order) >= 3 and order[0] == "SELL" and order[1] == "WHEAT":
            qty = min(max(0, int(order[2])), max(0, stock - reserve))
            stock -= qty
            if qty <= 0:
                continue
            order[2] = qty
        guarded.append(order)
    if animals and stock < reserve and len(guarded) < 10:
        guarded.append(["BUY_PRODUCT", "WHEAT", min(16, reserve - stock)])
    action["market"] = guarded[:10]
    return _base._align_hands(action, obs)


def _terminal_market_guard(obs, action):
    action = _base._copy_action(action)
    step = int(_base._get(obs, "step", 0) or 0)
    if step < 696:
        return action
    private = _base._get(obs, "private", {}) or {}
    shed = _base._get(private, "shed", {}) or {}
    market = [list(order) for order in action.get("market", []) or []]
    selling = {order[1] for order in market if len(order) >= 3 and order[0] == "SELL"}
    for item in ("WOOL", "MILK", "FERTILIZER", "MELON", "STRAWBERRY", "WHEAT"):
        qty = int(_base._get(shed, item, 0) or 0)
        if qty > 0 and item not in selling and len(market) < 10:
            market.append(["SELL", item, qty])
    action["market"] = market[:10]
    return _base._align_hands(action, obs)


def agent(obs, config=None):
    try:
        action = _base._top_agent(obs)
        action = _base._hybrid_repair(obs, action)
        action = _feed_market_guard(obs, action)
        action = _base._market_floor_guard(obs, action)
        action = _terminal_market_guard(obs, action)
        return _base._align_hands(action, obs)
    except Exception:
        farm = _base._farm(obs, _base._seat(obs))
        return {
            "farmer": ["PASS"],
            "hands": [["PASS"] for _ in (_base._get(farm, "hands", []) or [])],
            "market": [],
        }
