import sys, numpy as np, torch
from day_env import DayEnv
from model import ActorCritic

def run(ckpt, opponent, seeds):
    model = ActorCritic()
    if ckpt:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ck.get("model", ck))
    model.eval()
    rows = []
    for s in seeds:
        env = DayEnv(opponent=opponent, seed=s)
        obs, info = env.reset(seed=s)
        done = False
        while not done:
            with torch.no_grad():
                logits, _ = model(torch.as_tensor(obs).unsqueeze(0))
                macro = model.make_dist(logits).mode()[0].numpy()
            obs, _, term, trunc, info = env.step(macro)
            done = term or trunc
        rows.append((info["money"], info["opponent_money"]))
    return np.array(rows)

if __name__ == "__main__":
    ckpt = sys.argv[1] if sys.argv[1] != "none" else None
    for opp in sys.argv[2:]:
        r = run(ckpt, opp, range(10))
        print(f"{opp:<14} mine={r[:,0].mean():9.0f}  opp={r[:,1].mean():9.0f}  "
              f"win={np.mean(r[:,0] > r[:,1]):.0%}")