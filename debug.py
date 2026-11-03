import re, types
from kaggle_environments import make

# load main_v17.py with traceback printing injected into every `except Exception:`
src = open("main_v17.py", encoding="utf-8").read()
src = re.sub(r"^([ \t]*)except Exception:[ \t]*\r?$",
             r'\1except Exception:\n\1    __import__("traceback").print_exc()',
             src, flags=re.M)
m = types.ModuleType("opp")
m.__file__ = "main_v17.py"
exec(compile(src, "main_v17_dbg", "exec"), m.__dict__)

env = make("kaggriculture", debug=True)
env.reset()

o1 = dict(env.state[1].observation); o1["step"] = env.state[0].observation["step"]
print("seat1 obs keys:", list(o1.keys()))
print("player:", o1.get("player"), "| step:", o1.get("step"), "| farms:", len(o1.get("farms", [])))

for t in range(8):
    o1 = dict(env.state[1].observation)
    o1["step"] = env.state[0].observation["step"]
    a0 = m.agent(env.state[0].observation)
    a1 = m.agent(o1)
    print(f"t={t} seat0 farmer={a0['farmer']} market={len(a0['market'])} | "
          f"seat1 farmer={a1['farmer']} market={len(a1['market'])}")
    env.step([a0, a1])
    f = env.state[0].observation["farms"]
    print(f"     money seat0=${f[0]['money']:.0f} seat1=${f[1]['money']:.0f}")