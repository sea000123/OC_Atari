#!/usr/bin/env python
# coding: utf-8
'''
python data/generate_dataset.py 
'''
from copy import deepcopy
from os import makedirs
import pickle
from tqdm import tqdm
from argparse import ArgumentParser
import pandas as pd
from ocatari.core import OCAtari

# ---------------- env ----------------
SEED = 42
env = OCAtari(
    "ALE/MontezumaRevenge-v5",
    mode="both",
    render_mode="rgb_array",
    hud=True,
    buffer_window_size=1,   
)
# ---------------- load actions ----------------
with open("models/path.in", "r") as f:
    content = f.read().strip()

tokens = [t.strip() for t in content.replace("\n", ",").split(",") if t.strip() != ""]
expert_action = [int(t) for t in tokens]
expert_action = expert_action[:1000] # 只取前 1500 个动作

print(f"[path.in] loaded {len(expert_action)} actions")
steps = len(expert_action)

# ---------------- dataset containers ----------------
game_nr = 0
turn_nr = 0

dataset = {"INDEX": [], "RAM": [], "VIS": [], "HUD": []}
frames = []
r_objs = []
v_objs = []

# ---------------- main loop ----------------
for j in range(5):
    obs, info = env.reset(seed=SEED)
    env.action_space.seed(SEED)
    for i in tqdm(range(steps)):
        action = int(expert_action[i])

        obs, reward, terminated, truncated, info = env.step(action)

        step_id = f"{game_nr:05d}_{turn_nr:05d}"
        dataset["INDEX"].append(step_id)

        # 保存 step 之后的画面（与 objects 对齐）
        frame_after = env.render()
        frames.append(deepcopy(frame_after))

        # 保存对象（buffer_window_size=1 会影响 objects 稳定化窗口）
        r_objs.append(deepcopy(env.objects))
        v_objs.append(deepcopy(env.objects_v))

        dataset["VIS"].append([x for x in sorted(env.objects_v, key=lambda o: str(o))])
        dataset["RAM"].append([x for x in sorted(env.objects, key=lambda o: str(o)) if x.hud is False])
        dataset["HUD"].append([x for x in sorted(env.objects, key=lambda o: str(o)) if x.hud is True])

        turn_nr += 1

        if terminated or truncated:
            obs, info = env.reset(seed=SEED)
            turn_nr = 0
            game_nr += 1
# --------- debug / print categories ----------
print("\n================= Dataset keys =================")
print(list(dataset.keys()))
print("CSV columns:", ["INDEX", "RAM", "HUD", "VIS"])
print(f"Num samples: {len(dataset['INDEX'])}")

print("\n================= Env attributes (filtered) =================")
# 打印 env 上所有属性名（过滤掉大量 __xxx__ 和函数）
attrs = [a for a in dir(env) if not a.startswith("__")]
# 只保留“看起来像数据”的（排除可调用的函数/方法）
data_attrs = []
for a in attrs:
    try:
        v = getattr(env, a)
        if callable(v):
            continue
        data_attrs.append(a)
    except Exception:
        pass

# 更关心的关键字段放前面
priority = ["mode", "hud", "buffer_window_size", "objects", "objects_v",
            "action_space", "observation_space", "ale", "env", "unwrapped"]
print("Priority attrs present:")
for p in priority:
    if hasattr(env, p):
        try:
            v = getattr(env, p)
            # 不要把大对象全打印出来
            if p in ["objects", "objects_v"]:
                print(f" - {p}: type={type(v)}, len={len(v) if v is not None else None}")
            else:
                print(f" - {p}: type={type(v)}")
        except Exception as e:
            print(f" - {p}: <error> {e}")

print("\nOther non-callable attrs (names only):")
print(data_attrs)

print("\n================= Last obs/info summary =================")
# 注意：这里假设你 loop 里最后一次 step 的 obs/info 仍在作用域中
try:
    import numpy as np
    if isinstance(obs, np.ndarray):
        print(f"obs: np.ndarray shape={obs.shape}, dtype={obs.dtype}")
    else:
        print(f"obs: type={type(obs)}")
except Exception as e:
    print("obs: <error>", e)

try:
    if isinstance(info, dict):
        print("info keys:", list(info.keys()))
    else:
        print(f"info: type={type(info)}")
except Exception as e:
    print("info: <error>", e)

print("\n================= What you DID store =================")
print("Stored frames: frames.pkl (RGB images from env.render())")
print("Stored objects: objects_r.pkl (env.objects), objects_v.pkl (env.objects_v)")
print("Stored csv cols: INDEX/RAM/HUD/VIS (object lists serialized)")
print("======================================================\n")
# -----------------------------------------

env.close()

def collect_classes(objs, skip_hud=False):
    s = set()
    for obj_list in objs:
        for o in obj_list:
            if skip_hud and getattr(o, "hud", False):
                continue
            s.add(o.__class__.__name__)
    return sorted(s)

classes_r_all  = collect_classes(r_objs, skip_hud=False)
classes_r_nohud= collect_classes(r_objs, skip_hud=True)

classes_v_all  = collect_classes(v_objs, skip_hud=False)
classes_v_nohud= collect_classes(v_objs, skip_hud=True)

classes_union_nohud = sorted(set(classes_r_nohud) | set(classes_v_nohud))

print("RAM classes (all):", classes_r_all)
print("RAM classes (no HUD):", classes_r_nohud)
print("VIS classes (all):", classes_v_all)
print("VIS classes (no HUD):", classes_v_nohud)
print("UNION (RAM+VIS, no HUD):", classes_union_nohud)

name2id = {n:i for i,n in enumerate(classes_union_nohud)}
print("name2id:", name2id)


# ---------------- save ----------------
df = pd.DataFrame(dataset, columns=["INDEX", "RAM", "HUD", "VIS"])

makedirs("data/datasets/", exist_ok=True)
prefix = "montezuma"  # 原来固定成 montezuma

df.to_csv(f"data/datasets/{prefix}.csv", index=False)
pickle.dump(v_objs, open(f"data/datasets/{prefix}_objects_v.pkl", "wb"))
pickle.dump(r_objs, open(f"data/datasets/{prefix}_objects_r.pkl", "wb"))
pickle.dump(frames, open(f"data/datasets/{prefix}_frames.pkl", "wb"))

print(f"Finished montezuma -> data/datasets/{prefix}.*")
