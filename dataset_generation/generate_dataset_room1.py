#!/usr/bin/env python
# coding: utf-8

import random
# appends parent path to syspath to make ocatari importable
# like it would have been installed as a package
import sys
from copy import deepcopy
from os import path, makedirs
import torch
import matplotlib.pyplot as plt
import pandas as pd
# sys.path.append(path.dirname(path.dirname(path.abspath(__file__)))) # noqa
from ocatari.core import OCAtari
# from ocatari.utils import load_agent, parser, make_deterministic
from ocatari.utils import load_agent, make_deterministic
# from ocatari.vision.space_invaders import objects_colors
from ocatari.vision.pong import objects_colors
from ocatari.vision.utils import mark_bb, make_darker
import pickle
from tqdm import tqdm
from argparse import ArgumentParser  # 添加这行
import numpy as np
import cv2
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device="cpu"
'''
python dataset_generation/generate_dataset.py \
  -g ALE/MontezumaRevenge-v5 \
  -m ram \
  -hud\
  -dqn 
'''
# 创建 ArgumentParser 对象
parser = ArgumentParser(description='OCAtari Configuration')


parser.add_argument("-g", "--game", type=str, default="Pong",
                    help="game to evaluate (e.g. 'Pong')")
parser.add_argument("-i", "--interval", type=int, default=1000,
                    help="The frame interval (default 10)")
parser.add_argument("-m", "--mode", choices=["vision", "ram"],
                    default="ram", help="The frame interval")
parser.add_argument("-hud", "--hud", action="store_true",
                    default=True, help="Detect HUD")
parser.add_argument("-dqn", "--dqn", action="store_true",
                    default=True, help="Use DQN agent")
opts = parser.parse_args()

# Init the environment
env = OCAtari(opts.game, mode="both", render_mode='rgb_array', hud=True)
observation, info = env.reset()

# Set up an agent
if opts.dqn:
    opts.path = f"models/MDQN_modern_dqn/model_49750000.gz"
    # dqn_agent = load_agent(opts, env.action_space.n)
    dqn_agent,policy = load_agent(opt=opts, env=env ,device=device)
if hasattr(dqn_agent, "to"):
    dqn_agent = dqn_agent.to(device)
if hasattr(dqn_agent, "eval"):
    dqn_agent.eval()

# make environment deterministic
#env.step(2)
# make_deterministic(seed=42, env=env)
#make_deterministic(seed=42,mdp=env)
SEED = 42
observation, info = env.reset(seed=SEED)
env.action_space.seed(SEED)

stack = deque(maxlen=4)

def preprocess(frame_rgb: np.ndarray) -> np.ndarray:
    # frame_rgb: (210,160,3) uint8
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)          # (210,160)
    gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)  # (84,84)
    return gray  # uint8

# reset 后先填满 4 帧（用同一帧复制）
frame = env.render()
g = preprocess(frame)
for _ in range(4):
    stack.append(g)


# Init an empty dataset
game_nr = 0
turn_nr = 0
dataset = {"INDEX": [],  # "OBS": [],
           "RAM": [], "VIS": [], "HUD": []}
frames = []
r_objs = []
v_objs = []

# Generate 10,000 samples
for i in tqdm(range(10000)):
    # 取当前帧（渲染出来的 RGB）
    frame = env.render()
    g = preprocess(frame)
    stack.append(g)

    obs_stack = np.stack(list(stack), axis=0)     # (4,84,84) uint8

    # 转 torch： (1,4,84,84) float32 in [0,1]
    obs_t = torch.from_numpy(obs_stack).to(device).float().unsqueeze(0) / 255.0
    # print("frame", frame.shape, frame.dtype)
    # print("obs_t", obs_t.shape, obs_t.dtype, obs_t.min().item(), obs_t.max().item())

    action, _q = policy(obs_t, eps=0.0)
    action = int(action)
    obs, reward, terminated, truncated, info = env.step(action)

    # 更新 observation（你后面存数据可能还要用）
    observation = obs
    # make a short print every 1000 steps
    # if i % 1000 == 0:
    #    print(f"{i} done")

    step = f"{'%0.5d' % (game_nr)}_{'%0.5d' % (turn_nr)}"
    dataset["INDEX"].append(step)
    frame = env.render()              # RGB
    frames.append(deepcopy(frame))    # ✅ 存真正的图像
    r_objs.append(deepcopy(env.objects))
    v_objs.append(deepcopy(env.objects_v))
    # dataset["OBS"].append(observation.flatten().tolist())
    dataset["VIS"].append(
        [x for x in sorted(env.objects_v, key=lambda o: str(o))])
    dataset["RAM"].append(
        [x for x in sorted(env.objects, key=lambda o: str(o)) if x.hud == False])
    dataset["HUD"].append(
        [x for x in sorted(env.objects, key=lambda o: str(o)) if x.hud == True])
    turn_nr = turn_nr + 1

    # if a game is terminated, restart with a new game and update turn and game counter
    if terminated or truncated:
        observation, info = env.reset(seed=SEED)
        frame = env.render()
        g = preprocess(frame)
        stack.clear()
        for _ in range(4):
            stack.append(g)
        turn_nr = 0
        game_nr += 1


    # The interval defines how often images are saved as png files in addition to the dataset
    if i % opts.interval == 0:
        """
        print("-"*50)
        print(f"Frame {i}")
        print("-"*50)
        fig, axes = plt.subplots(1, 2)
        for observation, objects_list, title, ax in zip([observation,obs2], [env.objects, env.objects_v], ["ram", "vis"], axes):
            print(f"{title}: ", sorted(objects_list, key=lambda o: str(o)))
            for obj in objects_list:
                opos = obj.xywh
                ocol = obj.rgb
                sur_col = make_darker(ocol, 0.2)
                mark_bb(observation, opos, color=sur_col)
                # mark_point(observation, *opos[:2], color=(255, 255, 0))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(observation)
            ax.set_title(title)
        plt.suptitle(f"frame {i}", fontsize=20)
        plt.show()
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(1, 1, 1)
        for obj in env.objects:
            opos = obj.xywh
            ocol = obj.rgb
            sur_col = make_darker(ocol, 0.8)
            mark_bb(obs3, opos, color=sur_col)
        ax2.imshow(obs3)
        ax2.set_xticks([])
        ax2.set_yticks([])
        plt.show()
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(1, 1, 1)
        for obj in env.objects_v:
            opos = obj.xywh
            ocol = obj.rgb
            sur_col = make_darker(ocol, 0.8)
            mark_bb(obs4, opos, color=sur_col)
        ax3.imshow(obs4)
        ax3.set_xticks([])
        ax3.set_yticks([])
        plt.show()
        """
env.close()

df = pd.DataFrame(dataset, columns=['INDEX', 'RAM', 'HUD', 'VIS'])
makedirs("data/datasets/", exist_ok=True)
# prefix = f"{opts.game}_dqn" if opts.dqn else f"{opts.game}_random"
prefix="montezuma"
df.to_csv(f"data/datasets/{prefix}.csv", index=False)
pickle.dump(v_objs, open(f"data/datasets/{prefix}_objects_v.pkl", "wb"))
pickle.dump(r_objs, open(f"data/datasets/{prefix}_objects_r.pkl", "wb"))
pickle.dump(frames, open(f"data/datasets/{prefix}_frames.pkl", "wb"))
print(f"Finished {opts.game}")
