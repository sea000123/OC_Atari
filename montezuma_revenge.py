# appends parent path to syspath to make ocatari importable
# like it would have been installed as a package
'''
python scripts/demo/demo_montezuma_revenge.py -p models/model_49750000
python montezuma_revenge.py
'''
import matplotlib.pyplot as plt
from ocatari.core import OCAtari
from ocatari.vision.utils import mark_bb, make_darker
from ocatari.utils import load_agent
import torch
import numpy as np
import os
import imageio.v2 as imageio

game_name = "MontezumaRevenge-v4"
MODE = "vision"
HUD = True
STEPS=5000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("videos", exist_ok=True)
video_path = os.path.join("videos", f"montezuma_run_mdqn_{STEPS}steps.mp4")
fps = 30
writer = imageio.get_writer(
    video_path,
    format="FFMPEG",         # ✅ 强制使用 ffmpeg 插件
    fps=fps,
    codec="libx264",
    quality=8
)  # quality 0-10
# 关键：用 dqn observation（84x84 灰度 + 4 帧堆栈）
env = OCAtari(
    game_name,
    mode=MODE,
    hud=HUD,
    obs_mode="dqn",               # <= 关键
    render_mode="rgb_array",
    buffer_window_size=4          # <= DQN 一般需要 4 帧
)
obs, info = env.reset()

agent,policy = load_agent(
    opt="models/MDQN_modern_dqn/model_49750000.gz", 
    env=env, device=device)
agent = agent.to(device).eval()

'''
import inspect
print("POLICY info:")
print(policy)                    # 看看 partial 绑定了什么
print(inspect.signature(policy.func))  # 这是 _epsilon_greedy 的签名
print("bound keywords:", policy.keywords)

functools.partial(<function _epsilon_greedy at 0x7518f30a0af0>, model=AtariNet(
  (_AtariNet__features): Sequential(
    (0): Conv2d(4, 32, kernel_size=(8, 8), stride=(4, 4))
    (1): ReLU(inplace=True)
    (2): Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
    (3): ReLU(inplace=True)
    (4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
    (5): ReLU(inplace=True)
  )
  (_AtariNet__head): Sequential(
    (0): Linear(in_features=3136, out_features=512, bias=True)
    (1): ReLU(inplace=True)
    (2): Linear(in_features=512, out_features=18, bias=True)
  )
))
(obs, model, eps=0)
bound keywords: {'model': AtariNet(
  (_AtariNet__features): Sequential(
    (0): Conv2d(4, 32, kernel_size=(8, 8), stride=(4, 4))
    (1): ReLU(inplace=True)
    (2): Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
    (3): ReLU(inplace=True)
    (4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
    (5): ReLU(inplace=True)
  )
  (_AtariNet__head): Sequential(
    (0): Linear(in_features=3136, out_features=512, bias=True)
    (1): ReLU(inplace=True)
    (2): Linear(in_features=512, out_features=18, bias=True)
  )
)}
'''
# level
# env._env.unwrapped.ale.setRAM(57, 9)

# room
# env._env.unwrapped.ale.setRAM(3, 10)

# player
# env._env.unwrapped.ale.setRAM(42, 150)
# env._env.unwrapped.ale.setRAM(43, 160)

# env._env.unwrapped.ale.setRAM(44, 100)
# env._env.unwrapped.ale.setRAM(45, 240)

# items
# env._env.unwrapped.ale.setRAM(65, 128)

with torch.no_grad():
    for i in range(STEPS):
        #action = policy(state=env.dqn_obs, epsilon=0.0)
        obs_t = torch.from_numpy(obs).to(device=device, dtype=torch.uint8).unsqueeze(0)
        action, _q = policy(obs_t, eps=0.0)
        action = int(action)

        obs, reward, terminated, truncated, info = env.step(action)

        if i % 5 == 0:
            #print(env.objects)
            # ram = env._env.unwrapped.ale.getRAM()
            # print(ram)
            # print(ram[34])
            # 你这里 mark_bb 画框的 obs 是 84x84 灰度（dqn），不是原 210x160 RGB
            # 如果你想在 RGB 上画框并显示，建议用 env.getScreenRGB()
            rgb = env.getScreenRGB()
            writer.append_data(rgb)

            for obj in env.objects:
                x, y = obj.xy
                if x < 160 and y < 210:
                    opos = obj.xywh
                    ocol = obj.rgb
                    sur_col = make_darker(ocol)
                    mark_bb(rgb, opos, color=sur_col)

            # grid = np.concatenate([obs[0], obs[1], obs[2], obs[3]], axis=1)  # (84, 336)
            # plt.imshow(grid, cmap="gray") # , cmap="gray"
            plt.imshow(rgb)
            plt.axis("off")
            #plt.show()

        if terminated or truncated:
            obs, info = env.reset()
        # modify and display render
writer.close()
env.close()
print("Saved video to:", video_path)
