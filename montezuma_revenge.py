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
game_name = "MontezumaRevenge-v4"
MODE = "vision"
HUD = True
env = OCAtari(game_name, mode=MODE, hud=HUD,
               render_mode= "rgb_array",buffer_window_size=1)
obs, info = env.reset()

agent,policy = load_agent(opt="models/MDQN_modern_dqn/model_49750000.gz", 
                     env=env, device="cuda")
import inspect
print("POLICY info:")
print(policy)                    # 看看 partial 绑定了什么
print(inspect.signature(policy.func))  # 这是 _epsilon_greedy 的签名
print("bound keywords:", policy.keywords)

# level
# env._env.unwrapped.ale.setRAM(57, 9)

# room
# env._env.unwrapped.ale.setRAM(3, 10)

# player
# env._env.unwrapped.ale.setRAM(42, 150)
env._env.unwrapped.ale.setRAM(43, 160)

# env._env.unwrapped.ale.setRAM(44, 100)
# env._env.unwrapped.ale.setRAM(45, 240)

# items
env._env.unwrapped.ale.setRAM(65, 128)
agent.eval()
with torch.no_grad():
    for i in range(1000):
        #action = policy(state=env.dqn_obs, epsilon=0.0)
        action = int(policy(obs, eps=0.0)) 
        obs, reward, terminated, truncated, info = env.step(action)

        if i % 5 == 0:
            # obse2 = deepcopy(obse)
            print(env.objects)
            ram = env._env.unwrapped.ale.getRAM()
            # print(ram)
            print(ram[34])
            for obj in env.objects:
                x, y = obj.xy
                if x < 160 and y < 210:
                    opos = obj.xywh
                    ocol = obj.rgb
                    sur_col = make_darker(ocol)
                    mark_bb(obs, opos, color=sur_col)
                # mark_point(obs, *opos[:2], color=(255, 255, 0))

            plt.imshow(obs)
            plt.show()

        if terminated or truncated:
            observation, info = env.reset()
        # modify and display render
env.close()
