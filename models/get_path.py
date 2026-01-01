import minari

# 先在命令行跑：minari download atari/montezumarevenge/expert-v0

ds = minari.load_dataset(
    "atari/montezumarevenge/expert-v0",
    download=True   # 关键：允许自动下载
)


ep = ds[1]                     # 第0条expert轨迹（也可以 ds[i]）
print("ep:", ep)
# actions = ep.actions           # 一般是 shape (T,) 的离散动作
# action_list = actions.tolist() # 你要的“数字组成的列表”

# print(len(action_list))
# print(action_list[:200])       # 先看前200个
# # print(action_list)           # 全量输出（会很长）

# for ep in ds:
#     print("Trajectory length:", len(ep))