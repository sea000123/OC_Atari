from ocatari.core import OCAtari
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
import cv2
import minari

STEPS=3000
MODEL="expert"
if MODEL=="expert":
    ds = minari.load_dataset(
        "atari/montezumarevenge/expert-v0",
    )
    ep=ds[-1] # length 1204
    ep_actions = ep.actions 
    expert_action = ep_actions.tolist()
    expert_action =[0]
else:
    with open('path/path.in', 'r') as file:
        content = file.read().strip()
        expert_action = [int(num.strip()) for num in content.split(',')]
    # length 3086
env = OCAtari(
    "ALE/MontezumaRevenge-v5",
    mode="vision",
    hud=False,
    render_mode="rgb_array",
    buffer_window_size=1,
)

obs, info = env.reset()

# 用 render() 拿 RGB 帧
frame = env.render()
h, w, _ = frame.shape

video = cv2.VideoWriter(
    f"videos/vem_visualization_{MODEL}_{STEPS}steps.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    30,
    (w, h)
)
obs, info = env.reset()
for step in range(STEPS):
    #action = env.action_space.sample()
    action=expert_action[step]
    obs, reward, terminated, truncated, info = env.step(action)

    frame = env.render()

    for obj in env.objects:
        x, y = obj.xy
        w_obj, h_obj = obj.wh

        cv2.rectangle(
            frame,
            (x, y),
            (x + w_obj, y + h_obj),
            (0, 255, 0),
            1
        )

        cv2.putText(
            frame,
            obj.category,
            (x, max(y - 4, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1
        )

    video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    if terminated or truncated:
        obs, info = env.reset()

video.release()
env.close()
print("Saved vem_visualization.mp4")
