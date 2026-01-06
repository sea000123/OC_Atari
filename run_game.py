from perception.core import Perception
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
import cv2
import minari

STEPS=3000
MODEL="standard"
if MODEL=="expert":
    ds = minari.load_dataset(
        "atari/montezumarevenge/expert-v0",
    )
    ep=ds[-1] # length 1204
    ep_actions = ep.actions 
    expert_action = ep_actions.tolist()
else:
    with open('models/path.in', 'r') as file:
        content = file.read().strip()
        expert_action = [int(num.strip()) for num in content.split(',')]
    # length 3086
env = Perception(
    "ALE/MontezumaRevenge-v5",
    mode="vision",
    hud=True,
    render_mode="rgb_array",
    buffer_window_size=1,
)

obs, info = env.reset()
frame = env.render()
h, w, _ = frame.shape

video = cv2.VideoWriter(
    f"videos/vem_{MODEL}_{STEPS}steps.mp4",
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
        if str(obj.category) == "Life" or str(obj.category) == "Score":
            # print("1",end="")
            continue

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
print(f"videos/vem_{MODEL}_{STEPS}steps.mp4")
