from ocatari.core import OCAtari
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

import cv2
import numpy as np
from ultralytics import YOLO

STEPS = 800
MODEL_TAG = "train"

# 1) 读取动作序列（你原来的）
with open('models/path.in', 'r') as file:
    content = file.read().strip()
    expert_action = [int(num.strip()) for num in content.split(',')]

# 2) 加载你训练好的 YOLO 模型（改这里：换成你的 best.pt 路径）
YOLO_WEIGHTS = "data/runs/detect/train/weights/best.pt"
detector = YOLO(YOLO_WEIGHTS)

# 推理参数（可调）
CONF_THRES = 0.25
IOU_THRES = 0.5
IMGSZ = 160   # 你训练时用的 imgsz=160 就保持一致

# 3) 创建环境：只需要能 render 出 RGB 即可
env = OCAtari(
    "ALE/MontezumaRevenge-v5",
    mode="vision",           # 其实只要 render，mode 不关键
    hud=False,
    render_mode="rgb_array",
    buffer_window_size=1,
)

obs, info = env.reset()
frame = env.render()
h, w, _ = frame.shape

# 4) 创建视频写入器
video = cv2.VideoWriter(
    f"videos/yolo_annotation_{MODEL_TAG}_{STEPS}steps.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    30,
    (w, h)
)

# 可选：颜色（你也可以改成固定颜色）
def color_for_cls(cls_id: int):
    # 简单可复现的“伪随机颜色”
    rng = np.random.default_rng(cls_id + 12345)
    c = rng.integers(60, 256, size=3)
    return int(c[0]), int(c[1]), int(c[2])

obs, info = env.reset()
for step in range(STEPS):
    action = expert_action[step] if step < len(expert_action) else 0
    obs, reward, terminated, truncated, info = env.step(action)

    frame = env.render()  # RGB (H,W,3) uint8

    # 5) 用 YOLO 推理（关键改动）
    # verbose=False 防止刷屏
    results = detector.predict(
        source=frame,
        imgsz=IMGSZ,
        conf=CONF_THRES,
        iou=IOU_THRES,
        verbose=False
    )

    r = results[0]
    if r.boxes is not None and len(r.boxes) > 0:
        # xyxy: (N,4) in pixel coords
        boxes = r.boxes.xyxy.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()

        for (x1, y1, x2, y2), cid, cf in zip(boxes, cls_ids, confs):
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            name = detector.names.get(cid, str(cid))
            label = f"{name} {cf:.2f}"

            col = color_for_cls(cid)

            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            cv2.putText(
                frame, label,
                (x1, max(y1 - 5, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                col,
                1,
                cv2.LINE_AA
            )

    # 写视频（OpenCV 要 BGR）
    video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    if terminated or truncated:
        obs, info = env.reset()

video.release()
env.close()
print("Saved:", f"videos/yolo_annotation_{MODEL_TAG}_{STEPS}steps.mp4")
