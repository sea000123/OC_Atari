from perception.core import Perception
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

import os
import cv2
import numpy as np
from ultralytics import YOLO

# ================== CONFIG ==================
STEPS = 1500
MODEL_TAG = "train"

# "yolo" / "perception" / "none"
DETECTOR = "perception"
DETECTOR = "yolo"
if DETECTOR == "perception":
    USE_PERCPTRON = True
else:
    USE_PERCPTRON = False

CONF_THRES = 0.25
IOU_THRES = 0.5
IMGSZ = 160
FPS = 30

if DETECTOR not in ("yolo", "perception", "none"):
    raise ValueError("DETECTOR must be one of: 'yolo', 'perception', 'none'")

if DETECTOR == "perception" and not USE_PERCPTRON:
    raise ValueError("DETECTOR='perception' requires USE_PERCPTRON=True")

USE_YOLO = (DETECTOR == "yolo")
USE_PERCPTRON_OBJECTS = (DETECTOR == "perception")

# 1) 读取动作序列
with open('models/path.in', 'r') as file:
    content = file.read().strip()
    expert_action = [int(num.strip()) for num in content.split(',')]

# 2) 可选：加载 YOLO（只在 DETECTOR='yolo' 时）
detector = None
if USE_YOLO:
    YOLO_WEIGHTS = "data/runs/detect/train/weights/best.pt"
    detector = YOLO(YOLO_WEIGHTS)

# 3) 创建环境
if USE_PERCPTRON:
    env = Perception(
        "ALE/MontezumaRevenge-v5",
        mode="vision",
        hud=False,
        render_mode="rgb_array",
        buffer_window_size=1,
    )
else:
    env = gym.make(
        "ALE/MontezumaRevenge-v5",
        render_mode="rgb_array"
    )

obs, info = env.reset()
frame = env.render()
h, w, _ = frame.shape

# 4) 视频写入器
os.makedirs("videos", exist_ok=True)
out_path = f"videos/demo_{MODEL_TAG}_{STEPS}steps_{DETECTOR}.mp4"

video = cv2.VideoWriter(
    out_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    FPS,
    (w, h)
)

# 5) 颜色函数（YOLO 用）
def color_for_cls(cls_id: int):
    rng = np.random.default_rng(cls_id + 12345)
    c = rng.integers(60, 256, size=3)
    return int(c[0]), int(c[1]), int(c[2])

# 6) Perception objects 绘制函数（只在 DETECTOR='perception' 用）
def draw_perception_objects(frame_rgb, env):
    for obj in getattr(env, "objects", []):
        try:
            x, y = obj.xy
            w_obj, h_obj = obj.wh
            if x is None or y is None or w_obj is None or h_obj is None:
                continue
            x, y, w_obj, h_obj = int(x), int(y), int(w_obj), int(h_obj)

            cv2.rectangle(frame_rgb, (x, y), (x + w_obj, y + h_obj), (0, 255, 0), 1)
            cv2.putText(
                frame_rgb,
                getattr(obj, "category", "obj"),
                (x, max(y - 4, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
        except Exception:
            continue

# 7) 实时窗口
WINDOW_NAME = "Live Demo (q to quit)"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

try:
    obs, info = env.reset()

    for step in range(STEPS):
        action = expert_action[step] if step < len(expert_action) else 0
        obs, reward, terminated, truncated, info = env.step(action)

        frame = env.render()  # RGB

        # ✅ 互斥：只会走其中一个分支
        if USE_PERCPTRON_OBJECTS:
            draw_perception_objects(frame, env)

        elif USE_YOLO:
            results = detector.predict(
                source=frame,
                imgsz=IMGSZ,
                conf=CONF_THRES,
                iou=IOU_THRES,
                verbose=False
            )

            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
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

        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 实时显示
        cv2.imshow(WINDOW_NAME, bgr)
        key = cv2.waitKey(int(1000 / FPS)) & 0xFF
        if key == ord('q'):
            print("[INFO] Quit by user.")
            break
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            print("[INFO] Window closed.")
            break

        # 写视频
        video.write(bgr)

        if terminated or truncated:
            obs, info = env.reset()

finally:
    video.release()
    env.close()
    cv2.destroyAllWindows()

print("Saved:", out_path)