from ocatari.core import OCAtari
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

import os
import cv2
import numpy as np
from ultralytics import YOLO

STEPS = 1500
MODEL_TAG = "train"

# ====== 1) 读取动作序列 ======
with open('models/path.in', 'r') as file:
    content = file.read().strip()
    expert_action = [int(num.strip()) for num in content.split(',')]

# ====== 2) 加载 YOLO 模型 ======
YOLO_WEIGHTS = "data/runs/detect/train/weights/best.pt"
detector = YOLO(YOLO_WEIGHTS)

CONF_THRES = 0.25
IOU_THRES = 0.5
IMGSZ = 160

# ====== 3) 创建环境 ======
env = OCAtari(
    "ALE/MontezumaRevenge-v5",
    mode="vision",
    hud=False,
    render_mode="rgb_array",
    buffer_window_size=1,
)

obs, info = env.reset()
frame = env.render()
h, w, _ = frame.shape

# ====== 4) 创建视频写入器 ======
os.makedirs("videos", exist_ok=True)
out_path = f"videos/yolo_annotation_{MODEL_TAG}_{STEPS}steps.mp4"

fps = 30
video = cv2.VideoWriter(
    out_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

# ====== 5) 可选：颜色 ======
def color_for_cls(cls_id: int):
    rng = np.random.default_rng(cls_id + 12345)
    c = rng.integers(60, 256, size=3)
    return int(c[0]), int(c[1]), int(c[2])

# ====== 6) 创建实时播放窗口 ======
WINDOW_NAME = "OCAtari Live (press 'q' to quit)"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
# 你也可以固定显示大小（不影响保存分辨率）
# cv2.resizeWindow(WINDOW_NAME, w * 3, h * 3)

try:
    obs, info = env.reset()

    for step in range(STEPS):
        action = expert_action[step] if step < len(expert_action) else 0
        obs, reward, terminated, truncated, info = env.step(action)

        frame = env.render()  # RGB uint8

        # ====== YOLO 推理 ======
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

        # ====== 实时播放：OpenCV 显示要 BGR ======
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow(WINDOW_NAME, bgr)

        # waitKey 决定刷新频率：这里用 fps 控制播放速度
        key = cv2.waitKey(int(1000 / fps)) & 0xFF

        # 按 q 退出
        if key == ord('q'):
            print("[INFO] Quit by user (q).")
            break

        # 用户点 X 关闭窗口也要退出
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            print("[INFO] Window closed by user.")
            break

        # ====== 写入录像 ======
        video.write(bgr)

        # 回合结束就 reset（你的原逻辑）
        if terminated or truncated:
            obs, info = env.reset()

finally:
    # ====== 7) 无论如何都要释放资源，确保能保存文件 ======
    video.release()
    env.close()
    cv2.destroyAllWindows()

print("Saved:", out_path)
