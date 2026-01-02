import os
import pickle
import cv2
from tqdm import tqdm
import random

FRAMES_PKL = "datasets/montezuma_frames.pkl"
OBJS_PKL   = "datasets/montezuma_objects_v.pkl"

OUT_DIR = "yolo_dataset"
IMG_DIR = os.path.join(OUT_DIR, "images")
LBL_DIR = os.path.join(OUT_DIR, "labels")

for split in ["train", "val"]:
    os.makedirs(os.path.join(IMG_DIR, split), exist_ok=True)
    os.makedirs(os.path.join(LBL_DIR, split), exist_ok=True)

frames = pickle.load(open(FRAMES_PKL, "rb"))      # list of np.ndarray (H,W,3)
objs   = pickle.load(open(OBJS_PKL, "rb"))        # list of list[Object]

assert len(frames) == len(objs)

# 1) 收集类别集合（✅过滤 HUD）
classes = set()
for obj_list in objs:
    for o in obj_list:
        if getattr(o, "hud", False):  # ✅ 新增：跳过 HUD 类别
            continue
        classes.add(o.__class__.__name__)   # 或 o.category，如果有
classes = sorted(list(classes))
name2id = {n:i for i,n in enumerate(classes)}

print("Classes:", classes)

# 2) 划分 train/val
idxs = list(range(len(frames)))
random.shuffle(idxs)
split = int(0.9 * len(idxs))
train_ids, val_ids = idxs[:split], idxs[split:]

def save_one(i, split_name):
    img = frames[i]
    H, W = img.shape[:2]

    # 保存图片（YOLO 常用 jpg/png 均可）
    img_path = os.path.join(IMG_DIR, split_name, f"{i:06d}.png")
    # 注意：cv2 写入是 BGR
    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # 生成 label txt
    label_path = os.path.join(LBL_DIR, split_name, f"{i:06d}.txt")
    lines = []
    for o in objs[i]:
        if getattr(o, "hud", False):  # ✅ 新增：跳过 HUD 标注
            continue

        cls_name = o.__class__.__name__
        cid = name2id[cls_name]

        x, y, w, h = o.xywh  # xywh in pixel
        # xywh -> x_center,y_center
        xc = x + w / 2
        yc = y + h / 2

        # 归一化
        xc /= W
        yc /= H
        w  /= W
        h  /= H

        # 过滤掉异常框（避免训练崩）
        if w <= 0 or h <= 0:
            continue
        if xc < 0 or xc > 1 or yc < 0 or yc > 1:
            continue

        lines.append(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    with open(label_path, "w") as f:
        f.write("\n".join(lines))

for i in tqdm(train_ids):
    save_one(i, "train")
for i in tqdm(val_ids):
    save_one(i, "val")

# 3) 写 data.yaml
yaml_path = os.path.join(OUT_DIR, "data.yaml")
with open(yaml_path, "w") as f:
    f.write(f"path: {os.path.abspath(OUT_DIR)}\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write(f"nc: {len(classes)}\n")
    f.write(f"names: {classes}\n")

print("Done. YOLO dataset at:", OUT_DIR)

''' 
# train:
yolo detect train \
  data=yolo_dataset/data.yaml \
  model=yolov8n.pt \
  imgsz=160 \
  epochs=100 \
  batch=64 \
  workers=4 \
  device=0

  # validate
yolo detect val \
  model=runs/detect/train/weights/best.pt \
  data=yolo_dataset/data.yaml \
  imgsz=160

  # visualize results
yolo detect predict \
  model=runs/detect/train/weights/best.pt \
  source=yolo_dataset/images/val \
  imgsz=160 \
  conf=0.25 \
  save=True

'''
