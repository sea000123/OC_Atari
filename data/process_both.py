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

# ============== 1) 固定使用 UNION 类别顺序（你给的） ==============
UNION_CLASSES = [
    'Barrier', 'Beam', 'Key', 'Key_HUD', 'Ladder', 'Life', 'NoObject',
    'Platform', 'Player', 'Rope', 'Score', 'Skull', 'Sword', 'Sword_HUD', 'Wall'
]
name2id = {n: i for i, n in enumerate(UNION_CLASSES)}

# 是否导出 HUD 类别（Key_HUD, Sword_HUD 等）
INCLUDE_HUD = False  # <- 你要包含 HUD 就改成 True

# 是否导出 NoObject（通常 YOLO 不需要，把它当“背景”更合理）
INCLUDE_NOOBJECT = False

print("Using UNION_CLASSES:", UNION_CLASSES)
print("name2id:", name2id)
print("INCLUDE_HUD:", INCLUDE_HUD)
print("INCLUDE_NOOBJECT:", INCLUDE_NOOBJECT)

# ============== 2) 创建目录 ==============
for split in ["train", "val"]:
    os.makedirs(os.path.join(IMG_DIR, split), exist_ok=True)
    os.makedirs(os.path.join(LBL_DIR, split), exist_ok=True)

# ============== 3) 读取数据 ==============
frames = pickle.load(open(FRAMES_PKL, "rb"))  # list[np.ndarray(H,W,3) RGB]
objs   = pickle.load(open(OBJS_PKL, "rb"))    # list[list[Object]]
assert len(frames) == len(objs), f"len(frames)={len(frames)} != len(objs)={len(objs)}"

# ============== 4) 划分 train/val ==============
idxs = list(range(len(frames)))
random.shuffle(idxs)
split = int(0.9 * len(idxs))
train_ids, val_ids = idxs[:split], idxs[split:]

def should_skip_class(cls_name: str) -> bool:
    # 过滤 HUD：优先按类名后缀判断（因为你 union 里 HUD 是独立类名）
    if not INCLUDE_HUD and cls_name.endswith("_HUD"):
        return True
    # 过滤 NoObject：一般不当作检测目标
    if not INCLUDE_NOOBJECT and cls_name == "NoObject":
        return True
    return False

def save_one(i, split_name: str):
    img = frames[i]
    H, W = img.shape[:2]

    # 保存图片（cv2 保存 BGR）
    img_path = os.path.join(IMG_DIR, split_name, f"{i:06d}.png")
    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    label_path = os.path.join(LBL_DIR, split_name, f"{i:06d}.txt")
    lines = []

    for o in objs[i]:
        cls_name = o.__class__.__name__

        # 只使用 UNION 里的类别；不在则跳过
        if cls_name not in name2id:
            continue

        if should_skip_class(cls_name):
            continue

        cid = name2id[cls_name]

        # xywh in pixel
        x, y, w, h = o.xywh

        # xywh -> center
        xc = x + w / 2.0
        yc = y + h / 2.0

        # normalize
        xc /= W
        yc /= H
        w  /= W
        h  /= H

        # 过滤异常框
        if w <= 0 or h <= 0:
            continue
        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
            continue
        if w > 1.0 or h > 1.0:
            continue

        lines.append(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    with open(label_path, "w") as f:
        f.write("\n".join(lines))

# 导出
for i in tqdm(train_ids, desc="Export train"):
    save_one(i, "train")
for i in tqdm(val_ids, desc="Export val"):
    save_one(i, "val")

# ============== 5) 写 data.yaml ==============
yaml_path = os.path.join(OUT_DIR, "data.yaml")
with open(yaml_path, "w") as f:
    f.write(f"path: {os.path.abspath(OUT_DIR)}\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write(f"nc: {len(UNION_CLASSES)}\n")
    f.write(f"names: {UNION_CLASSES}\n")

print("Done. YOLO dataset at:", OUT_DIR)
