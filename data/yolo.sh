conda activate revenge2
python data/generate_dataset.py \
  -g ALE/MontezumaRevenge-v5 \
  -m ram \
  -hud\
  -dqn
cd data 
python po1.py
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