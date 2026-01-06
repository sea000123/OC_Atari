conda activate revenge2
python data/generate_dataset.py
cd data 
# python data/generate_dataset_old.py 
# python po1.py # use vision result, it can produce best result as "train1" model
python process_both.py # use ram data

yolo detect train \
  data=yolo_dataset/data.yaml \
  model=yolov8s.pt \
  imgsz=160 \
  epochs=80 \
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