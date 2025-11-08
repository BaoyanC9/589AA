#!/bin/bash

echo "WA-YOLO Training - Three Seed Experiment"
echo "=========================================="

echo "Starting training with seed 42..."
yolo detect train data=maritime.yaml model=yolov8.pt imgsz=960 epochs=100 batch=4 accumulate=2 amp=True mosaic=0.6 mixup=0.05 close_mosaic=10 workers=0 pin_memory=False persistent_workers=False seed=42 optimizer=SGD lr0=0.01 cos_lr=True patience=10 project=runs/detect name=WA-YOLO_seed_42 exist_ok=True

echo "Starting training with seed 43..."  
yolo detect train data=maritime.yaml model=yolov8.pt imgsz=960 epochs=100 batch=4 accumulate=2 amp=True mosaic=0.6 mixup=0.05 close_mosaic=10 workers=0 pin_memory=False persistent_workers=False seed=43 optimizer=SGD lr0=0.01 cos_lr=True patience=10 project=runs/detect name=WA-YOLO_seed_43 exist_ok=True

echo "Starting training with seed 44..."
yolo detect train data=maritime.yaml model=yolov8.pt imgsz=960 epochs=100 batch=4 accumulate=2 amp=True mosaic=0.6 mixup=0.05 close_mosaic=10 workers=0 pin_memory=False persistent_workers=False seed=44 optimizer=SGD lr0=0.01 cos_lr=True patience=10 project=runs/detect name=WA-YOLO_seed_44 exist_ok=True

echo "Training completed. Models saved in runs/detect/"