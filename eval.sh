#!/bin/bash

echo "WA-YOLO Comprehensive Evaluation"
echo "================================"

echo "1. Validating trained models..."
for seed in 42 43 44; do
    model_path="runs/detect/WA-YOLO_seed_${seed}/weights/best.pt"
    if [ -f "$model_path" ]; then
        echo "Validating seed $seed..."
        yolo detect val data=maritime.yaml model=$model_path imgsz=960 conf=0.25 iou=0.6 split=test
    fi
done

echo "2. Calculating APS metrics..."
python evaluate_aps.py

echo "3. Benchmarking inference speed..."
python benchmark_fps.py

echo "Evaluation completed."