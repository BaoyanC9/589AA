import torch
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path

def calculate_aps_metrics(model_paths, data_yaml, test_images_dir):
    all_aps_results = []
    
    for i, model_path in enumerate(model_paths):
        print(f"\nCalculating metrics for model {i+1}/{len(model_paths)}")
        
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            continue
   
        model = YOLO(model_path)

        metrics = model.val(
            data=data_yaml, 
            split='test', 
            verbose=True,
            conf=0.25,      
            iou=0.6,        
            device='cuda:0' if torch.cuda.is_available() else 'cpu',
            plots=True     
        )

        map50 = float(metrics.box.map50) if metrics.box.map50 is not None else 0.0
        map50_95 = float(metrics.box.map) if metrics.box.map is not None else 0.0
 
        precision = float(metrics.box.p) if hasattr(metrics.box, 'p') and metrics.box.p is not None else 0.0
        recall = float(metrics.box.r) if hasattr(metrics.box, 'r') and metrics.box.r is not None else 0.0
 
        if hasattr(metrics.box, 'maps'):
            maps = metrics.box.maps
            aps = float(maps[0]) if len(maps) > 0 else 0.0  
            apm = float(maps[1]) if len(maps) > 1 else 0.0    
            apl = float(maps[2]) if len(maps) > 2 else 0.0  
        else:
            aps = apm = apl = 0.0
        results = model.predict(
            source=test_images_dir, 
            conf=0.25,
            iou=0.6,
            save=False, 
            show=False,
            verbose=False,
            imgsz=960
        )
       
        SMALL_THRESHOLD = 32 * 32  
        small_detections = 0
        total_detections = 0
        small_confidences = []
        
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                if len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    conf = boxes.conf.cpu().numpy()
                    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                    small_indices = areas < SMALL_THRESHOLD
                    
                    small_detections += np.sum(small_indices)
                    total_detections += len(areas)
                    
                    if np.any(small_indices):
                        small_confidences.extend(conf[small_indices])

        small_ratio = small_detections / total_detections if total_detections > 0 else 0
        avg_small_confidence = np.mean(small_confidences) if small_confidences else 0
        
        run_result = {
            'run_id': i+1,
            'model_path': model_path,
            'map50': map50,
            'map50_95': map50_95,
            'precision': precision,
            'recall': recall,
            'aps': aps,
            'apm': apm,
            'apl': apl,
            'small_detections': small_detections,
            'total_detections': total_detections,
            'small_ratio': small_ratio,
            'avg_small_confidence': avg_small_confidence,
        }
        
        all_aps_results.append(run_result)
        
        print(f"  Run {i+1} comprehensive results:")
        print(f"    mAP@0.5: {map50:.4f}")
        print(f"    mAP@[0.5:0.95]: {map50_95:.4f}")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall: {recall:.4f}")
        print(f"    APS: {aps:.4f}")
        print(f"    APM: {apm:.4f}")
        print(f"    APL: {apl:.4f}")
        print(f"    Small objects detected: {small_detections}/{total_detections} ({small_ratio:.2%})")

    if all_aps_results:
        stats = {
            'map50_mean': np.mean([r['map50'] for r in all_aps_results]),
            'map50_std': np.std([r['map50'] for r in all_aps_results]),
            'map50_95_mean': np.mean([r['map50_95'] for r in all_aps_results]),
            'map50_95_std': np.std([r['map50_95'] for r in all_aps_results]),
            'precision_mean': np.mean([r['precision'] for r in all_aps_results]),
            'precision_std': np.std([r['precision'] for r in all_aps_results]),
            'recall_mean': np.mean([r['recall'] for r in all_aps_results]),
            'recall_std': np.std([r['recall'] for r in all_aps_results]),
            'aps_mean': np.mean([r['aps'] for r in all_aps_results]),
            'aps_std': np.std([r['aps'] for r in all_aps_results]),
            'apm_mean': np.mean([r['apm'] for r in all_aps_results]),
            'apm_std': np.std([r['apm'] for r in all_aps_results]),
            'apl_mean': np.mean([r['apl'] for r in all_aps_results]),
            'apl_std': np.std([r['apl'] for r in all_aps_results]),
            'num_runs': len(all_aps_results)
        }
        
        return stats, all_aps_results
    
    return None, []

if __name__ == "__main__":
    model_paths = [
        "runs/detect/yolov8s_maritime_seed_42/weights/best.pt",
        "runs/detect/yolov8s_maritime_seed_43/weights/best.pt", 
        "runs/detect/yolov8s_maritime_seed_44/weights/best.pt"
    ]

    data_yaml = "G:/ultralytics-main/datasets/FloW_IMG_splits/FloW_IMG_splits.yaml"
    test_images_dir = "G:/ultralytics-main/datasets/FloW_IMG_splits/test/images"

    existing_models = [mp for mp in model_paths if os.path.exists(mp)]
    
    if not existing_models:
        print("No trained models found. Please run training first.")
    else:
        print(f"Found {len(existing_models)} models for comprehensive evaluation")
        stats, all_results = calculate_aps_metrics(existing_models, data_yaml, test_images_dir)
        
        if stats:
            print(f"\n{'='*80}")
            print("COMPREHENSIVE EVALUATION RESULTS (Mean ± Std over multiple runs)")
            print(f"{'='*80}")
            print(f"mAP@0.5:              {stats['map50_mean']:.4f} ± {stats['map50_std']:.4f}")
            print(f"mAP@[0.5:0.95]:      {stats['map50_95_mean']:.4f} ± {stats['map50_95_std']:.4f}")
            print(f"Precision:            {stats['precision_mean']:.4f} ± {stats['precision_std']:.4f}")
            print(f"Recall:               {stats['recall_mean']:.4f} ± {stats['recall_std']:.4f}")
            print(f"APS (Small Objects):  {stats['aps_mean']:.4f} ± {stats['aps_std']:.4f}")
            print(f"APM (Medium Objects): {stats['apm_mean']:.4f} ± {stats['apm_std']:.4f}")
            print(f"APL (Large Objects):  {stats['apl_mean']:.4f} ± {stats['apl_std']:.4f}")
            print(f"Number of runs:       {stats['num_runs']}")