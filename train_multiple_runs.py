import torch
from ultralytics import YOLO
import numpy as np
import os
from pathlib import Path

def train_and_evaluate_multiple_runs():
    seeds = [42, 43, 44]
    all_results = []
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Training with seed: {seed}")
        print(f"{'='*60}")
 
        model = YOLO('yolov8s.pt')  

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        training_params = {
            'data': 'maritime.yaml',
            'epochs': 100,
            'batch': 4,
            'imgsz': 960,
            'optimizer': 'SGD',
            'lr0': 0.01,
            'momentum': 0.9,
            'patience': 10,
            'project': 'runs/detect',
            'name': f'yolov8s_maritime_seed_{seed}',
            'exist_ok': True,
            'amp': True,
            'workers': 0,
            'pin_memory': False,
            'persistent_workers': False,
            'seed': seed,
            'cos_lr': True,
            'mosaic': 0.6,
            'mixup': 0.05,
            'close_mosaic': 10,
            'accumulate': 2,
        }

        print(f"Training parameters for seed {seed}:")
        for key, value in training_params.items():
            print(f"  {key}: {value}")

        results = model.train(**training_params)

        print(f"Validation for seed {seed}...")
        metrics = model.val(
            conf=0.25,  
            iou=0.6,    
            device=device,
            split='test'
        )

        run_result = {
            'seed': seed,
            'map50': float(metrics.box.map50) if metrics.box.map50 is not None else 0.0,
            'map50_95': float(metrics.box.map) if metrics.box.map is not None else 0.0,
            'precision': float(metrics.box.p) if hasattr(metrics.box, 'p') and metrics.box.p is not None else 0.0,
            'recall': float(metrics.box.r) if hasattr(metrics.box, 'r') and metrics.box.r is not None else 0.0,
            'aps': float(metrics.box.maps[0]) if hasattr(metrics.box, 'maps') and len(metrics.box.maps) > 0 else 0.0,  # APS for small objects
        }
        
        print(f"Seed {seed} results:")
        print(f"  mAP@0.5: {run_result['map50']:.4f}")
        print(f"  mAP@[0.5:0.95]: {run_result['map50_95']:.4f}")
        print(f"  Precision: {run_result['precision']:.4f}")
        print(f"  Recall: {run_result['recall']:.4f}")
        print(f"  APS: {run_result['aps']:.4f}")
        
        all_results.append(run_result)

        model_save_path = f'runs/detect/yolov8s_maritime_seed_{seed}/weights/best.pt'
        print(f"Model saved to: {model_save_path}")

    if all_results:
        final_results = {
            'map50_mean': np.mean([r['map50'] for r in all_results]),
            'map50_std': np.std([r['map50'] for r in all_results]),
            'map50_95_mean': np.mean([r['map50_95'] for r in all_results]),
            'map50_95_std': np.std([r['map50_95'] for r in all_results]),
            'precision_mean': np.mean([r['precision'] for r in all_results]),
            'precision_std': np.std([r['precision'] for r in all_results]),
            'recall_mean': np.mean([r['recall'] for r in all_results]),
            'recall_std': np.std([r['recall'] for r in all_results]),
            'aps_mean': np.mean([r['aps'] for r in all_results]),
            'aps_std': np.std([r['aps'] for r in all_results]),
            'num_runs': len(all_results),
            'seeds': seeds
        }
        
        print(f"\n{'='*60}")
        print("FINAL RESULTS (Mean ± Std over 3 runs)")
        print(f"{'='*60}")
        print(f"mAP@0.5: {final_results['map50_mean']:.4f} ± {final_results['map50_std']:.4f}")
        print(f"mAP@[0.5:0.95]: {final_results['map50_95_mean']:.4f} ± {final_results['map50_95_std']:.4f}")
        print(f"Precision: {final_results['precision_mean']:.4f} ± {final_results['precision_std']:.4f}")
        print(f"Recall: {final_results['recall_mean']:.4f} ± {final_results['recall_std']:.4f}")
        print(f"APS: {final_results['aps_mean']:.4f} ± {final_results['aps_std']:.4f}")
        
        return final_results, all_results

    return None, all_results

if __name__ == "__main__":
    final_results, all_results = train_and_evaluate_multiple_runs()