import torch
import time
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import os
import cv2

def calculate_latency_metrics(model_paths, test_images_dir, device="cuda:0", device_name="Workstation", num_iterations=1000):
    all_latency_results = []
    model_results = []
    
    print(f"Latency Testing for {device_name}")
    print(f"Testing {len(model_paths)} models with {num_iterations} iterations")
    print("-" * 60)
    
    for model_path in model_paths:
        if not os.path.exists(model_path):
            print(f"  Model not found: {model_path}")
            continue
            
        try:
            model = YOLO(model_path)
            model_name = Path(model_path).parent.parent.name
  
            test_images = list(Path(test_images_dir).glob("*.jpg")) + list(Path(test_images_dir).glob("*.png"))
            
            if len(test_images) == 0:
                print(f"  No test images found in {test_images_dir}")
                continue
 
            test_image_path = test_images[0]
            img = cv2.imread(str(test_image_path))
            
            if img is None:
                print(f"  Could not read image: {test_image_path}")
                continue

            img = cv2.resize(img, (960, 960))

            print(f"  Warming up {model_name}...")
            warmup_iterations = max(10, num_iterations // 10)
            for _ in range(warmup_iterations):
                _ = model(img, device=device, verbose=False, imgsz=960, conf=0.25, iou=0.6)

            print(f"  Measuring latency for {model_name}...")
            latency_times = []
            
            for i in range(num_iterations):
                start_time = time.perf_counter()
                results = model(img, device=device, verbose=False, imgsz=960, conf=0.25, iou=0.6)
                end_time = time.perf_counter()
                
                latency_times.append((end_time - start_time) * 1000)  
            
            if len(latency_times) > 0:
                latency_array = np.array(latency_times)
                avg_latency = np.mean(latency_array)
                min_latency = np.min(latency_array)
                max_latency = np.max(latency_array)
                std_latency = np.std(latency_array)
            
                p50_latency = np.percentile(latency_array, 50)   
                p95_latency = np.percentile(latency_array, 95)   
                p99_latency = np.percentile(latency_array, 99)   

                fps = 1000.0 / avg_latency 
                p50_fps = 1000.0 / p50_latency
                p95_fps = 1000.0 / p95_latency
                
                model_result = {
                    'model_name': model_name,
                    'device': device_name,
                    'fps_mean': fps,
                    'fps_p50': p50_fps,
                    'fps_p95': p95_fps,
                    'latency_mean_ms': avg_latency,
                    'latency_p50_ms': p50_latency,
                    'latency_p95_ms': p95_latency,
                    'latency_p99_ms': p99_latency,
                    'latency_std_ms': std_latency,
                    'latency_min_ms': min_latency,
                    'latency_max_ms': max_latency,
                    'iterations': num_iterations
                }
                
                model_results.append(model_result)
                all_latency_results.append(model_result)
                
                print(f"    {model_name}:")
                print(f"      FPS: {fps:.2f} (P50: {p50_fps:.2f}, P95: {p95_fps:.2f})")
                print(f"      Latency: {avg_latency:.2f}ms (P50: {p50_latency:.2f}ms, P95: {p95_latency:.2f}ms)")
                
        except Exception as e:
            print(f"    Error testing {model_path}: {e}")
            continue

    if all_latency_results:
        fps_values = [r['fps_mean'] for r in all_latency_results]
        p50_latencies = [r['latency_p50_ms'] for r in all_latency_results]
        p95_latencies = [r['latency_p95_ms'] for r in all_latency_results]
        
        stats = {
            'fps_mean': np.mean(fps_values),
            'fps_std': np.std(fps_values),
            'fps_p50_mean': np.mean([r['fps_p50'] for r in all_latency_results]),
            'fps_p95_mean': np.mean([r['fps_p95'] for r in all_latency_results]),
            'latency_p50_mean': np.mean(p50_latencies),
            'latency_p50_std': np.std(p50_latencies),
            'latency_p95_mean': np.mean(p95_latencies),
            'latency_p95_std': np.std(p95_latencies),
            'num_models': len(model_paths),
            'num_successful': len(all_latency_results),
            'device': device_name,
            'iterations': num_iterations
        }
        
        print(f"\n{device_name} latency testing completed:")
        print(f"  Successful models: {stats['num_successful']}/{stats['num_models']}")
        print(f"  FPS: {stats['fps_mean']:.2f} ± {stats['fps_std']:.2f}")
        print(f"  P50 Latency: {stats['latency_p50_mean']:.2f} ± {stats['latency_p50_std']:.2f} ms")
        print(f"  P95 Latency: {stats['latency_p95_mean']:.2f} ± {stats['latency_p95_std']:.2f} ms")
        
        return stats, model_results
    
    return None, []

def benchmark_comprehensive_performance():
    model_paths = [
        "runs/detect/yolov8s_maritime_seed_42/weights/best.pt",
        "runs/detect/yolov8s_maritime_seed_43/weights/best.pt",
        "runs/detect/yolov8s_maritime_seed_44/weights/best.pt"
    ]
    
    test_images_dir = "G:/ultralytics-main/datasets/FloW_IMG_splits/test/images"

    existing_models = [mp for mp in model_paths if os.path.exists(mp)]
    
    if not existing_models:
        print("No trained models found for performance benchmarking")
        return None, None
    
    print("• Measuring P50 and P95 latency percentiles")
    print("• 3 seed models tested (seeds 42, 43, 44)")
    print("=" * 80)

    print("\n" + "=" * 50)
    print("WORKSTATION (GPU) PERFORMANCE")
    print("=" * 50)
    
    workstation_stats, workstation_details = calculate_latency_metrics(
        existing_models, test_images_dir,
        device='cuda:0' if torch.cuda.is_available() else 'cpu',
        device_name='Workstation_GPU',
        num_iterations=1000
    )
   
    print("\n" + "=" * 50)
    print("EMBEDDED DEVICE (CPU) PERFORMANCE")
    print("=" * 50)
    
    embedded_stats, embedded_details = calculate_latency_metrics(
        existing_models, test_images_dir,
        device='cpu',
        device_name='Embedded_CPU',
        num_iterations=1000
    )

    print("\n" + "=" * 80)
    print("FINAL PERFORMANCE RESULTS")
    print("=" * 80)
    
    if workstation_stats:
        print(f"\nWorkstation GPU Performance:")
        print(f"  FPS: {workstation_stats['fps_mean']:.2f} ± {workstation_stats['fps_std']:.2f}")
        print(f"  P50 FPS: {workstation_stats['fps_p50_mean']:.2f}")
        print(f"  P95 FPS: {workstation_stats['fps_p95_mean']:.2f}")
        print(f"  P50 Latency: {workstation_stats['latency_p50_mean']:.2f} ± {workstation_stats['latency_p50_std']:.2f} ms")
        print(f"  P95 Latency: {workstation_stats['latency_p95_mean']:.2f} ± {workstation_stats['latency_p95_std']:.2f} ms")
    
    if embedded_stats:
        print(f"\nEmbedded CPU Performance:")
        print(f"  FPS: {embedded_stats['fps_mean']:.2f} ± {embedded_stats['fps_std']:.2f}")
        print(f"  P50 FPS: {embedded_stats['fps_p50_mean']:.2f}")
        print(f"  P95 FPS: {embedded_stats['fps_p95_mean']:.2f}")
        print(f"  P50 Latency: {embedded_stats['latency_p50_mean']:.2f} ± {embedded_stats['latency_p50_std']:.2f} ms")
        print(f"  P95 Latency: {embedded_stats['latency_p95_mean']:.2f} ± {embedded_stats['latency_p95_std']:.2f} ms")

    if workstation_stats and embedded_stats:
        fps_ratio = embedded_stats['fps_mean'] / workstation_stats['fps_mean']
        latency_ratio = embedded_stats['latency_p50_mean'] / workstation_stats['latency_p50_mean']
        
        print(f"\nPerformance Comparison:")
        print(f"  Embedded FPS is {fps_ratio:.1%} of Workstation FPS")
        print(f"  Embedded latency is {latency_ratio:.1f}x Workstation latency")
        print(f"  Workstation is {1/fps_ratio:.1f}x faster than Embedded")
    
    print("\n" + "=" * 80)
    
    return workstation_stats, embedded_stats

def get_comprehensive_performance_metrics():
    print("Starting comprehensive performance benchmarking...")
    workstation_stats, embedded_stats = benchmark_comprehensive_performance()
    
    performance_results = {}
    
    if workstation_stats:
        performance_results['workstation'] = {
            'fps_mean': workstation_stats['fps_mean'],
            'fps_std': workstation_stats['fps_std'],
            'latency_p50_mean': workstation_stats['latency_p50_mean'],
            'latency_p50_std': workstation_stats['latency_p50_std'],
            'latency_p95_mean': workstation_stats['latency_p95_mean'],
            'latency_p95_std': workstation_stats['latency_p95_std']
        }
        print(f"\nFINAL WORKSTATION METRICS:")
        print(f"  FPS: {workstation_stats['fps_mean']:.2f} ± {workstation_stats['fps_std']:.2f}")
        print(f"  P50 Latency: {workstation_stats['latency_p50_mean']:.2f} ± {workstation_stats['latency_p50_std']:.2f} ms")
        print(f"  P95 Latency: {workstation_stats['latency_p95_mean']:.2f} ± {workstation_stats['latency_p95_std']:.2f} ms")
    
    if embedded_stats:
        performance_results['embedded'] = {
            'fps_mean': embedded_stats['fps_mean'],
            'fps_std': embedded_stats['fps_std'],
            'latency_p50_mean': embedded_stats['latency_p50_mean'],
            'latency_p50_std': embedded_stats['latency_p50_std'],
            'latency_p95_mean': embedded_stats['latency_p95_mean'],
            'latency_p95_std': embedded_stats['latency_p95_std']
        }
        print(f"\nFINAL EMBEDDED METRICS:")
        print(f"  FPS: {embedded_stats['fps_mean']:.2f} ± {embedded_stats['fps_std']:.2f}")
        print(f"  P50 Latency: {embedded_stats['latency_p50_mean']:.2f} ± {embedded_stats['latency_p50_std']:.2f} ms")
        print(f"  P95 Latency: {embedded_stats['latency_p95_mean']:.2f} ± {embedded_stats['latency_p95_std']:.2f} ms")
    
    return performance_results

if __name__ == "__main__":

    performance_results = get_comprehensive_performance_metrics()

    if performance_results:
        print("\nPERFORMANCE RESULTS SUMMARY:")
        for device, metrics in performance_results.items():
            print(f"  {device.capitalize()}:")
            print(f"    FPS: {metrics['fps_mean']:.2f} ± {metrics['fps_std']:.2f}")
            print(f"    P50 Latency: {metrics['latency_p50_mean']:.2f} ± {metrics['latency_p50_std']:.2f} ms")
            print(f"    P95 Latency: {metrics['latency_p95_mean']:.2f} ± {metrics['latency_p95_std']:.2f} ms")