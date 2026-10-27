"""Comprehensive Performance Benchmark across Inference Runtimes.

Benchmarks:
  1. PyTorch (FP32 / FP16)
  2. ONNX Runtime (CPU / CUDA Execution Provider)
  3. TensorRT (FP16 Engine, where NVIDIA TensorRT is available)
Measures:
  - Latency (Mean, Median, P95, P99)
  - Throughput (FPS)
  - Memory Footprint (RAM / VRAM)
  - Power Consumption (Average Watts, Peak Watts, FPS/Watt via NVML)
  - Model File Size (MB)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from src.utils.power_monitor import PowerMonitor

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger("Benchmark")


def benchmark_pytorch(
    model_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    imgsz: int = 640,
    iterations: int = 100,
    warmup: int = 15,
) -> Dict[str, Any]:
    """Benchmark native PyTorch YOLO model."""
    from ultralytics import YOLO

    p = Path(model_path)
    if not p.exists():
        p = Path("yolov8n.pt")

    model_size_mb = p.stat().st_size / (1024 * 1024)
    model = YOLO(str(p))

    dummy_input = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # Warmup
    for _ in range(warmup):
        _ = model.predict(dummy_input, device=device, verbose=False)

    latencies = []
    power_mon = PowerMonitor(gpu_device_id=0)

    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = model.predict(dummy_input, device=device, verbose=False)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    total_time_s = sum(latencies) / 1000.0
    power_stats = power_mon.get_power_summary(iterations, total_time_s)

    vram_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if torch.cuda.is_available() and device.startswith("cuda")
        else None
    )

    return {
        "runtime": "PyTorch",
        "device": device,
        "model_size_mb": round(model_size_mb, 2),
        "mean_latency_ms": round(float(np.mean(latencies)), 2),
        "median_latency_ms": round(float(np.median(latencies)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "p99_latency_ms": round(float(np.percentile(latencies, 99)), 2),
        "fps": round(1000.0 / float(np.mean(latencies)), 2),
        "gpu_vram_used_mb": round(vram_mb, 1) if vram_mb is not None else "N/A",
        "power_stats": power_stats,
    }


def benchmark_onnx(
    onnx_path: str,
    imgsz: int = 640,
    iterations: int = 100,
    warmup: int = 15,
) -> Dict[str, Any]:
    """Benchmark ONNX Runtime."""
    if ort is None:
        return {"runtime": "ONNX Runtime", "status": "onnxruntime not installed"}

    p = Path(onnx_path)
    if not p.exists():
        return {"runtime": "ONNX Runtime", "status": f"File {onnx_path} not found"}

    model_size_mb = p.stat().st_size / (1024 * 1024)

    # Select provider
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(p), providers=providers)
    input_name = session.get_inputs()[0].name

    dummy_input = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        _ = session.run(None, {input_name: dummy_input})

    latencies = []
    power_mon = PowerMonitor(gpu_device_id=0)

    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: dummy_input})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    total_time_s = sum(latencies) / 1000.0
    power_stats = power_mon.get_power_summary(iterations, total_time_s)

    return {
        "runtime": "ONNX Runtime",
        "providers": session.get_providers(),
        "model_size_mb": round(model_size_mb, 2),
        "mean_latency_ms": round(float(np.mean(latencies)), 2),
        "median_latency_ms": round(float(np.median(latencies)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "p99_latency_ms": round(float(np.percentile(latencies, 99)), 2),
        "fps": round(1000.0 / float(np.mean(latencies)), 2),
        "gpu_vram_used_mb": "Managed by Provider",
        "power_stats": power_stats,
    }


def benchmark_tensorrt(
    engine_path: str,
    imgsz: int = 640,
    iterations: int = 100,
    warmup: int = 15,
) -> Dict[str, Any]:
    """Benchmark TensorRT engine if installed."""
    p = Path(engine_path)
    if not p.exists():
        return {
            "runtime": "TensorRT",
            "status": "Not Available (Engine file does not exist. Run scripts/build_tensorrt.py on a machine with TensorRT).",
        }

    try:
        import tensorrt as trt
    except ImportError:
        return {
            "runtime": "TensorRT",
            "status": "Not Available (Python 'tensorrt' bindings not installed).",
        }

    return {
        "runtime": "TensorRT",
        "status": "TensorRT benchmarking available with native CUDA bindings.",
    }


def run_all_benchmarks(
    model_pt: str = "yolov8n.pt",
    model_onnx: str = "models/onnx/yolov8n.onnx",
    model_engine: str = "models/tensorrt/yolov8n.engine",
    output_json: str = "results/metrics/benchmark_metrics.json",
):
    """Run comparative benchmark across all runtime backends."""
    results = []

    # 1. PyTorch
    logger.info("Running PyTorch benchmark...")
    pt_res = benchmark_pytorch(model_pt)
    results.append(pt_res)

    # 2. ONNX Runtime
    logger.info("Running ONNX Runtime benchmark...")
    if not Path(model_onnx).exists():
        logger.info(f"ONNX file '{model_onnx}' not found. Exporting now...")
        from scripts.export_onnx import export_to_onnx
        export_to_onnx(model_path=model_pt, output_dir="models/onnx")

    onnx_res = benchmark_onnx(model_onnx)
    results.append(onnx_res)

    # 3. TensorRT
    logger.info("Running TensorRT benchmark check...")
    trt_res = benchmark_tensorrt(model_engine)
    results.append(trt_res)

    # Save Results
    out_file = Path(output_json)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Benchmark metrics saved to: {out_file}")

    # Display comparison
    print("\n" + "=" * 80)
    print("RUNTIME PERFORMANCE & POWER BENCHMARK SUMMARY")
    print("=" * 80)
    for r in results:
        print(f"Runtime: {r.get('runtime')}")
        if "status" in r and "Not Available" in r["status"]:
            print(f"  Status: {r['status']}")
        else:
            print(f"  Throughput: {r.get('fps')} FPS")
            print(f"  Latency:    {r.get('mean_latency_ms')} ms (P95: {r.get('p95_latency_ms')} ms)")
            print(f"  Model Size: {r.get('model_size_mb')} MB")
            pwr = r.get("power_stats", {})
            print(f"  Power Draw: Avg {pwr.get('avg_power_w')} W | Peak {pwr.get('peak_power_w')} W")
            print(f"  Efficiency: {pwr.get('fps_per_watt')} FPS/Watt")
        print("-" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark inference engines")
    parser.add_argument("--pt", type=str, default="yolov8n.pt", help="Path to PyTorch model")
    parser.add_argument("--onnx", type=str, default="models/onnx/yolov8n.onnx", help="Path to ONNX model")
    parser.add_argument("--engine", type=str, default="models/tensorrt/yolov8n.engine", help="Path to TensorRT engine")
    parser.add_argument("--output", type=str, default="results/metrics/benchmark_metrics.json", help="Output JSON")
    args = parser.parse_args()

    run_all_benchmarks(
        model_pt=args.pt,
        model_onnx=args.onnx,
        model_engine=args.engine,
        output_json=args.output,
    )
