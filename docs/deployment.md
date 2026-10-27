# Edge AI Deployment & Optimization Guide

This guide details how to export, optimize, and deploy the Unattended Object Detection pipeline on NVIDIA Jetson, edge appliances, or workstation GPUs using ONNX and TensorRT.

---

## 1. Optimization Workflow

```
[PyTorch Model (.pt)]
         │
         ▼  (scripts/export_onnx.py)
  [ONNX Graph (.onnx)]
         │
         ▼  (scripts/build_tensorrt.py)
[TensorRT Engine (.engine)]
         │
         ▼
[Edge Inference & Streamlit App]
```

---

## 2. Step 1: Exporting to ONNX

Convert PyTorch weights (`.pt`) to an optimized ONNX graph:
```bash
python scripts/export_onnx.py --model models/weights/best.pt --output_dir models/onnx --imgsz 640 --opset 12
```

**Key Parameters:**
- `--imgsz 640`: Standard surveillance input resolution.
- `--opset 12`: Maximum compatibility with TensorRT and ONNX Runtime.
- `--dynamic`: Optional dynamic batch size for multi-camera streams.

---

## 3. Step 2: Compiling to TensorRT

### Prerequisites
TensorRT requires:
- NVIDIA GPU (RTX series or Jetson Orin/Xavier)
- CUDA Toolkit (11.8+ or 12.x) and cuDNN
- Python package `tensorrt` or `trtexec` in PATH.

### Installation on Ubuntu / Jetson:
```bash
sudo apt-get install -y tensorrt
pip install --upgrade tensorrt tensorrt-cu12
```

### Build Engine Command:
```bash
python scripts/build_tensorrt.py --onnx models/onnx/yolov8n.onnx --output models/tensorrt/yolov8n.engine --fp16
```

This applies kernel auto-tuning, layer fusion, and FP16 half-precision math to dramatically increase frame rates and lower latency.

---

## 4. Step 3: Benchmarking Latency & Power

Benchmark the runtime backends on your hardware:
```bash
python scripts/benchmark.py --pt yolov8n.pt --onnx models/onnx/yolov8n.onnx
```

### Metrics Reported:
- **Throughput (FPS)**
- **Latency (ms):** Mean, Median, P95, and P99 percentiles
- **Memory Footprint:** Host RAM and GPU VRAM consumption
- **Power Draw:** Real-time milliwatt measurement via NVIDIA NVML (`pynvml`), reporting Average Watts, Peak Watts, and Energy Efficiency (FPS/Watt).
