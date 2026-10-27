# Missing Components & Phased Engineering Action Plan

**Project Title:** Unattended / Abandoned Object Detection with Temporal Reasoning for Edge AI Surveillance  
**Repository Path:** `C:\Users\Aditya\.gemini\antigravity\scratch\abandoned_object_detection`  

---

## 1. Unvarnished Gap Analysis: Implemented vs. Incomplete Components

| Expected System Capability | Status in Current Repository | File Reference | Technical Reality & Reason |
| :--- | :---: | :--- | :--- |
| **Object Detection Pipeline** | **VERIFIED FUNCTIONAL** | `src/detection/yolo_detector.py` | Runs YOLOv8n via Ultralytics API with COCO class mapping |
| **COCO Pretrained Weights** | **VERIFIED PRESENT** | `yolov8n.pt` (6.25 MB) | Official Ultralytics weights at project root |
| **Custom Fine-Tuned Weights** | **MISSING** | `models/weights/` | Directory contains only `.gitkeep`. No fine-tuning executed |
| **ABODA Dataset** | **NOT DOWNLOADED** | `data/raw/aboda/` | Ingestion script exists, but raw dataset was never downloaded |
| **PETS2006 Dataset** | **NOT DOWNLOADED** | `data/raw/pets_avss/` | Ingestion script exists, but raw dataset was never downloaded |
| **ByteTrack Algorithm** | **VERIFIED FUNCTIONAL** | `src/tracking/byte_tracker.py` | Pure NumPy/SciPy 2-stage tracking with bag threshold optimization |
| **Spatial-Temporal Associator** | **VERIFIED FUNCTIONAL** | `src/association/owner_associator.py` | Multi-factor weighted composite scoring engine |
| **Temporal State Machine** | **VERIFIED FUNCTIONAL** | `src/temporal/state_machine.py` | 7-state automaton with stationary check, hysteresis & occlusion grace |
| **Deep Temporal LSTM/MLP** | **UNTRAINED SKELETON** | `src/temporal/temporal_model.py` | Architecture defined in code, but inactive (`use_temporal_model: false`)|
| **Interactive Streamlit UI** | **VERIFIED FUNCTIONAL** | `app.py` | Web dashboard, webcam/phone feeds, controls, and HTML event tables |
| **Dual Audible Alert** | **VERIFIED FUNCTIONAL** | `app.py` (lines 20-38, 320-350) | Background `winsound.Beep` + browser Web Audio API oscillator |
| **ONNX Runtime Export & Benchmark**| **VERIFIED FUNCTIONAL** | `models/onnx/yolov8n.onnx` | Exported and benchmarked at 50.16 ms latency on CPU |
| **NVIDIA TensorRT Engine** | **NONFUNCTIONAL ON HOST** | `scripts/build_tensorrt.py` | Script present, but host lacks NVIDIA GPU / CUDA / TensorRT SDK |
| **Unit Test Suite** | **VERIFIED PASSING** | `tests/` | 15/15 unit tests pass via `pytest tests/ -v` |

---

## 2. Phased Action Plan for Future Work / Production Hardening

### Phase 1: Real-World Dataset Acquisition & Ingestion (Immediate Priority)
1. **Download ABODA:**
   - Download the 11 official ABODA AVI video sequences and XML ground truth annotations into `data/raw/aboda/`.
   - Run `python scripts/prepare_aboda.py` to extract frames and generate normalized YOLO labels in `data/processed/yolo_dataset/`.
2. **Download PETS2006:**
   - Download multi-camera scenarios S1 through S7 into `data/raw/pets_avss/`.
   - Run `python scripts/prepare_pets_avss.py` to extract high-density occlusion sequences.
3. **Sequence-Level Verification:**
   - Verify that all train, validation, and test splits strictly separate distinct video sequences to eliminate temporal data leakage.

### Phase 2: Domain-Specific Fine-Tuning (Model Enhancement)
1. **Execute Fine-Tuning:**
   - Run `python scripts/train_detector.py --config configs/detector.yaml --epochs 50 --batch 16`.
   - Fine-tune YOLOv8n on the processed 2-class dataset (`person`, `bag`).
2. **Checkpoint Deployment:**
   - Copy the resulting `best.pt` into `models/weights/best.pt`.
   - The Streamlit UI will automatically discover `models/weights/best.pt` and place it as the default option in the sidebar selector.
3. **Re-evaluate Detector:**
   - Run `python scripts/evaluate_detector.py` to generate validation mAP@0.5 and mAP@0.5:0.95 curves and confusion matrices.

### Phase 3: Hardware Acceleration with TensorRT (Edge Deployment)
1. **Target Hardware Setup:**
   - Set up an edge system with an NVIDIA GPU (e.g., NVIDIA Jetson Orin Nano / Jetson Xavier or an RTX workstation).
   - Install CUDA Toolkit 12.x, cuDNN, and TensorRT via `pip install tensorrt tensorrt-cu12`.
2. **Build Engine:**
   - Run `python scripts/build_tensorrt.py --onnx models/onnx/yolov8n.onnx --output models/tensorrt/yolov8n.engine --fp16`.
3. **Benchmark:**
   - Run `python scripts/benchmark.py` on the Jetson to measure real hardware power draw in Watts, GPU VRAM utilization, and FPS per Watt.

### Phase 4: Camera Perspective Calibration (Algorithmic Hardening)
1. **Homography Calibration:**
   - Add a camera calibration module using 4 marked ground reference points to calculate planar homography matrix $H$.
2. **Metric Distance Enforcement:**
   - Transform bounding-box bottom center points to real ground-plane coordinates $(X_m, Y_m)$.
   - Replace pixel distance thresholds ($180\text{ px}$) with real metric distances ($2.5\text{ meters}$), making the system invariant to camera focal length and object distance.
