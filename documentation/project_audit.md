# Project Implementation Audit: Unattended / Abandoned Object Detection with Temporal Reasoning

**Date of Audit:** October 27, 2026  
**Auditor:** Senior Computer Vision Engineer & Academic Project Evaluator  
**Repository Path:** `C:\Users\Aditya\.gemini\antigravity\scratch\abandoned_object_detection`  
**Host Environment:** Windows 11, Intel64 Family 6 Model 140 (8 Logical Cores, CPU-only), Python 3.11.9  

---

## Executive Summary

This forensic audit evaluates the actual codebase against the stated project goals:
*“Unattended / Abandoned Object Detection with Temporal Reasoning for Edge AI Surveillance.”*

The codebase contains a **fully functioning, end-to-end deterministic edge surveillance pipeline** consisting of:
1. Object detection via Ultralytics YOLOv8n (COCO-pretrained).
2. Class-consistent multi-object tracking via a custom two-stage ByteTrack implementation.
3. Multi-factor spatial-temporal person–bag ownership association.
4. Deterministic 7-state finite state machine with stationary validation, hysteresis, and occlusion tolerance.
5. Interactive Streamlit user interface (`app.py`) supporting live webcam, phone stream over WiFi (Iriun / IP Webcam), file uploads, dual audible siren alerts, and live telemetry.
6. ONNX export pipeline and CPU-based benchmark scripts.

However, several components frequently referenced in academic literature and proposal documents (fine-tuned custom weights, real ABODA / PETS2006 datasets, trained deep LSTM temporal models, and NVIDIA TensorRT compilation) are either **unexecuted, absent, or nonfunctional on the current host machine**.

---

## Itemized Forensic Audit (Points 1 – 15)

### 1. Programming Languages and Libraries Used
* **Primary Language:** Python 3.11.9 (verified in `.venv\Scripts\python.exe`).
* **Deep Learning Framework:** `torch==2.14.1+cpu`, `torchvision==0.19.1+cpu` (`torch.cuda.is_available() == False`).
* **Computer Vision & Image Processing:** `opencv-python==4.11.0.86`, `ultralytics==8.3.179`, `pillow==11.1.0`.
* **Scientific Computing & Optimization:** `numpy==1.26.4`, `scipy==1.15.2` (specifically `scipy.optimize.linear_sum_assignment` for Hungarian bipartite matching).
* **Data Processing & Serialization:** `pandas==2.2.3`, `pyarrow==17.0.0` (pinned to prevent NumPy 2.0 ABI conflicts), `pyyaml==6.0.2`.
* **Interactive Frontend & Audio:** `streamlit==1.66.0`, `winsound` (Windows native C-extension for hardware beeps), Web Audio API (inline JavaScript synthesizer).
* **Deployment & Benchmarking:** `onnx==1.17.0`, `onnxruntime==1.20.1`, `psutil==7.0.0`, `pynvml==12.0.0`.
* **Testing:** `pytest==9.1.1` (15/15 unit tests passing).
* **Citation:** `requirements.txt`, `app.py` lines 1–55, `pyproject.toml` / `.venv\Lib\site-packages`.

---

### 2. Object Detection Model: Architecture and Version
* **Model Family:** YOLOv8 (You Only Look Once, Version 8), Nano variant (`yolov8n`).
* **Exact Architecture:**
  - Backbone: Modified CSPDarknet53 with C2f (Cross-Stage Partial with 2 convolutions) blocks.
  - Neck: Path Aggregation Network (PANet) with feature pyramid levels P3 (stride 8, small objects), P4 (stride 16, medium objects), P5 (stride 32, large objects).
  - Head: Anchor-free decoupled head separating bounding box regression (DFL — Distribution Focal Loss) and object classification (BCE Loss).
  - Parameters: ~3.16 Million (3,157,200 params).
  - Computational Complexity: 8.7 GFLOPs at 640×640 input resolution; ~3.1 GFLOPs at 384×384 input resolution.
* **Citation:** `yolov8n.pt`, `configs/detector.yaml` line 4, `src/detection/yolo_detector.py` lines 21–65.

---

### 3. Model Checkpoint / Weights Filename and Source
* **Filename:** `yolov8n.pt` (stored at project root, size: 6,549,796 bytes / 6.25 MB).
* **Source:** Official Ultralytics pretrained weights downloaded from Ultralytics release assets (`https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt`).
* **Trained Weights (`models/weights/best.pt`):** **DOES NOT EXIST.** Directory `models/weights/` contains only an empty `.gitkeep` file.
* **Citation:** File inspection of `models/weights/`, `yolov8n.pt`, and `app.py` lines 91–96.

---

### 4. Training Status: Pretrained vs. Fine-Tuned vs. From Scratch
* **Verified Status:** **100% PRETRAINED BASELINE.**
* **Explanation:** No fine-tuning or custom training was executed in this repository. Although a training script (`scripts/train_detector.py`) and configuration (`configs/detector.yaml`) exist, the dataset directory `data/processed/yolo_dataset/` contains 0 images, and `results/metrics/dataset_statistics.json` confirms `"total_images": 0`.
* **Examiner Warning:** In an exam or viva, you **must not** claim that you trained or fine-tuned YOLO. You must state:
  > *"We utilized the official COCO-pretrained YOLOv8n weights as a zero-shot detector and engineered a downstream tracking, spatial-temporal association, and deterministic state-reasoning architecture on top of it."*
* **Citation:** `results/metrics/dataset_statistics.json`, `scripts/train_detector.py`, `models/weights/`.

---

### 5. Object Classes Detected and Mapping Strategy
* **COCO Source Classes:** 80 classes.
* **Internal Target Representation (2 Classes):**
  - Class 0: `person`
  - Class 1: `bag` (Unattended Object Candidate)
* **Active COCO-to-Target Mapping in Code:**
  - `0 -> 0` (`person`)
  - `24 -> 1` (`backpack`)
  - `26 -> 1` (`handbag`)
  - `28 -> 1` (`suitcase`)
  - `63 -> 1` (`laptop` — added to detect left-behind portable electronic equipment)
* **Citation:** `configs/system.yaml` lines 12–17, `src/detection/yolo_detector.py` lines 71–77.

---

### 6. Multi-Object Tracking Algorithm
* **Algorithm:** ByteTrack (Zhang et al., ECCV 2022), custom class-consistent implementation in pure Python/NumPy/SciPy.
* **Implementation Details:**
  - Associates bounding boxes across frames using a 2-stage bipartite matching process based on Intersection-over-Union (IoU) distance ($1 - \text{IoU}$).
  - **Stage 1 (High-Confidence):** Matches active tracks against detections where $\text{conf} \ge \text{effective\_thresh}$. Detections match if cost $< \text{match\_thresh}$ (0.80).
  - **Stage 2 (Low-Confidence Recovery):** Matches remaining unmatched tracks against weak detections ($0.05 \le \text{conf} < \text{effective\_thresh}$) with relaxed cost $< \text{match\_thresh} + 0.10$ to maintain tracks through motion blur and partial occlusion.
  - **Bag Track Initiation Optimization:** For bags (`class_id == 1`), `effective_thresh` is adaptively set to $\min(\text{track\_thresh}, 0.20)$ to prevent dim or low-contrast luggage from being discarded.
  - **Track State & Motion Model:** Linear constant-velocity Kalman-like kinematic estimation tracking $[x_{center}, y_{center}, w, h, v_x, v_y]$ with a maximum track buffer of 60 frames (~2.0 seconds).
* **Citation:** `src/tracking/byte_tracker.py` lines 51–205, `src/tracking/track_state.py` lines 1–110, `configs/system.yaml` lines 18–25.

---

### 7. Person–Bag Ownership Association Algorithm
* **Algorithm:** Multi-Factor Spatial-Temporal Associator (`OwnerAssociator`).
* **Mathematical Association Formulation:**
  For every candidate pair of person $p$ and bag $b$, composite score $S(p, b)$ is computed:
  $$S(p, b) = w_{dist} \cdot S_{dist} + w_{temp} \cdot S_{temp} + w_{motion} \cdot S_{motion}$$
  Configured weights: $w_{dist} = 0.45$, $w_{temp} = 0.35$, $w_{motion} = 0.20$.
  - **Spatial Proximity ($S_{dist}$):**
    $$S_{dist} = \max\left(0.0, 1.0 - \frac{d_{edge}(p, b)}{d_{max}}\right)$$
    where $d_{edge}$ is minimum Euclidean distance between bounding-box perimeters ($d_{edge} = 0$ if boxes overlap), and $d_{max} = 250.0\text{ px}$.
  - **Temporal Consistency ($S_{temp}$):**
    $$S_{temp} = \min\left(1.0, \frac{\Delta t_{proximity}}{t_{attended\_min}}\right)$$
    accumulates persistence over time ($t_{attended\_min} = 1.0\text{ s}$).
  - **Kinematic Correlation ($S_{motion}$):**
    $$S_{motion} = \frac{\mathbf{v}_p \cdot \mathbf{v}_b}{\|\mathbf{v}_p\| \|\mathbf{v}_b\| + \epsilon}$$
    evaluates directional velocity alignment between moving person and moving bag.
* **Owner Confirmation Rule:** Owner is locked if $S(p, b) \ge 0.40$ continuously for $\ge 1.0\text{ second}$.
* **Citation:** `src/association/owner_associator.py` lines 40–165, `src/association/spatial_metrics.py` lines 1–115.

---

### 8. Temporal Reasoning Mechanism and State Machine
* **Mechanism:** Deterministic 7-State Finite State Machine (`TemporalStateMachine`).
* **States Defined:**
  1. `UNKNOWN`: Initial transitory state upon track creation.
  2. `ATTENDED`: Object stationary; confirmed owner within distance threshold ($d \le 180\text{ px}$).
  3. `OWNER_MOVING_AWAY`: Owner distance increasing with departure velocity $> 1.5\text{ px/s}$.
  4. `SEPARATED`: Owner distance $> 180\text{ px}$ (or owner left field of view); bag stationary ($< 25\text{ px}$ displacement over 3.0s window). Separation timer $\Delta t_{sep}$ begins accumulating.
  5. `SUSPECTED_ABANDONED`: Separation timer reaches $\ge 50\%$ of threshold ($\Delta t_{sep} \ge 7.5\text{ s}$).
  6. `ABANDONED`: Separation timer reaches or exceeds threshold ($\Delta t_{sep} \ge 15.0\text{ s}$). Security alert generated.
  7. `RECOVERED`: Triggered if owner returns within return threshold ($d \le 120\text{ px}$) OR if bag is physically moved/picked up ($> 25\text{ px}$ displacement).
* **Occlusion Tolerance:** If a bag track drops due to line-of-sight occlusion by walking pedestrians, the state machine holds state and timers for up to $3.0\text{ s}$ (`occlusion_grace_period`) before track deletion.
* **Alert Cooldown:** Rate-limits repeated alerts for the same physical object to once every $30.0\text{ s}$.
* **Citation:** `src/temporal/states.py` lines 10–45, `src/temporal/state_machine.py` lines 37–293.

---

### 9. Abandonment Decision Rules and Thresholds
From `configs/system.yaml` and runtime overrides in `app.py`:
* **Separation Distance Threshold ($d_{sep}$):** $180.0\text{ px}$ (owner classified as departed if $d > 180\text{ px}$).
* **Owner Return Threshold ($d_{ret}$):** $120.0\text{ px}$ (hysteresis margin of $60\text{ px}$ prevents boundary oscillation).
* **Abandonment Time Threshold ($T_{abandon}$):** $15.0\text{ seconds}$ (user configurable in UI from $3.0$ to $60.0\text{ s}$).
* **Stationary Movement Threshold:** $\le 25.0\text{ px}$ displacement over a rolling $3.0\text{ second}$ window.
* **Owner Departure Velocity:** $> 1.5\text{ px/s}$ relative separating velocity.
* **Occlusion Grace Window:** $3.0\text{ seconds}$.
* **Citation:** `configs/system.yaml` lines 38–49, `app.py` lines 100–110.

---

### 10. Input Sources Supported
* **Synthetic Video Files:** 3 staged test videos generated by OpenCV script (`data/local/videos/`):
  - `campus_true_abandoned.mp4` (Person leaves bag and walks away -> alert).
  - `campus_owner_returns.mp4` (Person walks away, returns at $t=8\text{s}$ -> timer resets).
  - `campus_normal_attended.mp4` (Person stands next to bag -> no alert).
* **User Video Upload:** Interactive `.mp4`, `.avi`, `.mov`, `.mkv` upload via Streamlit.
* **Live Webcam / Virtual Phone Cam:** Device indices `0` (Laptop Integrated Cam), `1` (Iriun Webcam / DroidCam), `2` (Secondary USB Cam).
* **Network CCTV / IP Stream:** RTSP / HTTP URL input field (e.g., `http://192.168.1.X:8080/video` via IP Webcam app).
* **Citation:** `app.py` lines 140–191.

---

### 11. User Interface and Alert Mechanisms
* **Framework:** Streamlit web application running at `http://localhost:8501`.
* **Visual Display:** Annotated video canvas displaying bounding boxes, track IDs, Euclidean association lines, abandonment countdown timers, and live FPS/latency overlays.
* **Visual Alert Banner:** Pulsing high-contrast red warning box:
  `🚨 SECURITY ALERT: Bag B0X left unattended for Y.Ys!`
* **Audible Alert Mechanism (Dual Channel):**
  1. *Hardware Siren:* Asynchronous `winsound.Beep(2200, 160)` executed in a background Python daemon thread (throttled to 1.5s interval to prevent audio stutter).
  2. *Browser Audio Synthesizer:* Inline Web Audio API JavaScript oscillator rendering a 880Hz -> 1320Hz alert tone in the browser tab.
* **Crash-Resilient Event Table:** Event log rendering using Pandas with automatic HTML table fallback to prevent PyArrow serialization exceptions.
* **Citation:** `app.py` lines 13–70, 200–225, 318–355.

---

### 12. Deployment Hardware and Inference Framework
* **Actual Runtime Tested:** Python 3.11 with PyTorch CPU backend (`torch==2.14.1+cpu`) and ONNX Runtime CPU (`onnxruntime==1.20.1` using `CPUExecutionProvider`).
* **Target Hardware Profile in Code:** Designed for Edge AI platforms (NVIDIA Jetson Nano / Xavier / Orin, Raspberry Pi 5, or x86 Edge IPCs). Includes `PowerMonitor` supporting NVIDIA NVML and Jetson `jtop`.
* **Measured Hardware Status on Current System:** Intel CPU, 8 logical threads, 0 CUDA GPUs available.
* **Citation:** `src/utils/power_monitor.py` lines 1–95, `results/metrics/benchmark_metrics.json`.

---

### 13. ONNX and TensorRT Implementation Status
* **ONNX:** **FULLY IMPLEMENTED AND VERIFIED.**
  - Exporter script: `scripts/export_onnx.py`.
  - Exported model: `models/onnx/yolov8n.onnx` and root `yolov8n.onnx` (12.26 MB, opset 12, FP32).
  - Benchmark result: 50.16 ms mean latency, 19.93 FPS on CPU.
* **TensorRT:** **CODE SKELETON PRESENT, BUT NONFUNCTIONAL ON THIS HOST.**
  - Builder script: `scripts/build_tensorrt.py` contains `trtexec` wrapper and `tensorrt` Python API builder logic.
  - Actual Status: No engine file exists in `models/tensorrt/`. Host machine lacks NVIDIA GPU, CUDA, and TensorRT SDK. `benchmark_metrics.json` officially records:
    `"TensorRT": "Not Available (Engine file does not exist. Run scripts/build_tensorrt.py on a machine with TensorRT)."`
* **Citation:** `scripts/export_onnx.py`, `scripts/build_tensorrt.py`, `results/metrics/benchmark_metrics.json` lines 38–41.

---

### 14. Existing Evaluation Scripts and Measured Results
* **`scripts/benchmark.py`:** Benchmarks inference latency and throughput.
  - PyTorch CPU: $61.37\text{ ms}$ ($16.29\text{ FPS}$) at $640\times 640$; at $384\times 384$: $\sim 24.2\text{ ms}$ ($41.3\text{ FPS}$).
  - ONNX Runtime CPU: $50.16\text{ ms}$ ($19.93\text{ FPS}$) at $640\times 640$.
* **`scripts/evaluate_system.py`:** Runs event-level evaluation on staged videos.
  - Results in `results/metrics/system_metrics.json` and `results/reports/system_evaluation_report.md`:
  - Due to synthetic test videos using simplified geometric box sprites, the standard COCO detector detected 0 bags, yielding $\text{TP}=0, \text{FP}=0, \text{FN}=1$ ($\text{Precision}=0.0, \text{Recall}=0.0, \text{F1}=0.0$).
* **Unit Test Suite:** 15 unit tests in `tests/` covering alert generation, association scores, spatial distance metrics, occlusion recovery, state transitions, stationary logic, and timer accumulation. All 15 pass (`15 passed in 3.65s`).
* **Citation:** `results/metrics/system_metrics.json`, `results/metrics/benchmark_metrics.json`, `tests/`.

---

### 15. Missing, Incomplete, Duplicated, or Nonfunctional Components
1. **Absence of Real Benchmark Datasets:** Neither ABODA nor PETS2006/AVSS datasets are present in `data/raw/`. Only `.gitkeep` files exist.
2. **Absence of Fine-Tuned Weights:** `models/weights/best.pt` does not exist. The project uses standard COCO `yolov8n.pt`.
3. **No Trained Deep Temporal Model:** `src/temporal/temporal_model.py` defines `TemporalLSTMClassifier` and `TemporalMLPClassifier`, but they are unweighted, untrained, and disabled (`use_temporal_model: false` in `system.yaml`).
4. **Nonfunctional TensorRT on Host:** No `.engine` file exists in `models/tensorrt/` due to lack of an NVIDIA GPU and TensorRT SDK on the development machine.
5. **No Full Quantitative Benchmark on Real Surveillance Sequences:** The system evaluation was only executed on 3 local synthetic MP4 videos, not on real public CCTV sequences.

---

## Audit Verification Summary Table

| Component | Status in Codebase | Implementation File | Verification Evidence |
| :--- | :---: | :--- | :--- |
| **YOLOv8n Detector** | Verified Functional | `src/detection/yolo_detector.py` | Runs inference via Ultralytics API |
| **COCO Pretrained Weights** | Verified Present | `yolov8n.pt` (6.25 MB) | Located at project root |
| **Fine-Tuned Weights (`best.pt`)** | **MISSING** | `models/weights/` | Directory contains only `.gitkeep` |
| **ByteTrack Algorithm** | Verified Functional | `src/tracking/byte_tracker.py` | 2-stage association with SciPy Hungarian |
| **Spatial-Temporal Associator**| Verified Functional | `src/association/owner_associator.py` | Multi-factor weighted composite scoring |
| **Finite State Machine** | Verified Functional | `src/temporal/state_machine.py` | 7 states, stationary logic, timer resets |
| **Streamlit Interface** | Verified Functional | `app.py` | Live webcam, WiFi camera, siren alarms |
| **ONNX Export & Runtime** | Verified Functional | `models/onnx/yolov8n.onnx` | 50.16 ms latency on ONNX CPU |
| **TensorRT Engine** | **NONFUNCTIONAL** | `scripts/build_tensorrt.py` | No GPU / engine file (`models/tensorrt/` empty)|
| **ABODA Dataset** | **NOT DOWNLOADED** | `data/raw/aboda/` | Empty directory (`.gitkeep` only) |
| **PETS2006 Dataset** | **NOT DOWNLOADED** | `data/raw/pets_avss/` | Empty directory (`.gitkeep` only) |
| **Unit Test Suite** | Verified Passing | `tests/` | 15/15 tests passing via pytest |
