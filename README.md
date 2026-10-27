# Unattended / Abandoned Object Detection with Temporal Reasoning

An end-to-end, edge-deployable Deep Learning and Computer Vision system designed to detect abandoned luggage and unattended objects in surveillance environments (airports, campus lobbies, transit corridors).

Unlike naive systems that classify every detected bag as "abandoned", this architecture integrates **Multi-Object Tracking (ByteTrack)**, **Temporal Owner–Bag Association**, and a **Deterministic Temporal State Machine** to verify sustained owner departure and object stationary persistence before raising an alarm.

---

## 🌟 Key Features & Pipeline Architecture

```
INPUT VIDEO / CCTV STREAM
           │
           ▼
PERSON + BAG DETECTION (Lightweight YOLOv8 Nano)
           │
           ▼
MULTI-OBJECT TRACKING (ByteTrack with Class-Consistent Kalman States)
           │
           ▼
TEMPORAL OWNER–BAG ASSOCIATION
(Composite Scoring: Proximity + Consistency + Co-movement)
           │
           ▼
TEMPORAL REASONING STATE MACHINE
(UNKNOWN ➔ ATTENDED ➔ OWNER_MOVING_AWAY ➔ SEPARATED ➔ SUSPECTED ➔ ABANDONED)
           │
      ┌────┴───────────────────────────┐
      ▼                                ▼
STATIONARY VERIFICATION       OWNER SEPARATION VERIFICATION
(Displacement < Thresh)       (Distance > Thresh)
      └────┬───────────────────────────┘
           ▼
ABANDONMENT TIMER COUNTDOWN
           │
           ▼
🚨 ALARM TRIGGER & VISUAL DEMONSTRATION (Streamlit UI)
```

---

## 📂 Project Structure

```
abandoned_object_detection/
├── app.py                     # Streamlit Interactive Web Application
├── requirements.txt           # Python 3.11 dependencies
├── .gitignore                 # Version control exclusions
├── configs/
│   ├── detector.yaml          # YOLO training & inference hyperparameters
│   └── system.yaml            # Tracking, association, temporal thresholds
├── data/
│   ├── raw/
│   │   ├── aboda/             # Raw ABODA videos and XML annotations
│   │   └── pets_avss/         # PETS 2006 / AVSS benchmark videos
│   ├── processed/
│   │   └── yolo_dataset/      # Normalized YOLO train/val/test splits
│   ├── annotations/           # Ground truth event definitions
│   └── local/
│       ├── videos/            # Staged campus scenario MP4s
│       └── annotations/       # Event labels (JSON format)
├── models/
│   ├── weights/               # PyTorch model checkpoints (.pt)
│   ├── onnx/                  # Exported ONNX graphs (.onnx)
│   └── tensorrt/              # Compiled TensorRT engines (.engine)
├── scripts/
│   ├── prepare_aboda.py       # Ingests ABODA with sequence-level splits
│   ├── prepare_pets_avss.py   # Ingests PETS/AVSS benchmark sequences
│   ├── create_local_dataset.py# Generates staged campus test scenarios
│   ├── analyze_dataset.py     # Generates dataset class balance & metrics
│   ├── train_detector.py      # Transfer learning training pipeline
│   ├── evaluate_detector.py   # COCO mAP evaluation on held-out test split
│   ├── evaluate_system.py     # End-to-end multi-experiment ablation study
│   ├── export_onnx.py         # Exports PyTorch to ONNX format
│   ├── build_tensorrt.py      # Compiles ONNX to TensorRT FP16 engine
│   ├── benchmark.py           # Benchmarks PyTorch vs ONNX vs TensorRT + Power
│   └── run_inference.py       # Standalone CLI inference runner
├── src/
│   ├── detection/             # BaseDetector & YOLODetector wrapper
│   ├── tracking/              # Track dataclass & ByteTracker
│   ├── association/           # Spatial metrics & OwnerAssociator
│   ├── temporal/              # AbandonmentState enum & TemporalStateMachine
│   ├── evaluation/            # Detection & Event-level evaluation engines
│   ├── inference/             # SurveillancePipeline & PipelineVisualizer
│   └── utils/                 # YAML config loader, logger, PowerMonitor
├── results/
│   ├── figures/               # PR curves and dataset distributions
│   ├── metrics/               # System and detection JSON / CSV metrics
│   └── reports/               # Markdown evaluation summaries
├── docs/
│   ├── dataset.md             # Dataset acquisition & directory setup
│   ├── training.md            # Detector training guidelines
│   ├── local_dataset_collection.md # Campus recording & privacy protocol
│   ├── annotation.md          # YOLO and Event JSON formats
│   └── deployment.md          # ONNX & TensorRT deployment instructions
└── tests/                     # 15 Comprehensive Pytest unit tests
```

---

## 🛠️ Environment Setup

### 1. Create Virtual Environment (Python 3.11)

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📊 Dataset Preparation

### Primary Dataset: ABODA
1. Download the ABODA sequences from the [ABODA Dataset Repository](https://github.com/flysoaryang/ABODA).
2. Place the video files (`.avi`) and XML annotations into `data/raw/aboda/`.
3. Run sequence-level extraction (prevents temporal data leakage):
   ```bash
   python scripts/prepare_aboda.py --raw_dir data/raw/aboda --output_dir data/processed/yolo_dataset
   ```

### Staged Campus Dataset
For instant verification without external downloads, generate staged scenario videos covering edge cases:
```bash
python scripts/create_local_dataset.py --local_dir data/local
```

### Dataset Statistics & Distribution Plot
```bash
python scripts/analyze_dataset.py --dataset_dir data/processed/yolo_dataset
```

---

## 🎯 Model Training & Evaluation

### 1. Fine-Tune Detector (Transfer Learning)
```bash
python scripts/train_detector.py --config configs/detector.yaml --data data/processed/yolo_dataset/dataset.yaml --epochs 50 --batch 16
```

### 2. Evaluate Detector (COCO Metrics)
```bash
python scripts/evaluate_detector.py --model models/weights/best.pt --data data/processed/yolo_dataset/dataset.yaml --split test
```

### 3. Evaluate Complete System (Multi-Experiment Ablation)
Runs Experiments A, B, and D comparing naive detection vs tracking vs the proposed temporal state machine:
```bash
python scripts/evaluate_system.py --config configs/system.yaml --frame_stride 2
```
Outputs are written to:
- `results/metrics/system_metrics.json`
- `results/metrics/system_metrics.csv`
- `results/reports/system_evaluation_report.md`

---

## 🚀 Running Inference

### CLI Inference on Video or Webcam
```bash
python scripts/run_inference.py --source data/local/videos/campus_true_abandoned.mp4 --output results/annotated_output.mp4
```

To run on a live USB/CCTV webcam:
```bash
python scripts/run_inference.py --source 0
```

---

## 🖥️ Streamlit Interactive Dashboard

Launch the web interface:
```bash
streamlit run app.py
```

### Web UI Features:
- **Interactive Scenarios:** Test pre-loaded scenarios (True Abandoned, Owner Returns, Attended Luggage) or upload custom surveillance clips.
- **Real-Time Visual Overlays:** Bounding boxes, Track IDs (`P01`, `B01`), Owner connection line with Euclidean distance meter, state badges, and countdown timer.
- **Dynamic Threshold Tuning:** Change abandonment countdown (3s–60s), separation distance (50px–500px), and stationary sensitivity on the fly via sidebar sliders.
- **Live Telemetry & Logs:** Real-time FPS, Latency (ms), CPU/GPU usage, power consumption, and chronological security event log.

---

## ⚡ ONNX & TensorRT Optimization

### Export to ONNX
```bash
python scripts/export_onnx.py --model yolov8n.pt --output_dir models/onnx --imgsz 640
```

### Compile to TensorRT Engine (NVIDIA GPU required)
```bash
python scripts/build_tensorrt.py --onnx models/onnx/yolov8n.onnx --output models/tensorrt/yolov8n.engine --fp16
```

### Benchmark Runtimes & Measure Power Draw
```bash
python scripts/benchmark.py --pt yolov8n.pt --onnx models/onnx/yolov8n.onnx
```
*Note: Measures GPU power draw via NVML (`pynvml`) and calculates throughput efficiency (FPS/Watt).*

---

## 🧪 Running Unit Tests

Execute the automated test suite covering distance metrics, stationary detection, owner association, timer progression, state transitions, alerts, and recovery:
```bash
pytest tests/ -v
```

---

## 🔬 Experimental Results

### Baseline vs. Ablation Comparison

| Experiment | Pipeline Configuration | Precision | Recall | F1 Score | False Alarms / Hr |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Exp A** | Detector Only (Naive) | 0.005 | 1.000 | 0.010 | ~1440.0 / hr |
| **Exp B** | Detector + ByteTrack (No Owner) | 0.333 | 1.000 | 0.500 | 144.0 / hr |
| **Exp D** | **Full Pipeline (Temporal State Machine)** | **1.000** | **1.000** | **1.000** | **0.0 / hr** |

*Key Takeaway:* The temporal state machine eliminates false alarms caused by people sitting next to luggage (`NORMAL_ATTENDED`) and owner return events (`OWNER_RETURNS`), cutting false alarms per hour to zero on controlled evaluation scenarios.

---

## ⚠️ Limitations & Future Work

1. **Long-Term Heavy Occlusion:** When a luggage item is obscured by heavy crowds for > 3.0s (the occlusion grace period), track continuity may fragment. Integrating ReID appearance embeddings (DeepSORT / BoT-SORT) will further strengthen re-identification.
2. **Dense Luggage Clusters:** At airport baggage carousels where 10+ bags sit contiguous to each other, centroid clustering requires calibration.
3. **Perspective Distortion:** Homography calibration from camera pixels to real-world ground meters improves distance measurement consistency across deep corridors.
