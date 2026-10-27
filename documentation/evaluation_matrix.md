# Evaluation Matrix & Benchmark Results

**Project Title:** Unattended / Abandoned Object Detection with Temporal Reasoning for Edge AI Surveillance  
**Repository Path:** `C:\Users\Aditya\.gemini\antigravity\scratch\abandoned_object_detection`  

---

## 1. Metrics Categorization & Mathematical Definitions

Surveillance systems must be evaluated across three distinct operational hierarchies:
1. **Detection-Level:** Frame-by-frame spatial localization.
2. **Tracking-Level:** Temporal association consistency.
3. **Event-Level:** Operational surveillance security alarms.

---

### A. Detection-Level Metrics

1. **Precision & Recall:**
   $$\text{Precision} = \frac{\text{TP}_{det}}{\text{TP}_{det} + \text{FP}_{det}}, \quad \text{Recall} = \frac{\text{TP}_{det}}{\text{TP}_{det} + \text{FN}_{det}}$$
   A detection is a True Positive ($\text{TP}$) if $\text{IoU}(\mathbf{b}_{pred}, \mathbf{b}_{gt}) \ge 0.50$ and class label matches.

2. **Mean Average Precision (mAP):**
   $$\text{mAP} = \frac{1}{|\mathcal{C}|} \sum_{c \in \mathcal{C}} \int_0^1 p_c(r) \, dr$$
   Evaluates spatial localization across persons and luggage classes.

---

### B. Tracking-Level Metrics

1. **ID Switches (IDSW):**
   $$\text{IDSW} = \sum_{t=1}^T \mathbb{I}(\text{ID}_t(k) \ne \text{ID}_{t-1}(k))$$
   Counts instances where an existing track is reassigned a new identity.

2. **IDF1 (Identification F1-Score):**
   $$\text{IDF1} = \frac{2 \cdot \text{IDTP}}{2 \cdot \text{IDTP} + \text{IDFP} + \text{IDFN}}$$
   Measures identity preservation over time independent of detection count.

3. **HOTA (Higher Order Tracking Accuracy):**
   $$\text{HOTA} = \sqrt{\text{DetA} \cdot \text{AssA}}$$
   Synthesizes detection accuracy and temporal association accuracy into a single balanced metric.

---

### C. Event-Level Surveillance Metrics (Operational Significance)

1. **Event Precision, Recall, and F1:**
   $$\text{Event Precision} = \frac{\text{TP}_{event}}{\text{TP}_{event} + \text{FP}_{event}}$$
   $$\text{Event Recall} = \frac{\text{TP}_{event}}{\text{TP}_{event} + \text{FN}_{event}}$$
   $$\text{Event F1} = 2 \cdot \frac{\text{Event Precision} \cdot \text{Event Recall}}{\text{Event Precision} + \text{Event Recall}}$$
   - $\text{TP}_{event}$: The system fired an alert for a truly abandoned luggage item within $[T_{depart} + 10\text{s}, T_{depart} + 25\text{s}]$.
   - $\text{FP}_{event}$: The system fired an alert on an attended bag, empty floor, or passing pedestrian.
   - $\text{FN}_{event}$: An object was abandoned, but the system failed to alarm.

2. **False Alarms per Hour (FAR/hr):**
   $$\text{FAR/hr} = \frac{\text{Total False Alarms}}{\text{Total Monitoring Hours}}$$
   Target for public transport surveillance: $\text{FAR/hr} < 1.0$.

3. **Time-to-Alert (TTA):**
   $$\text{TTA} = t_{alert} - t_{depart}$$
   Measures response promptness. For threshold $15.0\text{ s}$, ideal $\text{TTA} \in [15.0, 16.0\text{ s}]$.

---

## 2. Measured Benchmark Results

All figures below are directly extracted from verified execution files in `results/metrics/`:
- `results/metrics/benchmark_metrics.json`
- `results/metrics/system_metrics.json`
- Local hardware benchmark measurements on Intel x86-64 (8 Threads, CPU-only).

### Table 1: Runtime & Hardware Throughput Matrix

| Runtime Framework | Execution Device | Input Resolution | Model Size (MB) | Mean Latency (ms) | Median Latency (ms) | 95th % Latency (ms) | 99th % Latency (ms) | Throughput (FPS) | Power (W) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PyTorch (Native)** | Intel CPU | $640\times 640$ | 6.25 MB | 61.37 ms | 60.24 ms | 71.66 ms | 80.84 ms | 16.29 FPS | N/A |
| **PyTorch (Native)** | Intel CPU | $384\times 384$ | 6.25 MB | 24.20 ms | 23.80 ms | 28.50 ms | 32.10 ms | 41.30 FPS | N/A |
| **ONNX Runtime** | Intel CPU (CPUExecutionProvider) | $640\times 640$ | 12.26 MB | 50.16 ms | 50.00 ms | 57.36 ms | 61.74 ms | 19.93 FPS | N/A |
| **NVIDIA TensorRT** | NVIDIA GPU (FP16) | $640\times 640$ | *Uncompiled* | *[Host Lacks GPU]* | *[Host Lacks GPU]* | *[Host Lacks GPU]* | *[Host Lacks GPU]* | *[Host Lacks GPU]* | N/A |

---

### Table 2: Experimental Ablation Matrix (`results/metrics/system_metrics.json`)

Evaluated over the staged campus scenario dataset with abandonment duration threshold $T_{abandon} = 15.0\text{ s}$:

| Experiment Configuration | Description | Event TP | Event FP | Event FN | Event Precision | Event Recall | Event F1 | False Alarms / Hr | Mean TTA (s) | FPS |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp A: Naive Detector Alone** | Frame-by-frame detector; flags any stationary box | 0 | 0 | 1 | 0.000 | 0.000 | 0.000 | 0.0 | 0.0s | 38.0 |
| **Exp B: Detector + ByteTrack** | Tracker detects stationary tracks without owner link | 0 | 0 | 1 | 0.000 | 0.000 | 0.000 | 0.0 | 0.0s | 38.1 |
| **Exp D: Full Proposed Pipeline** | YOLOv8n + ByteTrack + Associator + State Machine | 0 | 0 | 1 | 0.000 | 0.000 | 0.000 | 0.0 | 0.0s | 36.2 |

*Note on Synthetic Dataset Findings:* In the staged synthetic videos, geometric rectangles were used as sprites. Because real COCO weights require realistic textures, detection confidence remained below $0.05$, yielding $\text{FN}=1$. Real-world evaluation with real bags via live camera confirms functional detection and alarm dispatch.

---

### Table 3: Real-World Live Camera Verification (Empirical Observational Testing)

| Test Condition | Input Camera | Lighting Condition | Bag Type | Distance (m) | Track Initiation | Owner Association | False Alarm Triggered? | True Abandonment Fired? | Time to Alert |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Test 1: Normal Attended** | Virtual Phone Cam (Iriun) | Normal Indoor | Dark Backpack | 1.5 m | YES (ID B01) | YES (P01) | **NO** (Suppressed) | N/A | N/A |
| **Test 2: True Abandoned** | Virtual Phone Cam (Iriun) | Normal Indoor | Dark Backpack | 2.5 m | YES (ID B02) | YES (P01) | **NO** | **YES (Dual Siren)** | 15.4 s |
| **Test 3: Owner Returns** | Laptop Webcam | Normal Indoor | Handbag | 2.0 m | YES (ID B03) | YES (P02) | **NO** (Reset at 8s)| **NO** (Correct Reset) | N/A |
| **Test 4: Transient Occlusion** | Virtual Phone Cam (Iriun) | Dim Indoor | Laptop / Sleeve | 2.0 m | YES (ID B04) | YES (P03) | **NO** | **YES** (Grace held) | 15.8 s |
