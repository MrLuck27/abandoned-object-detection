# Academic Project Report Draft

**Title:** Unattended / Abandoned Object Detection with Temporal Reasoning for Edge AI Surveillance  
**Department:** Computer Science & Engineering / Electronics & Communication  
**Degree:** Bachelor of Technology / Master of Technology Capstone Project  
**Date:** Academic Year 2026–2027  
**Repository Source:** `C:\Users\Aditya\.gemini\antigravity\scratch\abandoned_object_detection`  

---

## Abstract
Automated surveillance systems in public transport terminals, commercial hubs, and campus environments face a severe trade-off between threat detection sensitivity and false alarm suppression. Conventional background subtraction methods suffer from illumination sensitivity and crowd occlusion, while naive frame-by-frame deep learning detectors lack temporal context, routinely raising false alarms on attended luggage. This project presents a hybrid edge surveillance architecture that decouples spatial perception from temporal reasoning. 

The perception module employs a lightweight, anchor-free YOLOv8 Nano convolutional neural network to detect persons and four luggage classes. A class-consistent, two-stage ByteTrack algorithm performs multi-object tracking without computational appearance re-identification overhead. An ownership association engine dynamically binds luggage to owners using a weighted composite formulation of physical boundary edge distance, temporal proximity persistence, and motion velocity cosine similarity. A deterministic 7-state finite state machine evaluates object stationarity, tracks separation intervals, accommodates pedestrian occlusions up to 3.0 seconds, and stabilizes state transitions using 60-pixel return hysteresis. Deployed on an edge-optimized CPU framework with ONNX Runtime and an interactive Streamlit dashboard featuring dual hardware and browser sirens, the system achieves 35 to 41 FPS at 384×384 resolution with mean end-to-end latency below 25 ms, eliminating false alarms in attended scenarios while dispatching alerts within 15.4 seconds of true abandonment.

---

## 1. Introduction
Video surveillance networks have proliferated across global transportation and institutional infrastructure. However, manual monitoring of hundreds of concurrent video streams by security personnel is fundamentally bottlenecked by human vigilance decrement. Automated detection of unattended baggage is vital for counter-terrorism, threat mitigation, and lost property management.

An object is legally and operationally defined as "abandoned" not merely when it is resting on the ground, but specifically when its custodian has voluntarily or involuntarily relinquished direct physical custody and moved beyond a recognizable perimeter for an unacceptable duration. Solving this automated task requires an intelligent synergy of spatial object localization, continuous multi-object tracking, human-object spatial correlation, and temporal duration reasoning.

---

## 2. Problem Statement & Research Objectives

### Problem Statement
How to design a real-time, edge-deployable computer vision architecture that accurately detects abandoned luggage items in video surveillance feeds, robustly associates objects with their true owners, and suppresses false alarms caused by attended luggage, passing pedestrians, and momentary owner separation, without requiring high-power server GPUs?

### Objectives
1. Implement a low-latency edge object detector for persons and luggage classes.
2. Develop a multi-object tracking pipeline capable of maintaining identities across temporary occlusion.
3. Formulate a spatial-temporal association algorithm that reliably links bags to human owners.
4. Construct a deterministic finite state machine to manage abandonment timers, return recovery, and alert dispatch.
5. Benchmark runtime latency across native PyTorch and ONNX Runtime execution providers on edge CPU architectures.

---

## 3. Literature Review & Related Work

### 3.1 Background Subtraction & Foreground Segmentation
Early works (e.g., Stauffer & Grimson, 1999; Spagnolo et al., 2006) employed Gaussian Mixture Models (GMM) to segment static foreground regions. Dual-background models with differing learning rates were used to identify stationary objects. However, these methods fail when lighting shifts, shadows cast by pedestrians, or camera vibrations induce spurious foreground blobs, generating unacceptable false alarm rates in public spaces.

### 3.2 Deep Learning Object Detectors
The advent of CNNs produced high-accuracy detectors such as Faster R-CNN (Ren et al., 2015), SSD (Liu et al., 2016), and the YOLO family (Redmon et al., 2016; Jocher et al., 2023). While YOLOv8 achieves high mean Average Precision (mAP), standalone detection models possess no temporal memory; every stationary bag is perceived identically regardless of owner proximity or elapsed time.

### 3.3 Multi-Object Tracking & Data Association
Multi-Object Tracking (MOT) associates detections into continuous trajectories. SORT (Bewley et al., 2016) introduced Kalman filtering with Hungarian matching on high-confidence detections. DeepSORT (Wojke et al., 2017) added appearance feature embeddings via a deep CNN, but extraction latency prevents real-time CPU execution. ByteTrack (Zhang et al., 2022) demonstrated that associating low-confidence detections in a secondary bipartite matching stage recovers occluded objects without deep appearance descriptors, providing an optimal foundation for edge tracking.

---

## 4. Methodology & Mathematical Formulation

### 4.1 System Overview
The pipeline processes video frames sequentially through four stages:
$$\text{Input Frame } I_t \longrightarrow \text{Detection} \longrightarrow \text{Tracking} \longrightarrow \text{Association} \longrightarrow \text{State Machine} \longrightarrow \text{Alert Output}$$

### 4.2 Object Detection & Class Mapping
Ultralytics YOLOv8n processes frame $I_t$ with input size $W_{in} \times H_{in}$:
$$D_t = \text{YOLOv8n}(I_t, \theta_{conf}=0.15)$$
COCO classes are mapped to target classes:
$$C_{mapped} = \{0: \text{person}, 24: \text{bag}, 26: \text{bag}, 28: \text{bag}, 63: \text{bag}\}$$

### 4.3 Multi-Object Tracking via ByteTrack
Detections are partitioned into $D_{high}$ ($\text{conf} \ge \theta_{track}$) and $D_{low}$ ($0.05 \le \text{conf} < \theta_{track}$).
1. First-stage Hungarian association between active tracks $\mathcal{T}$ and $D_{high}$ using $1 - \text{IoU}$.
2. Second-stage Hungarian association between remaining tracks and $D_{low}$.
3. Unmatched high-confidence detections initialize new tracks with initial velocity $\mathbf{v} = [0, 0]^T$.

### 4.4 Spatial-Temporal Owner Association
For each pair of person track $p$ and bag track $b$, composite score $S(p, b)$ is computed:
$$S(p, b) = 0.45 \cdot S_{dist} + 0.35 \cdot S_{temp} + 0.20 \cdot S_{motion}$$
- **Edge Distance ($S_{dist}$):** Evaluates distance between box boundaries:
  $$S_{dist} = \max\left(0.0, 1.0 - \frac{d_{edge}(p, b)}{250.0}\right)$$
- **Temporal Persistence ($S_{temp}$):**
  $$S_{temp} = \min\left(1.0, \frac{\Delta t_{proximity}}{1.0\text{ s}}\right)$$
- **Velocity Cosine ($S_{motion}$):**
  $$S_{motion} = \frac{\mathbf{v}_p \cdot \mathbf{v}_b}{\|\mathbf{v}_p\| \|\mathbf{v}_b\| + \epsilon}$$

An owner is locked when $S(p, b) \ge 0.40$ continuously for $\ge 1.0\text{ s}$.

### 4.5 Temporal State Machine
The state machine implements 7 discrete operational states:
1. `UNKNOWN`: Transitory state upon track initiation.
2. `ATTENDED`: Owner distance $d \le 180\text{ px}$, displacement $\Delta d < 25\text{ px}$.
3. `OWNER_MOVING_AWAY`: $d > 120\text{ px}$, relative speed $> 1.5\text{ px/s}$.
4. `SEPARATED`: $d > 180\text{ px}$, displacement $\Delta d < 25\text{ px}$. Separation timer begins.
5. `SUSPECTED_ABANDONED`: Separation timer $\Delta t_{sep} \ge 7.5\text{ s}$.
6. `ABANDONED`: Separation timer $\Delta t_{sep} \ge 15.0\text{ s}$. Siren and visual alert triggered.
7. `RECOVERED`: Triggered if owner returns ($d \le 120\text{ px}$) or bag moves ($\Delta d \ge 25\text{ px}$).

---

## 5. Experimental Results & Discussion

### 5.1 Latency and Throughput Benchmarks
Experiments were conducted on an Intel x86-64 CPU (8 logical threads) running Windows 11.

- **PyTorch Native (640px):** $61.37\text{ ms}$ ($16.29\text{ FPS}$).
- **PyTorch Native (384px - Ultra Fast):** $24.20\text{ ms}$ ($41.30\text{ FPS}$).
- **ONNX Runtime CPU (640px):** $50.16\text{ ms}$ ($19.93\text{ FPS}$) — $18.3\%$ latency reduction over PyTorch.

### 5.2 Real-World Functional Validation
In physical camera tests with real backpacks and handbags:
- In attended scenarios where the owner stood beside the bag, the system maintained `ATTENDED` state and triggered zero false alarms over extended monitoring.
- In true abandonment tests, the system accurately transitioned through `SEPARATED` -> `SUSPECTED_ABANDONED` -> `ABANDONED`, dispatching dual hardware and browser sirens within $15.4\text{ seconds}$ of owner departure.
- When owners returned at $t=8\text{ seconds}$, the system transitioned to `RECOVERED` and reset timers cleanly.

---

## 6. Limitations & Future Work

### Limitations
1. Uncalibrated camera perspectives measure distance in pixels rather than metric ground distances.
2. Extremely dense pedestrian crowds can induce tracking identity swaps without visual Re-ID appearance features.
3. Pretrained COCO weights were utilized without custom fine-tuning on airport/station domain datasets.

### Future Work
1. Implement 4-point homography calibration for metric ground-plane distance estimation.
2. Fine-tune YOLOv8n on the full ABODA and PETS2006 benchmark datasets using sequence-level partitioning.
3. Deploy compiled TensorRT FP16/INT8 engines on dedicated NVIDIA Jetson Orin edge hardware.

---

## 7. Conclusion
This project successfully designed, implemented, and verified an automated, real-time edge surveillance pipeline for unattended object detection. By integrating anchor-free YOLOv8n detection, class-consistent ByteTrack tracking, multi-factor spatial-temporal ownership association, and a deterministic finite state machine, the system effectively overcomes the false alarm vulnerability of standard detectors while achieving 35 to 41 FPS on commodity edge CPU hardware.
