# Architectural Comparison: Object Detectors & Multi-Object Trackers

**Project Title:** Unattended / Abandoned Object Detection with Temporal Reasoning for Edge AI Surveillance  
**Repository Path:** `C:\Users\Aditya\.gemini\antigravity\scratch\abandoned_object_detection`  

---

## 1. Object Detection Architecture Comparison

To justify the selection of **YOLOv8 Nano (`yolov8n`)**, we compare it against alternative detector architectures suitable for edge deployment.

| Metric / Feature | YOLOv8n (Selected) | YOLOv5n | YOLOv9-Compact (t) | YOLOv10n | RT-DETR-Tiny | Faster R-CNN (ResNet-50) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Detection Paradigm** | Anchor-Free Single-Stage | Anchor-Based Single-Stage | Generalized GELAN | Dual-Assignment Anchor-Free | Vision Transformer (DETR) | Two-Stage Region Proposal |
| **Parameters (M)** | **3.16 M** | 1.90 M | 2.00 M | 2.30 M | 4.80 M | 41.50 M |
| **FLOPs (640px)** | **8.7 G** | 4.5 G | 7.7 G | 6.7 G | 9.0 G | 180.0 G |
| **mAP@0.5:0.95 (COCO)**| **37.3%** | 28.0% | 38.3% | 38.5% | 46.5% | 38.0% |
| **CPU Latency (Intel x86)** | **~24 ms (384px)** / 61 ms (640px) | ~20 ms (384px) | ~28 ms (384px) | ~25 ms (384px) | ~65 ms (384px) | ~350 ms (640px) |
| **Small Object Sensitivity**| High (PANet P3 Level) | Moderate | High (Programmable Gradient) | High | Very High (Self-Attention) | High |
| **NMS Dependency** | Requires NMS (IoU 0.45) | Requires NMS | Requires NMS | NMS-Free (Consistent Dual Match)| NMS-Free (Direct Set Match) | Requires RPN + Fast R-CNN NMS |
| **ONNX Export Ease** | Native & Stable (Opset 12) | Native | Requires custom ops | Supported | Heavy attention ops | Complex custom plugins |
| **TensorRT Compatibility**| Excellent (Native FP16/INT8) | Excellent | Good | Good | Requires TensorRT 8.6+ | Difficult (Custom ROIAlign) |
| **Edge Hardware Profile** | Raspberry Pi 5, Jetson Nano | Pi 4 / Pi 5 | Jetson Orin | Jetson Orin | Jetson Orin / Dedicated GPU | Cloud Server Only |

---

### In-Depth Justification for YOLOv8n

1. **Why not YOLOv5n?** While YOLOv5n has fewer FLOPs (4.5G vs 8.7G), its anchor-based design struggles with atypical aspect ratios (such as thin straps of handbags, flat laptop cases, or elongated duffel bags). YOLOv8n's anchor-free decoupled head achieves $+9.3\%\text{ mAP}$ over YOLOv5n with negligible CPU latency difference.
2. **Why not RT-DETR-Tiny?** Real-Time DEtection TRansformer (RT-DETR) achieves remarkable accuracy, but multi-scale self-attention incurs heavy memory bandwidth overhead on edge CPUs without Tensor Cores, causing latency to spike beyond $60\text{ ms}$ ($< 15\text{ FPS}$).
3. **Why not Faster R-CNN?** Two-stage proposal extraction requires $180\text{ GFLOPs}$, dropping CPU throughput to $2 - 3\text{ FPS}$, completely violating real-time surveillance constraints.

---

## 2. Multi-Object Tracking Comparison: ByteTrack vs. DeepSORT

The tracking module maintains temporal identities of moving pedestrians and stationary bags. We compare the implemented **ByteTrack** with the conventional **DeepSORT** algorithm.

| Feature / Metric | ByteTrack (Implemented in Code) | DeepSORT (Alternative Baseline) | SORT (Basic Baseline) |
| :--- | :---: | :---: | :---: |
| **Association Data** | Bounding box spatial IoU & motion kinematics | Deep CNN Re-ID appearance embeddings + IoU | Bounding box IoU only |
| **Low-Confidence Detection Handling** | **2-Stage Bipartite Matching:** Retains detections down to $\text{conf}=0.05$ to recover occluded objects | **Discarded:** Drops all boxes below threshold ($\text{conf} < 0.40$) | **Discarded:** Ignores low confidence |
| **Deep Feature Extractor** | **None** (Zero CNN feature extraction cost) | **Required:** Small CNN run on every detected box ($128\text{D}$ or $512\text{D}$ vector) | None |
| **Tracking Latency per Frame** | **$< 1.0\text{ ms}$** on CPU | **$25 - 50\text{ ms}$** on CPU ($+3\text{ ms}$ per box) | $< 0.5\text{ ms}$ on CPU |
| **Susceptibility to Motion Blur** | Low (Low-confidence matching recovers blurred boxes) | High (Blurred boxes fail Re-ID embedding) | High (Lost tracks) |
| **ID Switch Recovery after Long Occlusion** | Moderate ($\le 2.0\text{ s}$ via track buffer & linear motion) | High (Re-ID appearance matches after long exit) | Poor ($< 0.5\text{ s}$) |
| **Implementation Complexity** | Lightweight pure Python / SciPy / NumPy | Complex (PyTorch Re-ID model + feature cache) | Minimal |
| **Suitability for Edge AI CPU** | **Optimal** (Does not steal cycles from detector) | Poor (Re-ID network chokes CPU throughput) | Good (but low accuracy) |

---

### Why ByteTrack is Superior for Abandoned Object Surveillance

1. **Stationary Object Recovery:** When a luggage item sits on a floor, indoor fluorescent lighting, shadows, and pedestrian occlusions cause its detector confidence to fluctuate (e.g., dropping from $0.45$ to $0.18$). DeepSORT and standard SORT immediately drop tracks when confidence drops below $0.40$, causing the state machine's separation timer to reset. ByteTrack's second-stage association matches low-confidence detections ($0.05 \le \text{conf} < 0.20$), maintaining unbroken track IDs.
2. **Zero Computational Overhead:** DeepSORT requires cropping image patches and passing them through a secondary convolutional neural network (e.g., MobileNet or ResNet-based Re-ID). On a CPU running 15 people and 5 bags, executing 20 Re-ID passes adds $40\text{ ms}$ per frame, cutting pipeline frame rate in half. ByteTrack completes Hungarian matching in $< 1.0\text{ ms}$.
3. **Synergy with State Machine:** Since stationary luggage does not move, spatial overlap ($1 - \text{IoU}$) between consecutive frames remains near $1.0$, rendering appearance embeddings redundant for stationary bags.
