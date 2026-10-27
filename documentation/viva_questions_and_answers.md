# Comprehensive Viva Examination Question Bank (100 Questions & Technical Answers)

**Project Title:** Unattended / Abandoned Object Detection with Temporal Reasoning for Edge AI Surveillance  
**Repository Source:** `C:\Users\Aditya\.gemini\antigravity\scratch\abandoned_object_detection`  

---

## LEVEL 1: Basic Project Understanding (Questions 1 – 10)

### Q1: What is the primary objective of your project?
**Answer:** The objective is to design and deploy an automated, real-time video surveillance system that detects abandoned or unattended luggage (backpacks, suitcases, handbags, laptops) in monitored public areas. Unlike naive object detectors that raise an alert whenever a stationary bag is visible, our system incorporates multi-object tracking, spatial-temporal owner association, and a deterministic finite state machine to raise alerts **only** when an object is confirmed stationary, its associated owner has moved beyond a safety perimeter, and the separation has persisted beyond a defined time threshold ($15.0\text{ s}$).

### Q2: What is the difference between an "attended", "unattended", and "abandoned" object in your system?
**Answer:**
- **Attended:** A stationary or moving bag where a detected human owner is within the proximity threshold ($d \le 180\text{ px}$) and kinematic correlation is maintained.
- **Unattended / Separated:** An object that is stationary ($< 25\text{ px}$ displacement over $3.0\text{ s}$) whose owner has separated beyond the threshold ($d > 180\text{ px}$) or is absent, and the separation duration $\Delta t_{sep} < 15.0\text{ s}$.
- **Abandoned:** An unattended stationary object whose separation duration has met or exceeded the temporal threshold ($\Delta t_{sep} \ge 15.0\text{ s}$). An active security alert is dispatched.

### Q3: What are the main functional components of your pipeline?
**Answer:**
1. **Object Detection:** Ultralytics YOLOv8n detecting persons and luggage classes.
2. **Multi-Object Tracking:** Two-stage ByteTrack maintaining consistent track IDs across frames and computing velocity vectors.
3. **Spatial-Temporal Association:** Bipartite matching between persons and bags using Euclidean proximity, temporal persistence, and velocity cosine similarity.
4. **Temporal State Machine:** A 7-state deterministic automaton tracking state transitions, managing timers, handling pedestrian occlusions, and enforcing hysteresis upon owner return.
5. **UI & Alert Dispatch:** Streamlit dashboard with dual audio sirens (Windows `winsound` hardware beeper and browser Web Audio API) and real-time telemetry.

### Q4: Why can’t a simple object detector (like YOLO alone) solve the abandoned luggage problem?
**Answer:** A pure object detector operates frame-by-frame without temporal memory. It only outputs bounding boxes and class probabilities. If a passenger places their bag on the floor while waiting at a bus stop, YOLO detects both the person and the bag. A detector-only system would either flag every stationary bag (generating hundreds of false alarms per hour) or ignore all bags. Detection cannot distinguish between an owner standing next to their bag and an owner who walked away.

### Q5: What programming language, frameworks, and versions did you use?
**Answer:** Python 3.11.9, PyTorch 2.14.1 (CPU), Ultralytics 8.3.179 (YOLOv8), OpenCV 4.11.0, NumPy 1.26.4, SciPy 1.15.2, Pandas 2.2.3, PyArrow 17.0.0, and Streamlit 1.66.0.

### Q6: What hardware did your system execute and test on?
**Answer:** Tested on an Intel x86-64 8-thread CPU workstation running Windows 11. The pipeline is architected for Edge AI devices such as NVIDIA Jetson Orin/Nano or Raspberry Pi 5. On our Intel CPU, inference runs at $\sim 34 - 41\text{ FPS}$ at $384\text{px}$ resolution and $\sim 16 - 24\text{ FPS}$ at $640\text{px}$.

### Q7: What input sources does your application accept?
**Answer:** (1) Local benchmark/synthetic video files (`.mp4`), (2) User-uploaded video files, (3) USB / Laptop webcams via OpenCV index `0`, (4) Virtual phone cameras (Iriun Webcam / DroidCam via index `1`), and (5) IP network camera streams via RTSP/HTTP URL.

### Q8: What happens in your user interface when an abandoned object is detected?
**Answer:** (1) A high-contrast red animated warning banner flashes (`🚨 SECURITY ALERT`), (2) a dual audible alert triggers: Windows OS hardware beeps via `winsound.Beep` and browser Web Audio API chimes, (3) a red bounding box and countdown timer are drawn over the bag, and (4) an immutable event is logged into the tabular event log.

### Q9: What are the primary failure modes of real-world surveillance systems in this domain?
**Answer:** False positives caused by standing owners occluded by crowds, dynamic lighting/shadows causing bag track fragmentation, abandoned bags picked up by non-owners (theft), and crowded scenes where Euclidean distance links the bag to an unrelated bystander.

### Q10: How does your system prevent false alarms when an owner temporarily ties their shoe or stands slightly further away?
**Answer:** By combining a spatial buffer ($180\text{ px}$ separation threshold) with a temporal delay threshold ($15.0\text{ s}$). If the owner stays within $180\text{ px}$, the bag remains `ATTENDED`. If they step further, the state becomes `SEPARATED` or `SUSPECTED_ABANDONED`, but no alarm fires unless the separation persists continuously for $15.0\text{ seconds}$.

---

## LEVEL 2: Computer Vision & Deep Learning Fundamentals (Questions 11 – 20)

### Q11: Explain Intersection-over-Union (IoU) and its mathematical formula.
**Answer:** IoU measures the spatial overlap between two bounding boxes $A$ and $B$:
$$\text{IoU}(A, B) = \frac{\text{Area}(A \cap B)}{\text{Area}(A \cup B)} = \frac{\text{Area}(A \cap B)}{\text{Area}(A) + \text{Area}(B) - \text{Area}(A \cap B)}$$
It ranges from $0.0$ (no overlap) to $1.0$ (identical boxes). In ByteTrack, cost matrix entries are $1 - \text{IoU}$.

### Q12: What is Non-Maximum Suppression (NMS), and why is it necessary?
**Answer:** Dense object detectors generate multiple overlapping anchor-free candidate boxes for a single physical object. NMS sorts boxes by confidence score, selects the highest-scoring box, and discards all remaining boxes having an $\text{IoU} > \text{threshold}$ (typically $0.45$). This ensures one bounding box per object.

### Q13: What is the difference between anchor-based and anchor-free object detectors?
**Answer:** Anchor-based detectors (e.g., YOLOv3/v4/v5, Faster R-CNN) evaluate predefined bounding box shapes (anchors) at each grid cell and predict offsets. Anchor-free detectors (e.g., YOLOv8, CenterNet, FCOS) directly predict the object center point, distance to 4 box boundaries (top, bottom, left, right), and class probability, eliminating manual anchor clustering and improving generalizability on atypical aspect ratios.

### Q14: Explain the difference between Precision and Recall.
**Answer:**
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
Precision measures how many detected objects/alerts were truly correct (minimizing false alarms). Recall measures how many actual objects/alerts were successfully detected (minimizing missed security threats).

### Q15: What is Mean Average Precision (mAP)?
**Answer:** mAP is the mean of Average Precision (AP) across all target classes. AP is the area under the interpolated Precision-Recall curve. $\text{mAP@0.5}$ calculates AP at an IoU matching threshold of $0.50$. $\text{mAP@[0.5:0.95]}$ averages AP across IoU thresholds from $0.50$ to $0.95$ in increments of $0.05$.

### Q16: Why is mAP alone an inadequate metric for evaluating an abandoned object detection system?
**Answer:** mAP only evaluates whether bounding boxes overlap with ground truth in individual frames. It evaluates spatial localization, not temporal reasoning. A system with $0.95\text{ mAP}$ can fail completely in surveillance if its state machine triggers an alert every 2 seconds or crashes upon pedestrian occlusion. Event-level metrics (Event Precision, Event Recall, False Alarms per Hour, and Time-to-Alert) are essential.

### Q17: What is the difference between latency and throughput (FPS)?
**Answer:**
- **Latency (ms):** The end-to-end elapsed time required to process a single frame from capture to visualization ($\Delta t = t_{end} - t_{start}$).
- **Throughput (FPS):** The number of frames processed per second ($\text{FPS} = 1000 / \text{latency\_ms}$).
In edge streaming, low latency ($< 50\text{ ms}$) guarantees real-time responsiveness without accumulating frame buffer lag.

### Q18: What is the difference between one-stage and two-stage object detectors?
**Answer:** Two-stage detectors (e.g., Faster R-CNN) first use a Region Proposal Network (RPN) to generate candidate regions, then classify and refine them in a second stage (high accuracy, high latency: $100 - 200\text{ ms}$). One-stage detectors (e.g., YOLO, SSD) predict bounding boxes and class scores directly in a single forward pass through the convolutional backbone, achieving real-time rates ($15 - 60+\text{ FPS}$).

### Q19: What is Feature Pyramid Network (FPN) and Path Aggregation Network (PANet)?
**Answer:** FPN constructs top-down pathways to propagate high-level semantic features down to low-level spatial features. PANet extends FPN by adding a bottom-up path augmentation that injects low-level precise localization signals directly into high-level layers, significantly enhancing small object detection (such as small bags or handbags).

### Q20: What is the Hungarian (Munkres) Algorithm, and where is it used in your code?
**Answer:** The Hungarian algorithm solves the linear sum assignment problem for bipartite graph matching in $\mathcal{O}(N^3)$ polynomial time. In our code, it is invoked via `scipy.optimize.linear_sum_assignment` in `src/tracking/byte_tracker.py` to match tracks to detections based on IoU distance, and in `src/association/owner_associator.py` to associate person tracks with bag tracks.

---

## LEVEL 3: Detection Models & Architecture (Questions 21 – 30)

### Q21: Exactly which YOLO architecture does your implementation use?
**Answer:** YOLOv8 Nano (`yolov8n`), released by Ultralytics in January 2023.

### Q22: Why did you choose YOLOv8 Nano over larger models like YOLOv8x or YOLOv8m?
**Answer:** Edge AI surveillance mandates real-time throughput on constrained hardware without costly dedicated GPUs. YOLOv8n has only $3.2\text{M}$ parameters and $8.7\text{ GFLOPs}$, operating at $> 35\text{ FPS}$ on a CPU at $384\text{px}$, whereas YOLOv8x ($68.2\text{M}$ parameters, $257\text{ GFLOPs}$) would drop to $1 - 3\text{ FPS}$ on a CPU, inducing severe pipeline lag.

### Q23: What is the C2f module in YOLOv8?
**Answer:** The C2f (Cross-Stage Partial with 2 convolutions) module is an evolution of YOLOv5's C3 module, inspired by ELAN (Efficient Layer Aggregation Networks). It splits feature maps and routes them through multiple parallel bottleneck branches with dense residual shortcuts, maximizing gradient flow while maintaining lightweight computation.

### Q24: What loss functions are used in YOLOv8?
**Answer:** YOLOv8 uses a combination of:
1. **Task-Aligned Focal Loss / BCE:** For classification.
2. **Distribution Focal Loss (DFL):** For box regression modeling coordinate uncertainty as a continuous distribution.
3. **CIoU (Complete IoU) Loss:** Evaluates overlap, center point distance, and aspect ratio consistency.

### Q25: What is the input resolution of your YOLO detector in this project?
**Answer:** In `configs/system.yaml` and `configs/detector.yaml`, the default is $640\times 640$ pixels. In our interactive Streamlit application (`app.py`), we added an adaptive performance slider with options:
- *Ultra Fast:* $384\times 384$ ($> 35\text{ FPS}$ on CPU).
- *Balanced:* $480\times 480$ ($25 - 30\text{ FPS}$).
- *Max Detail:* $640\times 640$ ($15 - 20\text{ FPS}$).

### Q26: What classes in the COCO dataset correspond to luggage, and how are they mapped?
**Answer:**
- COCO Class 0: `person` -> Target Class 0 (`person`)
- COCO Class 24: `backpack` -> Target Class 1 (`bag`)
- COCO Class 26: `handbag` -> Target Class 1 (`bag`)
- COCO Class 28: `suitcase` -> Target Class 1 (`bag`)
- COCO Class 63: `laptop` -> Target Class 1 (`bag`) (included to catch unattended electronics).

### Q27: Did you train or fine-tune YOLOv8n in your project?
**Answer:** *(Honest Viva Answer)* No. We utilized official COCO-pretrained weights (`yolov8n.pt`) as a zero-shot detector. The core research and engineering contributions of our project reside in the multi-object tracking, spatial-temporal owner association algorithm, and deterministic finite state machine built on top of the detector. Training scripts (`scripts/train_detector.py`) and dataset preparation tools (`scripts/prepare_aboda.py`) were created, but no model was fine-tuned on custom datasets due to hardware constraints.

### Q28: How does your detector handle small bags at long distances?
**Answer:** Small bags produce low feature map activations and often receive lower confidence scores ($0.10 - 0.25$). We solved this by lowering our detector confidence slider to $0.15$ (down to $0.05$ supported) and modifying ByteTrack to initiate bag tracks at $\min(\text{track\_thresh}, 0.20)$, ensuring small or low-contrast bags are tracked immediately.

### Q29: What is the size of the `yolov8n.pt` weight file?
**Answer:** $6.55\text{ MB}$ ($6,549,796\text{ bytes}$).

### Q30: What is the format of YOLO bounding box annotations?
**Answer:** Normalized coordinates $[x_{center}, y_{center}, \text{width}, \text{height}]$ where all values are normalized between $0.0$ and $1.0$ relative to image width and height:
$$x_{center} = \frac{x_{min} + x_{max}}{2 \cdot W}, \quad w = \frac{x_{max} - x_{min}}{W}$$

---

## LEVEL 4: Datasets, Annotations & Preprocessing (Questions 31 – 40)

### Q31: What is the ABODA dataset?
**Answer:** ABODA (Abandoned Object Dataset in Airport) is a benchmark dataset created by the University of Central Florida containing 11 video sequences recorded in airport and indoor public areas. It contains ground truth annotations for abandoned items, stationary objects, and owner separation events.

### Q32: Is the ABODA dataset downloaded in your repository?
**Answer:** No. The directory `data/raw/aboda/` contains only `.gitkeep`. The script `scripts/prepare_aboda.py` was created to parse ABODA XML annotations and extract frames into YOLO format, but the raw video sequences were not downloaded into the repository.

### Q33: What is the PETS2006 dataset?
**Answer:** The PETS2006 (Performance Evaluation of Tracking and Surveillance) benchmark comprises multi-sensor CCTV sequences recorded in a train station concourse with varying difficulty levels (S1 to S7) illustrating abandoned luggage, owner departure, and high pedestrian occlusion.

### Q34: What data was actually used to test and verify your code?
**Answer:** We used two sources:
1. Three staged synthetic campus evaluation scenarios generated programmatically by `scripts/create_local_dataset.py` (`campus_true_abandoned.mp4`, `campus_owner_returns.mp4`, `campus_normal_attended.mp4`).
2. Live real-world video feeds captured from webcams and phone cameras connected via Iriun Webcam and IP Webcam.

### Q35: What is temporal data leakage, and how must video datasets be split?
**Answer:** Temporal data leakage occurs when consecutive video frames (e.g., frame $t$ and frame $t+1$) from the same video are randomly distributed into train and test sets. Because consecutive frames are nearly identical, the model memorizes background pixels and achieves artificially inflated test scores. To prevent leakage, splitting **must be sequence-level**: entire video clips or distinct recording sessions are assigned to train, validation, or test sets.

### Q36: How does `scripts/prepare_aboda.py` split video sequences?
**Answer:** It performs sequence-level partitioning, assigning distinct video sequence IDs to train ($70\%$), validation ($15\%$), and test ($15\%$) sets, ensuring no sequence shares frames across splits.

### Q37: What frame stride was configured for dataset extraction?
**Answer:** Frame stride of 5 (`sample_rate = 5`), meaning 1 frame out of every 5 is extracted ($\sim 6\text{ frames/second}$ from a $30\text{ FPS}$ video) to reduce redundant nearly identical frames while preserving motion dynamics.

### Q38: What data augmentations would you apply when fine-tuning for abandoned luggage?
**Answer:** Mosaic augmentation, MixUp, random perspective, HSV color jitter (to handle indoor fluorescent vs daylight lighting changes), horizontal flip, and random affine scaling.

### Q39: Why are synthetic geometric sprites inadequate for testing real pretrained YOLO models?
**Answer:** As observed in `results/metrics/system_metrics.json`, synthetic videos rendered with simple colored rectangles lack natural textures, gradients, and edge features learned by YOLO from COCO. The pretrained neural network detects zero objects on geometric sprites, causing false negatives ($FN=1, TP=0$).

### Q40: How would you collect and label a custom campus dataset for this project?
**Answer:** Following `docs/local_dataset_collection.md`: record 1080p CCTV footage from fixed cameras across library lobbies, cafeteria, and building entrances. Use CVAT or LabelImg to annotate two classes (`person` and `bag`), record time-stamped ground truth logs of when owners leave and return, and split by camera location.

---

## LEVEL 5: Multi-Object Tracking & ByteTrack (Questions 41 – 50)

### Q41: What tracking algorithm did you implement, and where is it located?
**Answer:** ByteTrack, implemented in `src/tracking/byte_tracker.py`.

### Q42: How does ByteTrack differ from SORT and DeepSORT?
**Answer:**
- **SORT:** Uses simple Kalman filtering and Hungarian matching on high-confidence detections only. Discards all low-confidence boxes.
- **DeepSORT:** Adds a deep appearance descriptor (Re-ID network) for visual cosine matching. This increases latency significantly ($+20 - 40\text{ ms}$ on CPU).
- **ByteTrack:** Retains low-confidence detections ($0.05 \le \text{conf} < 0.35$) instead of discarding them. It performs a 2-stage association: first matching high-confidence boxes, then matching remaining tracks against low-confidence boxes. This recovers tracks through occlusion and motion blur with zero deep Re-ID compute overhead.

### Q43: What is the state representation of a track in your system?
**Answer:** In `src/tracking/track_state.py`, each track stores bounding box $[x_1, y_1, x_2, y_2]$, center $[x_c, y_c]$, width $w$, height $h$, instantaneous velocity vector $[v_x, v_y]$, detection confidence, observation history, and `time_since_update`.

### Q44: How are velocity vectors calculated for moving objects?
**Answer:**
$$\mathbf{v}(t) = \frac{\mathbf{p}(t) - \mathbf{p}(t - \Delta t)}{\Delta t}$$
where $\mathbf{p}(t) = [x_c, y_c]$ is the bounding box center and $\Delta t$ is elapsed timestamp seconds. In code, velocity is smoothed over the last 5 observations.

### Q45: What is `track_buffer`, and what value is used in your system?
**Answer:** `track_buffer` is the maximum number of consecutive missed frames a lost track is retained before being permanently deleted. In `configs/system.yaml`, `track_buffer = 60` frames ($\sim 2.0\text{ seconds}$ at $30\text{ FPS}$).

### Q46: What happens when an object is occluded for 10 frames in ByteTrack?
**Answer:** Its `time_since_update` increments by 10. The track is not deleted because $10 \le 60$. When the object re-emerges, even with a low confidence score ($0.15$), Stage 2 of ByteTrack associates the low-confidence detection with the existing track, preserving the same track ID.

### Q47: What is an ID switch, and why is it dangerous in abandoned object detection?
**Answer:** An ID switch occurs when the tracker swaps identities between two nearby objects or drops a track and creates a new ID for the same object. If a stationary bag's track ID swaps from B01 to B02, the temporal separation timer resets to zero, delaying or preventing the abandonment alert.

### Q48: How does your implementation enforce class consistency in tracking?
**Answer:** In `src/tracking/byte_tracker.py` (lines 165–205), tracking association is executed independently per class (`active_classes = {0, 1}`). A person detection can never match a bag track, and a bag detection can never match a person track.

### Q49: What optimization did you make to ByteTrack for bag detection?
**Answer:** Standard ByteTrack requires $\text{conf} \ge 0.35$ to initiate a new track. Because bags on floors often score between $0.15$ and $0.30$, we set an adaptive threshold `effective_thresh = min(self.track_thresh, 0.20)` for `class_id == 1`. This allows bags to initiate tracks immediately.

### Q50: What is HOTA and IDF1?
**Answer:**
- **IDF1:** Identification F1-score measures how well the tracker maintains consistent identities over time, computing the ratio of correctly identified detections over average ground truth and computed detections.
- **HOTA (Higher Order Tracking Accuracy):** Balances detection accuracy ($\text{DetA}$), association accuracy ($\text{AssA}$), and localization accuracy ($\text{LocA}$) via $\text{HOTA} = \sqrt{\text{DetA} \cdot \text{AssA}}$.

---

## LEVEL 6: Spatial-Temporal Owner Association (Questions 51 – 60)

### Q51: How does your system associate a person with a bag?
**Answer:** Via `src/association/owner_associator.py`. For every active bag, candidate persons within $d_{max} = 250\text{ px}$ are evaluated using a multi-factor composite score:
$$S = w_{dist} \cdot S_{dist} + w_{temp} \cdot S_{temp} + w_{motion} \cdot S_{motion}$$
If $S \ge 0.40$ continuously for $t \ge 1.0\text{ second}$, the person is confirmed as the owner.

### Q52: What are the three weights used in composite association?
**Answer:** $w_{dist} = 0.45$, $w_{temp} = 0.35$, $w_{motion} = 0.20$. Proximity carries the highest weight, followed by temporal proximity duration, and finally directional motion correlation.

### Q53: How is spatial distance $S_{dist}$ computed between a person and a bag?
**Answer:** In `src/association/spatial_metrics.py`, we compute bounding-box edge-to-edge distance $d_{edge}$:
- If bounding boxes overlap, $d_{edge} = 0.0$ and $S_{dist} = 1.0$.
- If separated:
  $$d_{edge} = \sqrt{\Delta x^2 + \Delta y^2}, \quad S_{dist} = \max\left(0.0, 1.0 - \frac{d_{edge}}{d_{max}}\right)$$

### Q54: Why use bounding-box edge distance instead of centroid Euclidean distance?
**Answer:** A person is tall ($h \gg w$) and their center point is around their chest/torso, while a carried bag is near their feet or hands. Centroid distance can be $120\text{ px}$ even when the person is holding the bag. Edge-to-edge distance evaluates physical boundary contact ($0\text{ px}$ when held or touching), providing superior physical realism.

### Q55: How is kinematic correlation $S_{motion}$ formulated?
**Answer:** Using cosine similarity between person velocity vector $\mathbf{v}_p$ and bag velocity vector $\mathbf{v}_b$:
$$S_{motion} = \frac{\mathbf{v}_p \cdot \mathbf{v}_b}{\|\mathbf{v}_p\| \|\mathbf{v}_b\| + \epsilon}$$
If both person and bag move synchronously in the same direction, $S_{motion} \approx 1.0$. If they move in opposite directions, $S_{motion} \le 0.0$.

### Q56: What happens if a stranger walks past an unattended bag?
**Answer:** A stranger walking past is close for only $0.3 - 0.6\text{ seconds}$, so their $S_{temp}$ remains near $0.0$, and their composite score fails to surpass the threshold ($0.40$). Furthermore, ownership is locked to the original confirmed owner and cannot be reassigned to a passing pedestrian.

### Q57: What if an owner is not visible when the bag first appears?
**Answer:** If a bag appears with no person nearby ($d > 180\text{ px}$), the state machine classifies it as `UNKNOWN`, assigns `owner_id = None`, and immediately transitions it to `SEPARATED`. The separation timer begins right away.

### Q58: Can an owner be re-associated if they return?
**Answer:** Yes. If the confirmed owner returns within $d_{return} \le 120\text{ px}$, the state transitions to `RECOVERED` and then back to `ATTENDED`, resetting the abandonment timer.

### Q59: What is candidate history length?
**Answer:** `candidate_history_len = 45` frames ($\sim 1.5\text{ s}$). It maintains a rolling history buffer of candidate scores to smooth out instantaneous detection jitter.

### Q60: What if two people are walking together next to the bag?
**Answer:** The bipartite matching assigns the person with the highest composite score (closest edge distance and highest motion correlation) as the primary owner.

---

## LEVEL 7: Temporal Reasoning & State Machine (Questions 61 – 70)

### Q61: What are the 7 states in your finite state machine?
**Answer:** (1) `UNKNOWN`, (2) `ATTENDED`, (3) `OWNER_MOVING_AWAY`, (4) `SEPARATED`, (5) `SUSPECTED_ABANDONED`, (6) `ABANDONED`, (7) `RECOVERED`.

### Q62: What exact condition triggers the transition from `ATTENDED` to `SEPARATED`?
**Answer:** When the current distance between the bag and confirmed owner exceeds $d_{sep} = 180.0\text{ px}$, or if the owner leaves the field of view while the bag remains stationary.

### Q63: What condition triggers `SUSPECTED_ABANDONED`?
**Answer:** When the separation duration $\Delta t_{sep} \ge 0.5 \cdot T_{abandon} = 7.5\text{ seconds}$ (half of the $15.0\text{ s}$ threshold).

### Q64: What condition triggers `ABANDONED`?
**Answer:** When $\Delta t_{sep} \ge T_{abandon} = 15.0\text{ seconds}$ while the bag has remained stationary ($< 25\text{ px}$ displacement over $3.0\text{ s}$).

### Q65: How is object stationarity verified?
**Answer:** In `src/tracking/track_state.py`, `calculate_displacement(window_seconds=3.0)` computes the Euclidean distance between the bag's current center and its center $3.0\text{ seconds}$ prior:
$$\Delta d = \|\mathbf{p}(t) - \mathbf{p}(t - 3.0\text{s})\|$$
If $\Delta d < 25.0\text{ px}$, the bag is verified stationary.

### Q66: What happens if an abandoned bag is picked up and carried away?
**Answer:** Its displacement exceeds $25.0\text{ px}$ in the 3.0s window. `is_stationary` evaluates to `False`. The state machine immediately cancels the alert, transitions to `RECOVERED` with reason `"Bag moved/displaced"`, and clears the separation timer.

### Q67: What is hysteresis, and why are separation and return thresholds different?
**Answer:**
- Separation threshold: $d_{sep} = 180\text{ px}$.
- Return threshold: $d_{ret} = 120\text{ px}$.
Hysteresis creates a $60\text{ px}$ buffer zone. If both were identical ($180\text{ px}$), an owner pacing back and forth near $180\text{ px}$ would cause rapid flickering between `ATTENDED` and `SEPARATED`. Hysteresis stabilizes the state machine.

### Q68: How does your state machine handle pedestrian occlusion?
**Answer:** If a bag is temporarily blocked by pedestrians, its track is missing from current frame detections. The state machine activates `is_occluded = True` and starts `occlusion_start_time`. If the bag reappears within $3.0\text{ s}$ (`occlusion_grace_period`), the timer and state are preserved continuously. If missing $> 3.0\text{ s}$, the track is cleaned up.

### Q69: What is the alert cooldown mechanism?
**Answer:** Once an `ABANDONED` alert is fired, `last_alert_time` is updated. No secondary alert sound or notification is issued for the same bag until `alert_cooldown_seconds = 30.0s` elapses, preventing alert flooding.

### Q70: Is your temporal reasoning rule-based or deep learning-based?
**Answer:** It is a deterministic rule-based finite state machine. Although deep temporal models (`TemporalLSTMClassifier` and `TemporalMLPClassifier`) were defined in `src/temporal/temporal_model.py`, the state machine is active (`use_temporal_model: false`) because deterministic state machines are $100\%$ explainable, have $\mathcal{O}(1)$ runtime complexity, require zero training data, and guarantee predictable safety behavior.

---

## LEVEL 8: False Positives, Edge Cases & Failure Analysis (Questions 71 – 80)

### Q71: Edge Case: An owner sits on a bench 3 meters from their bag. Does your system alarm?
**Answer:** If 3 meters corresponds to $> 180\text{ px}$ in camera view, the system will enter `SEPARATED` and alarm after 15 seconds. In production CCTV, this is calibrated via camera perspective transformation (homography) so the threshold is measured in real meters ($2.5 - 3.0\text{ m}$) rather than uncalibrated pixels.

### Q72: Edge Case: The owner turns around and their back occludes the bag. What happens?
**Answer:** Occlusion tolerance holds the bag state for $3.0\text{ seconds}$. Since the owner is still within $180\text{ px}$, the bag remains `ATTENDED`. When the bag reappears, tracking resumes with the same ID.

### Q73: Edge Case: A thief steals the bag. What does the system do?
**Answer:** As the thief picks up and moves the bag, displacement surpasses $25\text{ px}$, transitioning the bag to `RECOVERED` with reason `"Bag moved/displaced"`. In advanced setups, this triggers a `THEFT_ALERT` if the moving person is not the confirmed owner.

### Q74: Failure Mode: Lightning flash or sudden shadow changes.
**Answer:** Severe lighting shifts can alter bounding box coordinates by a few pixels, but our stationary threshold ($25\text{ px}$ over 3s) absorbs normal noise. Extreme lighting can cause temporary miss-detections, which are protected by the 3-second occlusion grace period.

### Q75: Edge Case: Highly crowded scenes with 50+ people.
**Answer:** Crowded scenes increase the probability of ID switches and occlusions lasting longer than 3 seconds. ByteTrack's low-confidence association mitigates this, but extreme crowd density requires Re-ID appearance embeddings and homographic ground-plane tracking.

### Q76: What is "False Alarms Per Hour" (FAR/hr)?
**Answer:**
$$\text{FAR/hr} = \frac{\text{Total False Positive Abandonment Events}}{\text{Total Monitoring Duration in Hours}}$$
In critical infrastructure surveillance (airports, metros), FAR must be $< 1.0\text{ / hour}$ to prevent operator alarm fatigue.

### Q77: Why did your app crash with a PyArrow error previously, and how did you resolve it?
**Answer:** Streamlit's `st.dataframe` serializes DataFrames to Apache Arrow IPC format. The installed PyArrow version required NumPy 2.0+, whereas PyTorch 2.14 required NumPy 1.26.4, throwing an unhandled `ImportError` when an alert logged an event. We fixed this by pinning `pyarrow==17.0.0` (fully compatible with NumPy 1.26.4) and wrapping table updates with an automatic fallback to HTML table rendering.

### Q78: Why did PyTorch fail with WinError 1114 on Windows, and how did you resolve it?
**Answer:** On Windows, importing `pandas` or `pyarrow` before `torch` causes the OS dynamic linker to initialize shared C++ runtime DLLs with flags that conflict with PyTorch's `c10.dll`. When PyTorch is imported later, `_load_dll_libraries()` fails with `WinError 1114`. We resolved this by placing `import torch` at the very top of `app.py`, ensuring `c10.dll` initializes first.

### Q79: What is Time-to-Alert (TTA)?
**Answer:** The elapsed time between the true ground truth moment of owner departure ($t_{depart}$) and the timestamp when the system emits the alert ($t_{alert}$):
$$\text{TTA} = t_{alert} - t_{depart}$$
In our system, nominal $\text{TTA} = T_{abandon} \approx 15.0\text{ seconds}$.

### Q80: How does camera vibration (e.g., wind on pole-mounted CCTV) affect stationarity?
**Answer:** Camera shake causes global pixel shifts across the entire frame. If camera shake exceeds $25\text{ px}$, a stationary bag might falsely appear moving. This is solved in production via background motion compensation (affine image stabilization using ORB/optical flow).

---

## LEVEL 9: Edge AI, ONNX, Optimization & Benchmarking (Questions 81 – 90)

### Q81: What is ONNX, and why did you export your model to ONNX?
**Answer:** Open Neural Network Exchange (ONNX) is an open format representing deep learning models as computation graphs. Exporting YOLOv8n to ONNX (`models/onnx/yolov8n.onnx`) decouples the model from the heavy PyTorch training framework, enabling deployment across highly optimized cross-platform runtimes like ONNX Runtime, OpenVINO, and TensorRT.

### Q82: What is the size difference between the PyTorch and ONNX models?
**Answer:** PyTorch weights `yolov8n.pt`: $6.25\text{ MB}$. ONNX model `yolov8n.onnx`: $12.26\text{ MB}$ (uncompressed explicit computation graph with constant tensors).

### Q83: Compare the measured latency between PyTorch and ONNX Runtime on your system.
**Answer:** From `results/metrics/benchmark_metrics.json`:
- **PyTorch CPU:** Mean latency $61.37\text{ ms}$ ($16.29\text{ FPS}$).
- **ONNX Runtime CPU:** Mean latency $50.16\text{ ms}$ ($19.93\text{ FPS}$).
ONNX Runtime achieved an $18.3\%$ reduction in latency on the exact same CPU.

### Q84: What is TensorRT, and how does it optimize inference?
**Answer:** NVIDIA TensorRT is an SDK for high-performance deep learning inference on NVIDIA GPUs. It optimizes models via:
1. **Layer & Tensor Fusion:** Combines Conv + BatchNorm + ReLU into a single GPU kernel.
2. **Kernel Auto-Tuning:** Selects the optimal CUDA kernel algorithm for the specific GPU architecture.
3. **Precision Calibration:** Quantizes weights to FP16 or INT8 using Tensor Cores.
4. **Dynamic Tensor Memory Management:** Reuses GPU memory pools to eliminate allocation overhead.

### Q85: Why is TensorRT not active on your current test system?
**Answer:** TensorRT requires an NVIDIA GPU with CUDA compute capability $\ge 5.0$, CUDA Toolkit, and NVIDIA TensorRT C++ libraries. The host machine is an Intel CPU-only system (`torch.cuda.is_available() == False`). While `scripts/build_tensorrt.py` is fully implemented, the `.engine` cannot be compiled without target NVIDIA hardware.

### Q86: What are FP32, FP16, and INT8 precision?
**Answer:**
- **FP32 (Single Precision):** 32-bit floating point (1 sign, 8 exponent, 23 mantissa). Highest accuracy, standard baseline.
- **FP16 (Half Precision):** 16-bit floating point (1 sign, 5 exponent, 10 mantissa). Reduces memory footprint by $50\%$ and doubles throughput on modern Tensor Cores with negligible accuracy loss ($< 0.5\%\text{ mAP}$).
- **INT8 (8-bit Integer):** 8-bit quantized integer representation. Reduces memory by $75\%$ and provides up to $3\times - 4\times$ speedup, but requires calibration to prevent dynamic range quantization loss.

### Q87: Why did downscaling incoming webcam frames from 1080p to 640px increase FPS in Streamlit?
**Answer:** Streaming uncompressed $1920\times 1080\times 3$ RGB frames over Streamlit's WebSocket connection consumes $\sim 6.2\text{ MB}$ per frame ($\sim 62\text{ MB/s}$ at 10 FPS), choking the WebSocket buffer and browser DOM. Downscaling to $640\text{px}$ reduces payload by $80\%$, dropping transfer latency and accelerating CPU OpenCV rendering.

### Q88: Why did you throttle telemetry metrics to update every 3 frames?
**Answer:** Updating 7 independent Streamlit metric widgets on every frame at 30 FPS dispatched 210 WebSocket RPC messages per second. Throttling metrics to update once every 3 frames reduced messaging overhead by $67\%$ while keeping human-readable telemetry perfectly responsive.

### Q89: How can power consumption be monitored on edge devices?
**Answer:** In `src/utils/power_monitor.py`:
- On NVIDIA Jetson: Reading hardware power rails via the `jtop` API (INA3221 voltage/current sensors).
- On NVIDIA Desktop/Server: Querying power draw in Watts via `pynvml` (`nvmlDeviceGetPowerUsage`).
- On Intel/AMD CPUs: Reading RAPL (Running Average Power Limit) energy MSRs via `psutil` or Linux `/sys/class/powercap/`.

### Q90: What is the computational complexity of your entire pipeline per frame?
**Answer:**
- Detection (YOLOv8n): $\mathcal{O}(W \cdot H)$ convolution operations $\approx 8.7\text{ GFLOPs}$ at 640px.
- Tracking (ByteTrack): Hungarian matching is $\mathcal{O}(N^3)$ where $N \le 20$ objects $\approx < 0.1\text{ ms}$.
- Association: Bipartite matching on $P \times B$ pairs $\approx < 0.05\text{ ms}$.
- State Machine: Hash-map state lookup and arithmetic updates $\approx \mathcal{O}(B) \approx < 0.01\text{ ms}$.
The neural network detection accounts for $> 95\%$ of total compute time.

---

## LEVEL 10: Research Novelty, Limitations & Examiner Defense (Questions 91 – 100)

### Q91: What is the novelty of your project?
**Answer:** Most published works either treat unattended object detection as a background subtraction task (which fails completely with moving crowds and camera jitter) or apply naive CNN detection (which cannot track ownership). Our novelty is the **hybrid multi-stage edge architecture** coupling lightweight anchor-free detection (YOLOv8n) with a zero-ReID two-stage tracker (ByteTrack), a multi-factor spatial-temporal ownership formulation (edge distance + persistence + velocity cosine), and a deterministic finite state machine with occlusion grace windows and return hysteresis.

### Q92: If an examiner asks: "Did you collect and train on a dataset?", what do you say?
**Answer:**
> *"No, sir/ma'am. We evaluated our system using zero-shot transfer inference with the official COCO-pretrained YOLOv8n weights. We designed dataset ingestion pipelines for ABODA and PETS2006, but our experimental contribution focused on downstream multi-object tracking, spatial-temporal association, deterministic state reasoning, and edge deployment optimization. We verified this with synthetic edge cases and real-world webcam tests."*

### Q93: Why didn't you use an end-to-end Deep Learning model (e.g., Video Transformer or 3D CNN) to predict abandonment directly from video?
**Answer:** End-to-end video models (e.g., TimeSformer, SlowFast, VideoMAE) require massive computational power ($> 100\text{ GFLOPs}$), cannot run real-time on edge CPUs, require thousands of labeled temporal video sequences to train, and act as "black boxes" that cannot provide explainable safety audits. Our modular architecture separates perception (YOLOv8n) from reasoning (deterministic state machine), providing strict explainability, sub-millisecond reasoning, and zero data requirement for state logic.

### Q94: What are the main limitations of your current implementation?
**Answer:**
1. Lack of ground-plane camera calibration (distances are measured in pixels rather than real-world meters).
2. Susceptibility to ID switches in dense pedestrian crowds without deep Re-ID appearance features.
3. Inability to distinguish an owner from an accomplice who takes the bag (theft vs recovery).
4. No evaluation yet on full benchmark sequences of ABODA or PETS2006.

### Q95: How would you calibrate pixels to real meters in a production deployment?
**Answer:** By computing a homography matrix $H$ using 4 known coplanar ground reference points:
$$\mathbf{x}_{ground} = H \mathbf{x}_{image}$$
This maps image coordinates directly to metric ground-plane coordinates $[X_m, Y_m]$, ensuring that distance thresholds (e.g., $2.5\text{ meters}$) remain uniform regardless of whether the object is close to the lens or far away in the background.

### Q96: What is your individual contribution to this group project?
*(Adapt based on your specific role)*:
- **Student A (Perception & Pipeline):** Configured YOLOv8n detector, class mapping strategy, PyTorch/ONNX inference pipelines, and Streamlit user interface integration.
- **Student B (Tracking & Association):** Implemented ByteTrack two-stage association, velocity vector kinematics, bounding-box edge distance formulations, and composite association scoring.
- **Student C (Temporal State Machine & Edge Optimization):** Engineered the 7-state finite state machine, occlusion tolerance windows, return hysteresis, dual audio siren alerts, and CPU latency benchmarking.

### Q97: How does your system perform if the owner leaves the camera frame entirely?
**Answer:** In `src/temporal/state_machine.py`, if the confirmed owner is no longer detected, their track drops. `rec.last_owner_distance` defaults to infinity ($> 180\text{ px}$). The state transitions to `SEPARATED`, `st.last_known_owner_id` is preserved in memory, and the separation countdown proceeds toward `ABANDONED`.

### Q98: What would happen if you deployed this on a Raspberry Pi 5?
**Answer:** A Raspberry Pi 5 (Quad-core Cortex-A76 @ 2.4GHz) running YOLOv8n via ONNX Runtime or NCNN with INT8 quantization achieves $\sim 15 - 20\text{ FPS}$ at $384\times 384$ resolution. The tracking and state machine consume $< 1\text{ ms}$, making the complete system fully viable on a \$60 single-board computer without a GPU.

### Q99: Why did you implement ByteTrack instead of DeepSORT?
**Answer:** DeepSORT runs a convolutional feature extractor (Re-ID network) on every detected bounding box in every frame. On an edge CPU, extracting Re-ID features for 10 people adds $30 - 60\text{ ms}$ of latency, dropping frame rate to $< 10\text{ FPS}$. ByteTrack achieves competitive tracking accuracy without any Re-ID network by intelligently associating low-confidence boxes, running in $< 1\text{ ms}$.

### Q100: If you had 3 more months to work on this project, what would you prioritize?
**Answer:**
1. Download and benchmark on the complete ABODA and PETS2006 datasets to produce academic-grade ROC curves and event F1-scores.
2. Fine-tune YOLOv8n on airport/campus luggage classes to improve small-bag recall.
3. Integrate homography-based perspective camera calibration.
4. Deploy on an NVIDIA Jetson Orin Nano with TensorRT FP16 to achieve $> 60\text{ FPS}$ multi-stream surveillance.
