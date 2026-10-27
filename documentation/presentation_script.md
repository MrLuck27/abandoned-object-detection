# Presentation Scripts, Elevator Pitch & Demonstration Guide

**Project Title:** Unattended / Abandoned Object Detection with Temporal Reasoning for Edge AI Surveillance  
**Repository Path:** `C:\Users\Aditya\.gemini\antigravity\scratch\abandoned_object_detection`  

---

## 1. The 2-Minute Elevator Pitch (Viva / Panel Quick Intro)

> *"Good morning, respected examiners and faculty members. Our group project addresses a critical challenge in modern automated surveillance: detecting unattended and abandoned luggage in high-density public venues such as railway terminals, airports, and university campuses.*
>
> *Traditional CCTV monitoring relies on human operators who suffer from visual fatigue, while standard AI approaches like background subtraction fail under lighting shifts and walking crowds. On the other hand, running a modern object detector like YOLO by itself causes hundreds of false alarms, because it cannot tell if a bag's owner is standing right next to it or has left.*
>
> *To solve this, we engineered a hybrid, real-time edge architecture that decouples spatial perception from temporal reasoning. Our pipeline uses a lightweight, anchor-free YOLOv8 Nano detector, paired with a class-consistent two-stage ByteTrack tracker. We then introduce a multi-factor spatial-temporal ownership associator that links bags to their owners using physical edge proximity, temporal persistence, and velocity vector correlation.*
>
> *Finally, our deterministic 7-state finite state machine enforces strict abandonment rules: the bag must remain verified stationary, the owner must separate beyond a 180-pixel perimeter, and that separation must persist continuously for 15 seconds. It includes return hysteresis to prevent timer flickering and tolerates pedestrian occlusions for up to 3 seconds.*
>
> *We deployed our system on an edge CPU pipeline with an interactive Streamlit dashboard, dual hardware and browser sirens, and ONNX Runtime acceleration, achieving 35 to 40 FPS at low latency without needing an expensive dedicated GPU. Thank you, and we welcome your questions."*

---

## 2. The 5-Minute Formal Presentation Script (Slide-by-Slide)

### Slide 1: Title & Team Introduction (0:00 – 0:30)
- **Visual:** Project title, group member names, university logo, system overview graphic.
- **Speaker:** "Respected panel, today we present our capstone engineering project: *Unattended and Abandoned Object Detection with Temporal Reasoning for Edge AI Surveillance*. Our objective was to develop an intelligent, low-latency surveillance framework capable of identifying left-behind luggage and raising alerts while suppressing false alarms in dynamic public environments."

### Slide 2: Problem Statement & Existing Limitations (0:30 – 1:15)
- **Visual:** Diagram contrasting human operator fatigue, background subtraction failure (shadows/crowds), and naive YOLO failure (attended bags flagged).
- **Speaker:** "Automated surveillance faces a severe dilemma. Background subtraction techniques like MOG2 fail under walking crowds and camera shake. Conversely, modern deep learning detectors like YOLO detect luggage accurately in individual frames, but lack temporal memory. If a student rests their backpack on the floor while waiting for a lecture, a naive detector alarms immediately. True surveillance requires understanding *spatial ownership, object stationarity, and the duration of human-object separation*."

### Slide 3: Proposed System Architecture (1:15 – 2:30)
- **Visual:** Modular pipeline flowchart: Input Video -> YOLOv8n (COCO-mapped) -> 2-Stage ByteTrack -> Owner Associator -> 7-State Finite State Machine -> Alert Engine.
- **Speaker:** "Our proposed architecture is structured into four cooperative modules:
  First, an anchor-free YOLOv8 Nano model extracts bounding boxes for people and luggage classes including backpacks, suitcases, handbags, and laptops.
  Second, our class-consistent ByteTrack algorithm associates detections across frames in two stages, recovering occluded boxes without deep Re-ID compute overhead.
  Third, our spatial-temporal associator evaluates candidate owners using a weighted composite function combining physical boundary edge distance, temporal proximity persistence, and motion vector correlation.
  Fourth, our deterministic 7-state finite state machine governs decision logic. An object transitions from ATTENDED to SEPARATED only if its owner moves beyond 180 pixels. The separation timer then accumulates, transitioning to SUSPECTED at 7.5 seconds, and triggering an active ABANDONED alert once 15 seconds elapse while verifying object stationarity."

### Slide 4: Key Technical Innovations & Edge Optimizations (2:30 – 3:45)
- **Visual:** Hysteresis graph ($180\text{px}$ vs $120\text{px}$), occlusion grace window ($3.0\text{s}$), ONNX benchmark comparison table.
- **Speaker:** "To make this system robust and edge-deployable, we implemented three critical engineering innovations:
  1. *Hysteresis and Occlusion Grace:* We separated the departure threshold (180px) from the return threshold (120px) to prevent boundary flickering, and maintained state through 3-second pedestrian occlusions.
  2. *Adaptive Track Initiation:* We modified ByteTrack to initiate bag tracks at a lower threshold of 0.20, allowing low-contrast bags on dim floors to be tracked reliably.
  3. *Edge Throughput Optimization:* By exporting the model to ONNX Runtime and optimizing input resolution to 384 pixels, we reduced latency by 18%, achieving over 35 FPS on an edge CPU workstation."

### Slide 5: Experimental Results, Limitations & Conclusion (3:45 – 5:00)
- **Visual:** Streamlit dashboard interface showing live webcam feed, bounding box overlays, countdown timer, dual audio alarm mechanism, and tabular log.
- **Speaker:** "We validated the system using synthetic staged scenarios and real-world webcam feeds. In our testing, the system successfully suppressed alarms during normal attended scenarios, correctly reset timers when owners returned, and fired dual hardware and browser sirens within 15.4 seconds of true abandonment.
  Current limitations include pixel-based distance metrics rather than metric homography and evaluation on CPU rather than physical Jetson TensorRT hardware. In future work, we plan to fine-tune on custom airport datasets and integrate camera perspective calibration. In conclusion, our modular pipeline demonstrates that combining lightweight deep detection with deterministic temporal reasoning provides an explainable, real-time edge security solution. Thank you."

---

## 3. Live Project Demonstration Protocol

When demonstrating the live application (`app.py`) to the examiners, execute these three test cases in order:

### Demo Scenario 1: Normal Attended Luggage (False Alarm Suppression)
1. In the sidebar, verify **⚡ Performance Mode** is on **🚀 Ultra Fast (384px)**.
2. Select **📹 Live Camera** -> **Device 1** (or Device 0) -> click **▶️ Start Processing**.
3. Stand in front of the camera holding a backpack. Place the backpack on the floor and remain standing beside it.
4. **Point out to Examiner:**
   - Notice the green bounding box labeled `Bag B01 [ATTENDED]`.
   - Point out the yellow association line connecting you (`P01`) to your bag (`B01`).
   - Highlight that the separation timer does not count, and **no alarm fires**, demonstrating false alarm suppression.

### Demo Scenario 2: True Abandoned Object (Alarm Generation)
1. With the bag on the floor, step backwards or walk out of the camera's view ($> 180\text{ px}$).
2. **Point out to Examiner:**
   - The label switches to yellow: `Bag B01 [SEPARATED]`.
   - The countdown timer begins: `Sep: 1.2s`, `3.5s`, `5.0s`.
   - At $7.5\text{s}$, the label changes to orange: `Bag B01 [SUSPECTED_ABANDONED]`.
   - At $15.0\text{s}$, the label turns red: `Bag B01 [ABANDONED]`.
   - The red animated banner flashes: `🚨 SECURITY ALERT: Bag B01 left unattended for 15.0s!`.
   - The dual audible siren beeps through the laptop speakers and browser tab.
   - The event is logged in the table on the right.

### Demo Scenario 3: Owner Returns (Timer Reset & Recovery)
1. Place the bag down again, step away until the timer reaches $\sim 6\text{s}$ (in `SEPARATED` state).
2. Walk back to the bag and stand beside it ($< 120\text{ px}$).
3. **Point out to Examiner:**
   - The state transitions immediately to `RECOVERED` and then back to `ATTENDED`.
   - The timer resets to $0.0\text{s}$ without triggering an alarm, proving hysteresis and recovery resilience.
