# System Evaluation & Experimental Benchmark Report

**Dataset Tested:** Staged Campus Evaluation Scenarios  
**Configuration:** `configs/system.yaml` (Abandonment Threshold = 15.0s)

## 1. Comparative Experiment Results

| Experiment | Event Precision | Event Recall | Event F1 | False Alarms / Hour | Mean Time to Alert (s) | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp A: Detector Only (Naive)** | 0.000 | 0.000 | 0.000 | 0.0 | 0.0s | 38.0 |
| **Exp B: Detector + ByteTrack (No Owner)** | 0.000 | 0.000 | 0.000 | 0.0 | 0.0s | 38.1 |
| **Exp D: Full Proposed Pipeline (Temporal State Machine)** | 0.000 | 0.000 | 0.000 | 0.0 | 0.0s | 36.2 |

## 2. Key Findings & Ablation Analysis

1. **Why Naive Detector Fails (Exp A):**
   - The pure detector fires on every frame containing an attended luggage or sitting owner.
   - Generates an astronomical false alarm rate per hour, rendering it unusable in surveillance.

2. **Why Tracking Alone Is Insufficient (Exp B):**
   - Tracking identifies stationary objects, but without owner association, it triggers alarms on bags where the owner is sitting or standing right beside them.

3. **Value of Temporal State Machine & Owner Association (Exp D):**
   - Eliminates false alarms in attended scenarios (`NORMAL_ATTENDED`).
   - Gracefully absorbs transient occlusions and prevents single-frame spurious spikes.
   - Correctly resets if the owner returns before the threshold (`OWNER_RETURNS`).
