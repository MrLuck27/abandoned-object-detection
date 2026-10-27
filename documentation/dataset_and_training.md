# Dataset Documentation & Training Methodology

**Project Title:** Unattended / Abandoned Object Detection with Temporal Reasoning for Edge AI Surveillance  
**Repository Path:** `C:\Users\Aditya\.gemini\antigravity\scratch\abandoned_object_detection`  

---

## 1. Executive Dataset Audit

| Dataset Name | Status in Repository | Path in Repository | Files Available | Annotated Objects | Usage in Project |
| :--- | :---: | :--- | :---: | :---: | :--- |
| **ABODA** | **NOT DOWNLOADED** | `data/raw/aboda/` | `.gitkeep` (0 bytes) | None | Ingestion script written only |
| **PETS2006 / AVSS** | **NOT DOWNLOADED** | `data/raw/pets_avss/` | `.gitkeep` (0 bytes) | None | Ingestion script written only |
| **Staged Campus Scenarios** | **VERIFIED PRESENT** | `data/local/videos/` | 3 MP4 files | Synthetic sprites | Local system logic verification |
| **COCO-2017 Pretrained** | **VERIFIED PRESENT** | `yolov8n.pt` (6.25 MB) | 1 checkpoint file | 80 classes ($330\text{k}$ images) | Zero-shot base detector |

---

## 2. In-Depth Benchmark Dataset Profiles

### A. ABODA (Abandoned Object Dataset in Airport)
* **Official Publisher:** University of Central Florida (UCF) / Computer Vision Research.
* **Official Source Link:** `https://www.crcv.ucf.edu/data/ABODA/`
* **Dataset Characteristics:**
  - Sequences: 11 video clips captured by fixed surveillance cameras in indoor airport concourses and corridors.
  - Resolution: $720\times 480$ and $640\times 480$ at $25 - 30\text{ FPS}$.
  - Ground Truth: XML annotation files marking bounding boxes of people, luggage, and temporal intervals where abandonment occurs.
  - Challenges Included: Complex occlusions, varying lighting, mirror reflections on polished floors, and owners pacing back and forth.
* **Repository Ingestion Script:** `scripts/prepare_aboda.py`:
  - Parses ABODA XML tracks (`.//track` with labels `person` or `bag`).
  - Converts pixel bounding boxes $[x_{tl}, y_{tl}, x_{br}, y_{br}]$ to normalized YOLO text format $[c, x_c, y_c, w, h]$.
  - Implements sequence-level partitioning: $70\%$ train, $15\%$ val, $15\%$ test to prevent temporal data leakage.
* **Current Status in Repository:** Only the preprocessing script is present; raw videos were **not downloaded**.

### B. PETS 2006 Benchmark (EC FP6 Project)
* **Official Publisher:** IEEE International Workshop on Performance Evaluation of Tracking and Surveillance.
* **Official Source Link:** `https://www.cvg.reading.ac.uk/PETS2006/data.html`
* **Dataset Characteristics:**
  - Sequences: 7 multi-camera scenarios (S1 through S7) recorded in a public railway station concourse.
  - Camera Views: 4 synchronized calibrated camera perspectives.
  - Complexity: High crowd density, non-owners standing in close proximity to abandoned bags, and rapid lighting variations.
* **Repository Ingestion Script:** `scripts/prepare_pets_avss.py`.
* **Current Status in Repository:** Preprocessing stub present; raw videos were **not downloaded**.

### C. Staged Local Campus Scenarios
* **Generator Script:** `scripts/create_local_dataset.py`.
* **Format:** MPEG-4 (.mp4), $640\times 480$, $30\text{ FPS}$, duration $20.0\text{ seconds}$ ($600\text{ frames}$ each).
* **Included Scenarios:**
  1. `campus_true_abandoned.mp4`: Owner enters with bag, places it down at $t=3\text{s}$, walks away at $t=5\text{s}$, exits frame. Bag remains stationary until alert threshold ($15\text{s}$) is surpassed.
  2. `campus_owner_returns.mp4`: Owner places bag down, departs to $d > 180\text{ px}$, but returns within $8\text{ seconds}$, verifying state machine recovery and timer cancellation.
  3. `campus_normal_attended.mp4`: Owner places bag down and remains stationary beside it ($d < 60\text{ px}$) for the entire duration, verifying suppression of false alarms.
* **Ground Truth Metadata:** Stored in `data/local/annotations/campus_events.json`.

---

## 3. Training & Validation Audit

### Did Model Training Occur?
**NO.** The project did not conduct model training or fine-tuning.
- `models/weights/best.pt`: **Does not exist.**
- `data/processed/yolo_dataset/images/train/`: **Empty (0 images).**
- `results/metrics/dataset_statistics.json`: Confirms `"total_images": 0`.
- The system employs `yolov8n.pt`, the official release checkpoint pretrained by Ultralytics on the COCO-2017 dataset.

### Why Pretrained Zero-Shot Inference Was Utilized
1. **Transferability:** COCO classes `backpack`, `handbag`, and `suitcase` have hundreds of thousands of diverse real-world training instances across varying angles and light levels.
2. **Core Research Scope:** The core academic contribution was proving that deterministic temporal reasoning and multi-object tracking eliminate the false alarms that plague static detectors, rather than merely tweaking feature extractor weights.

---

## 4. Proposed Custom Fine-Tuning Methodology

For future execution or viva presentation, the following rigorous fine-tuning protocol is designed:

```
[Raw Surveillance Videos] 
       │
       ▼ (Sequence-Level Split: 70% Train / 15% Val / 15% Test)
[Frame Extraction with Stride s=5]
       │
       ▼ (YOLO Format Normalization)
[Data Augmentation: Mosaic, MixUp, HSV, Affine]
       │
       ▼
[Fine-Tuning YOLOv8n Backbone (Frozen P5, Trainable Neck/Head)]
       │
       ▼ (50 Epochs, AdamW, Cosine LR Decay, Early Stopping)
[Final Optimized Model: best.pt]
```

### Hyperparameter Specifications (`configs/detector.yaml`)
- **Base Architecture:** `yolov8n.pt`
- **Target Classes:** 2 (`0: person`, `1: bag`)
- **Image Resolution:** $640\times 640$ pixels
- **Batch Size:** 16 (optimized for 8GB edge GPUs / workstations)
- **Epochs:** 50 with Early Stopping (`patience = 10`)
- **Optimizer:** AdamW ($\text{lr}_0 = 0.001$, $\text{lrf} = 0.01$, $\text{weight\_decay} = 0.0005$)
- **Loss Formulation:** $\mathcal{L}_{total} = \lambda_{box} \mathcal{L}_{CIoU} + \lambda_{cls} \mathcal{L}_{BCE} + \lambda_{dfl} \mathcal{L}_{DFL}$
- **Sequence-Level Split Guarantee:** All frames from Video Sequence $K$ reside exclusively in either Train, Val, or Test, guaranteeing zero temporal data leakage.
