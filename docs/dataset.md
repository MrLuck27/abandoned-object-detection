# Dataset Setup and Management Guide

This project supports three dataset sources:
1. **ABODA** (Abandoned Object Detection in Airport) — Primary Benchmark Dataset
2. **PETS 2006 / AVSS 2007** — Additional Luggage Surveillance Benchmarks
3. **Local Campus Dataset** — Controlled Staged Real-world Video Sequences

---

## 1. Primary Dataset: ABODA

### Description
The **ABODA** dataset is an established benchmark designed for unattended luggage detection in public transit and airport lobbies. It contains real surveillance sequences with varying illumination, crowd density, occlusions, and owner behaviors.

### Obtaining the Dataset
- Due to academic hosting agreements, download the dataset directly from the official research repository or University host:
  - Official ABODA Archive: [ABODA Dataset Project Page](https://github.com/flysoaryang/ABODA) or University mirror.
- Place the downloaded sequences into:
  ```
  data/raw/aboda/
    ├── video1.avi
    ├── video1.xml
    ├── video2.avi
    ├── video2.xml
    └── ...
  ```

### Data Preprocessing & Sequence-Level Splitting
To prevent **temporal data leakage**, entire video sequences must remain within the same split (train, validation, or test):
```bash
python scripts/prepare_aboda.py --raw_dir data/raw/aboda --output_dir data/processed/yolo_dataset --sample_rate 5
```
This extracts frames at 5-frame intervals, maps bounding boxes to YOLO format, and outputs `data/processed/yolo_dataset/dataset.yaml`.

---

## 2. Additional Benchmarks: PETS 2006 & AVSS

### Description
- **PETS 2006**: IEEE International Workshop on Performance Evaluation of Tracking and Surveillance benchmark (Scenarios S1 through S7).
- **AVSS 2007**: IEEE Advanced Video and Signal Based Surveillance abandoned baggage sequences.

### Placement & Processing
1. Place videos and XML annotations into:
   ```
   data/raw/pets_avss/
   ```
2. Convert annotations and create temporal ground-truth files:
   ```bash
   python scripts/prepare_pets_avss.py --raw_dir data/raw/pets_avss --output_dir data/processed/yolo_dataset
   ```

---

## 3. Local Controlled Campus Dataset

For immediate testing, staged demonstrations, and local fine-tuning:
```bash
python scripts/create_local_dataset.py --local_dir data/local
```
This generates:
- `data/local/videos/campus_true_abandoned.mp4`
- `data/local/videos/campus_owner_returns.mp4`
- `data/local/videos/campus_normal_attended.mp4`
- `data/local/annotations/campus_events.json`

---

## 4. Dataset Directory Structure

```
data/
├── raw/
│   ├── aboda/           # Raw ABODA videos and XMLs
│   └── pets_avss/       # Raw PETS/AVSS benchmarks
├── processed/
│   └── yolo_dataset/
│       ├── dataset.yaml
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── labels/
│           ├── train/
│           ├── val/
│           └── test/
├── annotations/
│   └── benchmark_events.json
└── local/
    ├── videos/
    └── annotations/
```
