# Detector Training & Fine-Tuning Guide

This guide describes how to train and fine-tune the lightweight YOLO detector for person and baggage detection.

---

## 1. Overview & Strategy

For real-time edge processing (CCTV/Jetson/laptop), we employ a lightweight detector:
- **Baseline Architecture:** Ultralytics YOLOv8 Nano (`yolov8n.pt`)
- **Transfer Learning:** Initialized with pretrained COCO weights
- **Target Classes:**
  - Class `0`: `person`
  - Class `1`: `bag` (incorporating backpacks, handbags, suitcases, and luggage)

### Why Not Train From Scratch?
Edge surveillance environments benefit greatly from pretrained low-level visual features (edges, textures, human body parts). Fine-tuning pretrained weights accelerates convergence and prevents overfitting on small datasets.

---

## 2. Training Hyperparameters

All hyperparameters are controlled via `configs/detector.yaml`:

| Parameter | Recommended Value | Description |
| :--- | :---: | :--- |
| `model` | `yolov8n.pt` | Lightweight backbone |
| `epochs` | 50 | Total training epochs |
| `imgsz` | 640 | Input image resolution |
| `batch` | 16 (or auto) | Batch size |
| `lr0` | 0.01 | Initial learning rate |
| `lrf` | 0.01 | Final learning rate factor |
| `patience` | 15 | Early stopping epochs |
| `mosaic` | 1.0 | Mosaic data augmentation |
| `mixup` | 0.1 | Mixup augmentation |

---

## 3. Running Training

### Command
```bash
python scripts/train_detector.py --config configs/detector.yaml --data data/processed/yolo_dataset/dataset.yaml --epochs 50 --batch 16
```

### Resume Training
If training was interrupted:
```bash
python scripts/train_detector.py --resume
```

### Outputs
- Weights are automatically saved to:
  - `results/models/yolo_abandoned_detector/weights/best.pt`
  - Automatically copied to `models/weights/best.pt`
- Training curves, PR curves, and confusion matrices are saved to `results/models/yolo_abandoned_detector/`.

---

## 4. Model Evaluation

Evaluate the fine-tuned detector on the held-out test split:
```bash
python scripts/evaluate_detector.py --model models/weights/best.pt --data data/processed/yolo_dataset/dataset.yaml --split test
```

This outputs COCO metrics:
- Precision
- Recall
- mAP@0.5
- mAP@0.5:0.95
and saves the report to `results/metrics/detection_metrics.json`.
