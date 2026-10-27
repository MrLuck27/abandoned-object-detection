"""Training script for fine-tuning YOLO detector on person and bag/luggage classes."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Ultralytics must be installed to run training. Run `pip install ultralytics`.")

from src.utils.config import load_yaml, merge_configs

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger("TrainDetector")


def train_detector(
    config_path: str = "configs/detector.yaml",
    dataset_yaml: str = "data/processed/yolo_dataset/dataset.yaml",
    resume: bool = False,
    override_epochs: int = None,
    override_batch: int = None,
    override_device: str = None,
):
    """Fine-tune lightweight YOLO on custom dataset with transfer learning."""
    cfg = load_yaml(config_path)

    # Dataset check
    data_path = Path(dataset_yaml)
    if not data_path.exists():
        logger.error(
            f"Dataset configuration file not found at: {data_path.resolve()}.\n"
            f"Please run `python scripts/prepare_aboda.py` or `python scripts/create_local_dataset.py` first."
        )
        return

    model_name = cfg.get("model", "yolov8n.pt")
    epochs = override_epochs if override_epochs is not None else cfg.get("epochs", 50)
    batch = override_batch if override_batch is not None else cfg.get("batch", 16)
    imgsz = cfg.get("imgsz", 640)
    lr0 = cfg.get("lr0", 0.01)
    device = override_device if override_device is not None else str(cfg.get("device", "0"))
    import torch
    if device != "cpu" and not torch.cuda.is_available():
        logger.warning(f"CUDA device '{device}' requested but torch.cuda.is_available() is False. Using 'cpu'.")
        device = "cpu"
    workers = cfg.get("workers", 4)
    project = cfg.get("project", "results/models")
    name = cfg.get("name", "yolo_abandoned_detector")

    logger.info(
        f"Starting YOLO training:\n"
        f"  Base Model: {model_name}\n"
        f"  Dataset: {dataset_yaml}\n"
        f"  Epochs: {epochs}\n"
        f"  Batch: {batch}\n"
        f"  Image Size: {imgsz}\n"
        f"  Initial LR: {lr0}\n"
        f"  Patience: {patience}\n"
        f"  Device: {device}\n"
        f"  Output Project: {project}/{name}"
    )

    # Initialize YOLO with pretrained weights for transfer learning
    model = YOLO(model_name)

    # Train
    results = model.train(
        data=str(data_path.resolve()),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        lr0=lr0,
        patience=patience,
        device=device,
        workers=workers,
        project=project,
        name=name,
        resume=resume,
        exist_ok=True,
        save=True,
        plots=True,
    )

    best_weight_path = Path(project) / name / "weights" / "best.pt"
    if best_weight_path.exists():
        # Copy to standard models/weights/ directory
        std_weights_dir = Path("models/weights")
        std_weights_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_weight_path, std_weights_dir / "best.pt")
        logger.info(f"Training completed successfully! Best weights copied to: {std_weights_dir / 'best.pt'}")
    else:
        logger.info("Training completed.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train lightweight YOLO detector")
    parser.add_argument("--config", type=str, default="configs/detector.yaml", help="Path to detector config")
    parser.add_argument("--data", type=str, default="data/processed/yolo_dataset/dataset.yaml", help="Path to dataset yaml")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", type=str, default=None, help="Override compute device (e.g. '0' or 'cpu')")
    args = parser.parse_args()

    train_detector(
        config_path=args.config,
        dataset_yaml=args.data,
        resume=args.resume,
        override_epochs=args.epochs,
        override_batch=args.batch,
        override_device=args.device,
    )
