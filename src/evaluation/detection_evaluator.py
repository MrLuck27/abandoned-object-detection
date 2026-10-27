"""Detection evaluation engine for bounding box metrics and mAP reporting."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json

try:
    from ultralytics import YOLO
    _HAS_ULTRALYTICS = True
except ImportError:
    _HAS_ULTRALYTICS = False

logger = logging.getLogger("AbandonedObjectDetection.DetectionEvaluator")


class DetectionEvaluator:
    """Evaluates YOLO detector on a YOLO-format dataset split."""

    def __init__(self, model_path: str = "yolov8n.pt", data_yaml: Optional[str] = None):
        if not _HAS_ULTRALYTICS:
            raise ImportError("Ultralytics is required for detection evaluation.")
        self.model_path = model_path
        self.data_yaml = data_yaml
        self.model = YOLO(model_path)

    def evaluate(
        self,
        data_yaml: Optional[str] = None,
        split: str = "val",
        imgsz: int = 640,
        batch: int = 16,
        device: str = "0",
    ) -> Dict[str, Any]:
        """Run validation and extract standard COCO-style metrics.

        Returns:
            Dict containing Precision, Recall, mAP50, mAP50_95, and speed breakdown.
        """
        yaml_path = data_yaml or self.data_yaml
        if not yaml_path:
            raise ValueError("A dataset YAML configuration path must be specified.")

        logger.info(f"Evaluating model '{self.model_path}' on dataset '{yaml_path}' split='{split}'...")

        metrics = self.model.val(
            data=yaml_path,
            split=split,
            imgsz=imgsz,
            batch=batch,
            device=device,
            verbose=False,
        )

        results = {
            "model": self.model_path,
            "split": split,
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
            "fitness": float(metrics.fitness),
            "speed_preprocess_ms": float(metrics.speed.get("preprocess", 0.0)),
            "speed_inference_ms": float(metrics.speed.get("inference", 0.0)),
            "speed_postprocess_ms": float(metrics.speed.get("postprocess", 0.0)),
            "per_class": {},
        }

        # Class names
        names = metrics.names
        for cls_idx, cls_name in names.items():
            if cls_idx < len(metrics.box.p):
                results["per_class"][cls_name] = {
                    "precision": float(metrics.box.p[cls_idx]),
                    "recall": float(metrics.box.r[cls_idx]),
                    "map50": float(metrics.box.ap50[cls_idx]),
                    "map50_95": float(metrics.box.ap[cls_idx]),
                }

        return results
