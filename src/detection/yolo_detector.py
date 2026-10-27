"""Ultralytics YOLO wrapper for person and luggage/bag detection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

try:
    from ultralytics import YOLO
    _HAS_ULTRALYTICS = True
except ImportError:
    _HAS_ULTRALYTICS = False

from .detector import BaseDetector, Detection

logger = logging.getLogger("AbandonedObjectDetection.YOLODetector")


class YOLODetector(BaseDetector):
    """YOLO detector supporting custom trained models and COCO-pretrained fallbacks."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: str = "0",
        coco_mapping: Optional[Dict[int, int]] = None,
        imgsz: int = 480,
    ):
        """Initialize the YOLO detector.

        Args:
            model_path: Path to YOLO weights (.pt, .onnx, or engine).
            confidence_threshold: Min detection confidence.
            iou_threshold: IoU threshold for NMS.
            device: '0', 'cpu', etc.
            coco_mapping: Mapping from COCO class IDs to internal IDs.
            imgsz: Input resolution for YOLO inference (lower = faster FPS).
        """
        if not _HAS_ULTRALYTICS:
            raise ImportError("Ultralytics package is not installed. Run `pip install ultralytics`.")

        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz

        import torch
        if device != "cpu" and not torch.cuda.is_available():
            logger.warning(f"CUDA device '{device}' requested but torch.cuda.is_available() is False. Falling back to 'cpu'.")
            self.device = "cpu"
        else:
            self.device = device

        self.model = YOLO(model_path)

        # Internal target class names: 0: "person", 1: "bag"
        self.target_names = {0: "person", 1: "bag"}

        # Determine whether model is custom 2-class or COCO 80-class
        model_names = self.model.names if hasattr(self.model, "names") else {}
        self.is_custom_model = len(model_names) == 2 and 0 in model_names and 1 in model_names

        # Mapping for COCO pretrained weights
        if coco_mapping is not None:
            self.coco_mapping = coco_mapping
        else:
            self.coco_mapping = {
                0: 0,   # person
                24: 1,  # backpack
                26: 1,  # handbag
                28: 1,  # suitcase
                63: 1,  # laptop
            }

        logger.info(
            f"Initialized YOLODetector with model='{model_path}', device='{device}', "
            f"conf={confidence_threshold}, is_custom={self.is_custom_model}"
        )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run YOLO inference on frame and return filtered Detection objects."""
        # Ultralytics inference
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results or len(results) == 0:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        for i in range(len(xyxy)):
            raw_cls = classes[i]
            conf = float(confs[i])
            box = xyxy[i]

            if self.is_custom_model:
                if raw_cls in self.target_names:
                    target_cls = raw_cls
                    cls_name = self.target_names[target_cls]
                    detections.append(Detection(
                        bbox=box,
                        confidence=conf,
                        class_id=target_cls,
                        class_name=cls_name,
                    ))
            else:
                # Using COCO model: map person (0) and luggage classes (24, 26, 28)
                if raw_cls in self.coco_mapping:
                    target_cls = self.coco_mapping[raw_cls]
                    cls_name = self.target_names[target_cls]
                    detections.append(Detection(
                        bbox=box,
                        confidence=conf,
                        class_id=target_cls,
                        class_name=cls_name,
                    ))

        return detections
