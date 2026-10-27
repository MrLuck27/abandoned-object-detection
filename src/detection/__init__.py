"""Detection module providing Detection dataclass and detector wrappers."""

from .detector import BaseDetector, Detection
from .yolo_detector import YOLODetector

__all__ = ["BaseDetector", "Detection", "YOLODetector"]
