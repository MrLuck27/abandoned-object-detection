"""Abstract base class and data structures for object detection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class Detection:
    """Represents a single detected object in an image frame."""

    bbox: np.ndarray  # [x1, y1, x2, y2] in pixel coordinates
    confidence: float
    class_id: int
    class_name: str

    @property
    def x1(self) -> float:
        return float(self.bbox[0])

    @property
    def y1(self) -> float:
        return float(self.bbox[1])

    @property
    def x2(self) -> float:
        return float(self.bbox[2])

    @property
    def y2(self) -> float:
        return float(self.bbox[3])

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])

    @property
    def center(self) -> np.ndarray:
        """Calculate (center_x, center_y) centroid coordinates."""
        cx = (self.bbox[0] + self.bbox[2]) / 2.0
        cy = (self.bbox[1] + self.bbox[3]) / 2.0
        return np.array([cx, cy], dtype=np.float32)

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)


class BaseDetector(ABC):
    """Abstract interface that all object detectors must satisfy."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Perform object detection on a single frame.

        Args:
            frame: BGR numpy image array.

        Returns:
            List of Detection instances.
        """
        pass
