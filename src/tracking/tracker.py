"""Abstract Base Tracker interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List
from ..detection.detector import Detection
from .track_state import Track


class BaseTracker(ABC):
    """Abstract interface for multi-object tracking."""

    @abstractmethod
    def update(self, detections: List[Detection], timestamp: float) -> List[Track]:
        """Update tracker with frame detections.

        Args:
            detections: List of Detection instances for the current frame.
            timestamp: Video timestamp in seconds.

        Returns:
            List of active Track instances.
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset internal tracker state."""
        pass
