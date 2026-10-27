"""Track representation and state management."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple
import numpy as np


@dataclass
class Track:
    """Represents a single persistent object or person track."""

    track_id: int
    class_id: int                     # 0: person, 1: bag
    class_name: str
    bbox: np.ndarray                  # [x1, y1, x2, y2]
    confidence: float
    age: int = 1                      # Total frames tracked
    time_since_update: int = 0        # Frames since last matched detection
    hits: int = 1                     # Number of positive associations
    is_confirmed: bool = True
    last_timestamp: float = 0.0

    # History buffer of (timestamp, centroid_xy)
    history: Deque[Tuple[float, np.ndarray]] = field(
        default_factory=lambda: deque(maxlen=150)
    )

    def __post_init__(self):
        cx = (self.bbox[0] + self.bbox[2]) / 2.0
        cy = (self.bbox[1] + self.bbox[3]) / 2.0
        center = np.array([cx, cy], dtype=np.float32)
        if len(self.history) == 0:
            self.history.append((self.last_timestamp, center))

    @property
    def center(self) -> np.ndarray:
        """Current centroid [cx, cy]."""
        cx = (self.bbox[0] + self.bbox[2]) / 2.0
        cy = (self.bbox[1] + self.bbox[3]) / 2.0
        return np.array([cx, cy], dtype=np.float32)

    @property
    def velocity(self) -> np.ndarray:
        """Current velocity vector [vx, vy] in pixels/second."""
        if len(self.history) < 2:
            return np.zeros(2, dtype=np.float32)

        t_now, p_now = self.history[-1]
        t_prev, p_prev = self.history[-2]
        dt = t_now - t_prev
        if dt <= 1e-4:
            return np.zeros(2, dtype=np.float32)

        return (p_now - p_prev) / dt

    @property
    def speed(self) -> float:
        """Current speed magnitude in pixels/second."""
        v = self.velocity
        return float(np.linalg.norm(v))

    def update_detection(self, bbox: np.ndarray, confidence: float, timestamp: float):
        """Update track with new frame detection."""
        self.bbox = bbox
        self.confidence = confidence
        self.last_timestamp = timestamp
        self.age += 1
        self.hits += 1
        self.time_since_update = 0

        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        self.history.append((timestamp, np.array([cx, cy], dtype=np.float32)))

    def mark_missed(self):
        """Mark frame where track had no associated detection."""
        self.time_since_update += 1
        self.age += 1

    def calculate_displacement(self, window_seconds: float = 3.0) -> float:
        """Calculate maximum displacement (spread) from initial position over a sliding window.

        Args:
            window_seconds: Duration in seconds to look back.

        Returns:
            Max Euclidean distance in pixels between current centroid and any position in the window.
        """
        if len(self.history) < 2:
            return 0.0

        current_t, current_pt = self.history[-1]
        cutoff_t = current_t - window_seconds

        window_pts = [pt for t, pt in self.history if t >= cutoff_t]
        if len(window_pts) < 2:
            # Fall back to comparing against previous historical position
            window_pts = [self.history[-2][1], current_pt]

        displacements = [np.linalg.norm(current_pt - pt) for pt in window_pts]
        return float(max(displacements))

    def is_stationary(
        self, displacement_threshold: float = 25.0, window_seconds: float = 3.0
    ) -> bool:
        """Check if object has moved less than displacement_threshold over the window duration.

        Args:
            displacement_threshold: Threshold in pixels.
            window_seconds: Time window in seconds.

        Returns:
            True if maximum displacement is below threshold.
        """
        # If track has been seen for less than 1.0 second, use available history
        disp = self.calculate_displacement(window_seconds=window_seconds)
        return disp < displacement_threshold
