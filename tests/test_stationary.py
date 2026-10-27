"""Unit tests for stationary-object detection and displacement calculation."""

import numpy as np
import pytest
from src.tracking.track_state import Track


def test_stationary_detection_true():
    # Object stays around (100, 100) with minor jitter <= 2px
    trk = Track(
        track_id=1,
        class_id=1,
        class_name="bag",
        bbox=np.array([90, 90, 110, 110]),
        confidence=0.9,
        last_timestamp=0.0,
    )

    for i in range(1, 30):
        t = i * 0.1
        jitter = np.random.uniform(-1.0, 1.0, 4)
        box = np.array([90, 90, 110, 110]) + jitter
        trk.update_detection(box, 0.9, timestamp=t)

    assert trk.is_stationary(displacement_threshold=15.0, window_seconds=3.0) is True


def test_stationary_detection_false_when_moving():
    # Object moves from x=100 to x=300
    trk = Track(
        track_id=2,
        class_id=1,
        class_name="bag",
        bbox=np.array([90, 90, 110, 110]),
        confidence=0.9,
        last_timestamp=0.0,
    )

    for i in range(1, 30):
        t = i * 0.1
        box = np.array([90 + i * 5, 90, 110 + i * 5, 110])
        trk.update_detection(box, 0.9, timestamp=t)

    assert trk.is_stationary(displacement_threshold=15.0, window_seconds=3.0) is False
    assert trk.calculate_displacement(window_seconds=3.0) > 50.0
