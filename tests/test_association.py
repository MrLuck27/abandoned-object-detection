"""Unit tests for Owner-Bag Association."""

import numpy as np
import pytest
from src.tracking.track_state import Track
from src.association.owner_associator import OwnerAssociator


def test_owner_association_confirmation():
    associator = OwnerAssociator(
        max_association_distance=150.0,
        min_attended_duration=0.5,
        min_association_score=0.4,
    )

    # Person P1 at (100, 100), Bag B1 at (120, 110) - close proximity
    p1 = Track(1, 0, "person", np.array([80, 50, 120, 150]), 0.9, last_timestamp=0.0)
    b1 = Track(1, 1, "bag", np.array([110, 100, 130, 120]), 0.85, last_timestamp=0.0)

    # Simulate 10 frames (1 second duration)
    for i in range(10):
        t = i * 0.1
        p1.update_detection(p1.bbox, 0.9, t)
        b1.update_detection(b1.bbox, 0.85, t)
        records = associator.update([p1], [b1], timestamp=t)

    rec = records[1]
    assert rec.is_owner_confirmed is True
    assert rec.confirmed_owner_id == 1


def test_distant_person_not_associated():
    associator = OwnerAssociator(
        max_association_distance=100.0,
        min_attended_duration=0.5,
        min_association_score=0.4,
    )

    # Person P2 at (800, 800), Bag B1 at (100, 100) - far distance
    p2 = Track(2, 0, "person", np.array([780, 750, 820, 850]), 0.9, last_timestamp=0.0)
    b1 = Track(1, 1, "bag", np.array([90, 90, 110, 110]), 0.85, last_timestamp=0.0)

    for i in range(10):
        t = i * 0.1
        p2.update_detection(p2.bbox, 0.9, t)
        b1.update_detection(b1.bbox, 0.85, t)
        records = associator.update([p2], [b1], timestamp=t)

    rec = records[1]
    assert rec.is_owner_confirmed is False
    assert rec.confirmed_owner_id is None
