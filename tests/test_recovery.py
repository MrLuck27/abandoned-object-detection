"""Unit tests for recovery detection and occlusion handling."""

import numpy as np
import pytest
from src.tracking.track_state import Track
from src.association.owner_associator import BagOwnershipRecord
from src.temporal.state_machine import TemporalStateMachine
from src.temporal.states import AbandonmentState


def test_owner_returns_recovery():
    sm = TemporalStateMachine(
        abandonment_time_threshold=10.0,
        owner_distance_threshold=150.0,
        owner_return_distance=80.0,
    )

    bag = Track(1, 1, "bag", np.array([100, 100, 130, 130]), 0.9, last_timestamp=0.0)

    # 1. Owner departs (200px away) -> SEPARATED
    rec_far = BagOwnershipRecord(
        bag_id=1, confirmed_owner_id=1, is_owner_confirmed=True, last_owner_distance=200.0
    )
    sm.update([bag], {1: rec_far}, timestamp=1.0)
    st, _ = sm.update([bag], {1: rec_far}, timestamp=6.0)
    assert st[1].current_state in [AbandonmentState.SEPARATED, AbandonmentState.SUSPECTED_ABANDONED]

    # 2. Owner walks back within return distance (60px) at t=7.0s
    rec_returned = BagOwnershipRecord(
        bag_id=1, confirmed_owner_id=1, is_owner_confirmed=True, last_owner_distance=60.0
    )
    st_rec, alerts = sm.update([bag], {1: rec_returned}, timestamp=7.0)

    assert st_rec[1].current_state == AbandonmentState.RECOVERED
    assert st_rec[1].separation_start_time is None
    assert st_rec[1].separation_duration == 0.0
    assert len(alerts) == 0


def test_bag_moved_recovery():
    sm = TemporalStateMachine(
        abandonment_time_threshold=10.0,
        owner_distance_threshold=150.0,
        object_movement_threshold=20.0,
    )

    bag = Track(1, 1, "bag", np.array([100, 100, 130, 130]), 0.9, last_timestamp=0.0)
    rec_far = BagOwnershipRecord(
        bag_id=1, confirmed_owner_id=1, is_owner_confirmed=True, last_owner_distance=200.0
    )

    # Stationary separation for 5 seconds
    for step in range(1, 6):
        t = float(step)
        bag.update_detection(np.array([100, 100, 130, 130]), 0.9, timestamp=t)
        sm.update([bag], {1: rec_far}, timestamp=t)

    # Bag is suddenly moved by 80px (e.g. someone picks it up)
    bag.update_detection(np.array([180, 180, 210, 210]), 0.9, timestamp=6.0)
    st_moved, _ = sm.update([bag], {1: rec_far}, timestamp=6.0)

    assert st_moved[1].current_state == AbandonmentState.RECOVERED
    assert st_moved[1].separation_start_time is None


def test_occlusion_tolerance():
    sm = TemporalStateMachine(
        occlusion_grace_period=2.0,
    )

    bag = Track(1, 1, "bag", np.array([100, 100, 130, 130]), 0.9, last_timestamp=0.0)
    rec = BagOwnershipRecord(bag_id=1, is_owner_confirmed=True, last_owner_distance=100.0)

    # t=1.0s: bag visible
    states, _ = sm.update([bag], {1: rec}, timestamp=1.0)
    assert 1 in states

    # t=2.0s: bag occluded (empty bag_tracks list)
    states, _ = sm.update([], {}, timestamp=2.0)
    # Bag state maintained during grace period
    assert 1 in states
    assert states[1].is_occluded is True

    # t=4.5s: occlusion exceeds 2.0s grace period -> bag cleaned up
    states, _ = sm.update([], {}, timestamp=4.5)
    assert 1 not in states
