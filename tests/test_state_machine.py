"""Unit tests for state transitions across the full pipeline."""

import numpy as np
import pytest
from src.tracking.track_state import Track
from src.association.owner_associator import BagOwnershipRecord
from src.temporal.state_machine import TemporalStateMachine
from src.temporal.states import AbandonmentState


def test_state_transitions_pipeline():
    sm = TemporalStateMachine(
        abandonment_time_threshold=10.0,
        owner_distance_threshold=150.0,
        owner_return_distance=100.0,
    )

    bag = Track(1, 1, "bag", np.array([100, 100, 130, 130]), 0.9, last_timestamp=0.0)

    # 1. Initially owner is close (50px) -> ATTENDED
    rec_close = BagOwnershipRecord(
        bag_id=1, confirmed_owner_id=1, is_owner_confirmed=True, last_owner_distance=50.0
    )
    states, alerts = sm.update([bag], {1: rec_close}, timestamp=1.0)
    assert states[1].current_state == AbandonmentState.ATTENDED
    assert len(alerts) == 0

    # 2. Owner moves away (250px) -> SEPARATED
    rec_far = BagOwnershipRecord(
        bag_id=1, confirmed_owner_id=1, is_owner_confirmed=True, last_owner_distance=250.0
    )
    states, alerts = sm.update([bag], {1: rec_far}, timestamp=2.0)
    assert states[1].current_state == AbandonmentState.SEPARATED
    assert len(alerts) == 0

    # 3. 6 seconds later (t=8.0s) -> SUSPECTED_ABANDONED (> 50% of 10s)
    states, alerts = sm.update([bag], {1: rec_far}, timestamp=8.0)
    assert states[1].current_state == AbandonmentState.SUSPECTED_ABANDONED

    # 4. 11 seconds of separation (t=13.0s) -> ABANDONED
    states, alerts = sm.update([bag], {1: rec_far}, timestamp=13.0)
    assert states[1].current_state == AbandonmentState.ABANDONED
    assert len(alerts) == 1
    assert alerts[0].bag_id == 1
