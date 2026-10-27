"""Unit tests for abandonment timer accumulation."""

import numpy as np
import pytest
from src.tracking.track_state import Track
from src.association.owner_associator import BagOwnershipRecord
from src.temporal.state_machine import TemporalStateMachine
from src.temporal.states import AbandonmentState


def test_abandonment_timer_accumulation():
    sm = TemporalStateMachine(
        abandonment_time_threshold=10.0,
        owner_distance_threshold=150.0,
    )

    bag = Track(1, 1, "bag", np.array([100, 100, 130, 130]), 0.9, last_timestamp=0.0)

    # Ownership record showing owner is 300px away (separated)
    rec = BagOwnershipRecord(
        bag_id=1,
        confirmed_owner_id=5,
        is_owner_confirmed=True,
        last_owner_distance=300.0,
        is_owner_present_in_frame=True,
    )

    # Step through 5 seconds
    for step in range(50):
        t = step * 0.1
        bag.update_detection(bag.bbox, 0.9, t)
        states, _ = sm.update([bag], {1: rec}, timestamp=t)

    st = states[1]
    assert st.separation_start_time is not None
    assert np.isclose(st.separation_duration, 4.9, atol=0.2)
    assert st.current_state in [AbandonmentState.SEPARATED, AbandonmentState.SUSPECTED_ABANDONED]
