"""Unit tests for alert generation and cooldown behavior."""

import numpy as np
import pytest
from src.tracking.track_state import Track
from src.association.owner_associator import BagOwnershipRecord
from src.temporal.state_machine import TemporalStateMachine
from src.temporal.states import AbandonmentState


def test_alert_generation_and_cooldown():
    sm = TemporalStateMachine(
        abandonment_time_threshold=5.0,
        owner_distance_threshold=100.0,
        alert_cooldown_seconds=10.0,
    )

    bag = Track(1, 1, "bag", np.array([100, 100, 130, 130]), 0.9, last_timestamp=0.0)
    rec_far = BagOwnershipRecord(
        bag_id=1, confirmed_owner_id=2, is_owner_confirmed=True, last_owner_distance=200.0
    )

    # t=0s: separated
    sm.update([bag], {1: rec_far}, timestamp=0.0)

    # t=5.1s: triggers first alert
    _, alerts1 = sm.update([bag], {1: rec_far}, timestamp=5.1)
    assert len(alerts1) == 1
    assert alerts1[0].state == AbandonmentState.ABANDONED

    # t=6.0s: cooldown active (10s), so NO new alert should be raised
    _, alerts2 = sm.update([bag], {1: rec_far}, timestamp=6.0)
    assert len(alerts2) == 0

    # t=16.0s: cooldown expired (5.1 + 10 = 15.1s), new alert should be raised
    _, alerts3 = sm.update([bag], {1: rec_far}, timestamp=16.0)
    assert len(alerts3) == 1
