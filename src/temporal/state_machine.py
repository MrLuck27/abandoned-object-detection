"""Temporal State Machine for unattended object detection and reason verification."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..tracking.track_state import Track
from ..association.owner_associator import BagOwnershipRecord
from .states import AbandonmentState, StateTransitionEvent, AbandonmentAlert

logger = logging.getLogger("AbandonedObjectDetection.StateMachine")


@dataclass
class BagTrackState:
    """Internal state tracked for each bag."""

    bag_id: int
    current_state: AbandonmentState = AbandonmentState.UNKNOWN
    state_entered_time: float = 0.0
    separation_start_time: Optional[float] = None
    separation_duration: float = 0.0
    is_stationary: bool = True
    last_known_owner_id: Optional[int] = None
    last_seen_time: float = 0.0
    is_occluded: bool = False
    occlusion_start_time: Optional[float] = None
    alert_raised: bool = False
    last_alert_time: float = -9999.0
    displacement: float = 0.0
    current_owner_distance: float = 0.0


class TemporalStateMachine:
    """Deterministic, conservative temporal state machine for unattended object reasoning."""

    def __init__(
        self,
        abandonment_time_threshold: float = 15.0,
        owner_distance_threshold: float = 180.0,
        owner_moving_away_speed: float = 1.5,
        object_movement_threshold: float = 25.0,
        stationary_window_seconds: float = 3.0,
        occlusion_grace_period: float = 3.0,
        owner_return_distance: float = 120.0,
        alert_cooldown_seconds: float = 30.0,
    ):
        """Initialize Temporal State Machine.

        Args:
            abandonment_time_threshold: Duration in seconds to confirm ABANDONED.
            owner_distance_threshold: Distance in pixels above which owner is separated.
            owner_moving_away_speed: Speed in px/s indicating owner departure.
            object_movement_threshold: Max displacement in pixels to consider stationary.
            stationary_window_seconds: Time window for stationary verification.
            occlusion_grace_period: Seconds to tolerate missed detections before dropping track.
            owner_return_distance: Distance below which returning owner restores ATTENDED state.
            alert_cooldown_seconds: Seconds between alerts for the same bag.
        """
        self.abandonment_time_threshold = abandonment_time_threshold
        self.owner_distance_threshold = owner_distance_threshold
        self.owner_moving_away_speed = owner_moving_away_speed
        self.object_movement_threshold = object_movement_threshold
        self.stationary_window_seconds = stationary_window_seconds
        self.occlusion_grace_period = occlusion_grace_period
        self.owner_return_distance = owner_return_distance
        self.alert_cooldown_seconds = alert_cooldown_seconds

        self.bag_states: Dict[int, BagTrackState] = {}
        self.transition_history: List[StateTransitionEvent] = []
        self.active_alerts: List[AbandonmentAlert] = []

    def reset(self):
        """Reset state machine."""
        self.bag_states.clear()
        self.transition_history.clear()
        self.active_alerts.clear()

    def _transition(
        self,
        state_obj: BagTrackState,
        new_state: AbandonmentState,
        timestamp: float,
        reason: str,
    ):
        """Transition object to new state and log event."""
        if state_obj.current_state == new_state:
            return

        event = StateTransitionEvent(
            timestamp=timestamp,
            bag_id=state_obj.bag_id,
            owner_id=state_obj.last_known_owner_id,
            previous_state=state_obj.current_state,
            new_state=new_state,
            distance=state_obj.current_owner_distance,
            reason=reason,
        )
        self.transition_history.append(event)
        logger.info(
            f"[t={timestamp:.1f}s] Bag {state_obj.bag_id}: {state_obj.current_state.value} -> "
            f"{new_state.value} (Owner: {state_obj.last_known_owner_id}, Dist: {state_obj.current_owner_distance:.1f}px) "
            f"Reason: {reason}"
        )
        state_obj.current_state = new_state
        state_obj.state_entered_time = timestamp

    def update(
        self,
        bag_tracks: List[Track],
        ownership_records: Dict[int, BagOwnershipRecord],
        timestamp: float,
    ) -> Tuple[Dict[int, BagTrackState], List[AbandonmentAlert]]:
        """Update temporal state for all bags and check for alerts.

        Args:
            bag_tracks: Active bag tracks in current frame.
            ownership_records: Ownership records from OwnerAssociator.
            timestamp: Video timestamp in seconds.

        Returns:
            Tuple of (bag_states_dict, new_alerts_in_this_frame).
        """
        active_bag_map = {b.track_id: b for b in bag_tracks}
        new_alerts: List[AbandonmentAlert] = []

        # Handle active and occluded bags
        all_bag_ids = set(self.bag_states.keys()).union(set(active_bag_map.keys()))

        for bid in list(all_bag_ids):
            if bid not in self.bag_states:
                self.bag_states[bid] = BagTrackState(
                    bag_id=bid,
                    current_state=AbandonmentState.UNKNOWN,
                    state_entered_time=timestamp,
                    last_seen_time=timestamp,
                )

            st = self.bag_states[bid]

            # Check if bag is missing in this frame (potential occlusion)
            if bid not in active_bag_map:
                if not st.is_occluded:
                    st.is_occluded = True
                    st.occlusion_start_time = timestamp

                # If occluded longer than grace period, clean up
                occlusion_dur = timestamp - (st.occlusion_start_time or timestamp)
                if occlusion_dur > self.occlusion_grace_period:
                    logger.info(f"Bag {bid} lost longer than grace period ({occlusion_dur:.1f}s). Removing.")
                    del self.bag_states[bid]
                continue
            else:
                # Bag is actively visible
                st.is_occluded = False
                st.occlusion_start_time = None
                st.last_seen_time = timestamp

            bag = active_bag_map[bid]
            rec = ownership_records.get(bid)

            # 1. Update stationary status
            displacement = bag.calculate_displacement(window_seconds=self.stationary_window_seconds)
            is_stationary = displacement < self.object_movement_threshold
            st.displacement = displacement
            st.is_stationary = is_stationary

            # 2. Update owner information
            if rec is not None:
                if rec.is_owner_confirmed and rec.confirmed_owner_id is not None:
                    st.last_known_owner_id = rec.confirmed_owner_id
                st.current_owner_distance = rec.last_owner_distance
                owner_speed = rec.owner_relative_speed
            else:
                owner_speed = 0.0

            owner_id = st.last_known_owner_id
            owner_dist = st.current_owner_distance

            # --- State Transition Logic ---

            # Case: Object is NOT stationary (bag moved or was picked up)
            if not is_stationary:
                st.separation_start_time = None
                st.separation_duration = 0.0
                if st.current_state in [
                    AbandonmentState.SEPARATED,
                    AbandonmentState.SUSPECTED_ABANDONED,
                    AbandonmentState.ABANDONED,
                ]:
                    self._transition(st, AbandonmentState.RECOVERED, timestamp, "Bag moved/displaced")
                else:
                    self._transition(st, AbandonmentState.ATTENDED if owner_id else AbandonmentState.UNKNOWN, timestamp, "Bag in motion")
                continue

            # Case: Returning Owner Check
            if (
                owner_id is not None
                and owner_dist <= self.owner_return_distance
                and st.current_state in [
                    AbandonmentState.OWNER_MOVING_AWAY,
                    AbandonmentState.SEPARATED,
                    AbandonmentState.SUSPECTED_ABANDONED,
                    AbandonmentState.ABANDONED,
                ]
            ):
                st.separation_start_time = None
                st.separation_duration = 0.0
                self._transition(st, AbandonmentState.RECOVERED, timestamp, f"Owner returned (dist={owner_dist:.1f}px)")
                continue

            # If currently RECOVERED and owner is still close, return to ATTENDED
            if st.current_state == AbandonmentState.RECOVERED:
                if owner_dist <= self.owner_return_distance:
                    self._transition(st, AbandonmentState.ATTENDED, timestamp, "Owner attended")
                continue

            # Case: Initial state UNKNOWN
            if st.current_state == AbandonmentState.UNKNOWN:
                if owner_id is not None and owner_dist <= self.owner_distance_threshold:
                    self._transition(st, AbandonmentState.ATTENDED, timestamp, f"Owner P{owner_id:02d} confirmed nearby")
                elif owner_dist > self.owner_distance_threshold:
                    # Unattended object detected with no owner nearby
                    self._transition(st, AbandonmentState.SEPARATED, timestamp, "Object detected without nearby owner")
                    st.separation_start_time = timestamp

            # Case: ATTENDED state
            elif st.current_state == AbandonmentState.ATTENDED:
                if owner_dist > self.owner_distance_threshold:
                    self._transition(st, AbandonmentState.SEPARATED, timestamp, f"Owner separated (dist={owner_dist:.1f}px)")
                    st.separation_start_time = timestamp
                elif owner_speed > self.owner_moving_away_speed and owner_dist > self.owner_return_distance:
                    self._transition(st, AbandonmentState.OWNER_MOVING_AWAY, timestamp, f"Owner moving away (v_rel={owner_speed:.1f}px/s)")

            # Case: OWNER_MOVING_AWAY state
            elif st.current_state == AbandonmentState.OWNER_MOVING_AWAY:
                if owner_dist > self.owner_distance_threshold:
                    self._transition(st, AbandonmentState.SEPARATED, timestamp, f"Owner reached separation threshold (dist={owner_dist:.1f}px)")
                    st.separation_start_time = timestamp
                elif owner_dist <= self.owner_return_distance:
                    self._transition(st, AbandonmentState.ATTENDED, timestamp, "Owner returned close")

            # Case: SEPARATED, SUSPECTED_ABANDONED, or ABANDONED states
            if st.current_state in [
                AbandonmentState.SEPARATED,
                AbandonmentState.SUSPECTED_ABANDONED,
                AbandonmentState.ABANDONED,
            ]:
                if st.separation_start_time is None:
                    st.separation_start_time = timestamp

                st.separation_duration = max(0.0, timestamp - st.separation_start_time)

                # Timer progression
                if st.separation_duration >= self.abandonment_time_threshold:
                    if st.current_state != AbandonmentState.ABANDONED:
                        self._transition(
                            st,
                            AbandonmentState.ABANDONED,
                            timestamp,
                            f"Separation duration ({st.separation_duration:.1f}s) >= threshold ({self.abandonment_time_threshold}s)",
                        )

                    # Trigger alert if cooldown passed
                    if (timestamp - st.last_alert_time) >= self.alert_cooldown_seconds:
                        st.alert_raised = True
                        st.last_alert_time = timestamp
                        alert = AbandonmentAlert(
                            timestamp=timestamp,
                            bag_id=bid,
                            owner_id=owner_id,
                            state=AbandonmentState.ABANDONED,
                            separation_duration=st.separation_duration,
                            distance=owner_dist,
                            bag_bbox=bag.bbox.tolist(),
                            message=f"ABANDONED OBJECT: Bag B{bid:02d} left unattended for {st.separation_duration:.1f}s by Owner P{owner_id or 0:02d}",
                        )
                        new_alerts.append(alert)
                        self.active_alerts.append(alert)

                elif st.separation_duration >= (0.5 * self.abandonment_time_threshold):
                    if st.current_state == AbandonmentState.SEPARATED:
                        self._transition(
                            st,
                            AbandonmentState.SUSPECTED_ABANDONED,
                            timestamp,
                            f"Separation timer at {st.separation_duration:.1f}s (half threshold)",
                        )

        return self.bag_states, new_alerts
