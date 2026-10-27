"""Temporal Owner-Bag Association Engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..tracking.track_state import Track
from .spatial_metrics import (
    euclidean_distance,
    motion_cosine_similarity,
)


@dataclass
class CandidateOwnerStats:
    """Statistics accumulated for a candidate person track near a bag."""

    person_id: int
    first_seen_timestamp: float
    last_seen_timestamp: float
    proximity_frames: int = 0
    distance_samples: List[float] = field(default_factory=list)
    motion_sim_samples: List[float] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.last_seen_timestamp - self.first_seen_timestamp)

    @property
    def average_distance(self) -> float:
        if not self.distance_samples:
            return 9999.0
        return sum(self.distance_samples) / len(self.distance_samples)

    @property
    def average_motion_similarity(self) -> float:
        if not self.motion_sim_samples:
            return 0.5
        return sum(self.motion_sim_samples) / len(self.motion_sim_samples)


@dataclass
class BagOwnershipRecord:
    """State of ownership tracking for a single bag."""

    bag_id: int
    confirmed_owner_id: Optional[int] = None
    is_owner_confirmed: bool = False
    confirmation_timestamp: Optional[float] = None
    last_owner_distance: float = 0.0
    owner_relative_speed: float = 0.0  # Positive means moving away
    is_owner_present_in_frame: bool = False
    owner_centroid: Optional[np.ndarray] = None
    candidates: Dict[int, CandidateOwnerStats] = field(default_factory=dict)
    owner_distance_history: List[Tuple[float, float]] = field(default_factory=list)


class OwnerAssociator:
    """Performs temporal owner-bag association using spatial proximity, duration, and motion."""

    def __init__(
        self,
        weight_distance: float = 0.45,
        weight_temporal: float = 0.35,
        weight_motion: float = 0.20,
        max_association_distance: float = 250.0,
        min_attended_duration: float = 1.0,
        min_association_score: float = 0.40,
        candidate_history_len: int = 45,
    ):
        """Initialize OwnerAssociator.

        Args:
            weight_distance: Importance of Euclidean proximity.
            weight_temporal: Importance of duration person stays close.
            weight_motion: Importance of correlated movement when arriving.
            max_association_distance: Distance threshold to consider a person a candidate owner.
            min_attended_duration: Minimum seconds of proximity before confirming owner.
            min_association_score: Threshold composite score to confirm ownership.
            candidate_history_len: Maximum frames of candidate history to retain.
        """
        self.w_dist = weight_distance
        self.w_temp = weight_temporal
        self.w_motion = weight_motion
        self.max_dist = max_association_distance
        self.min_attended_dur = min_attended_duration
        self.min_score = min_association_score
        self.history_len = candidate_history_len

        # Map from bag_id to BagOwnershipRecord
        self.bag_records: Dict[int, BagOwnershipRecord] = {}

    def reset(self):
        """Reset all ownership records."""
        self.bag_records.clear()

    def calculate_candidate_score(
        self, stats: CandidateOwnerStats, current_dist: float
    ) -> Tuple[float, float, float, float]:
        """Compute the composite ownership score and individual components.

        Returns:
            (total_score, dist_score, temp_score, motion_score)
        """
        # Distance score: closer is higher [0..1]
        dist_score = max(0.0, 1.0 - (current_dist / self.max_dist))

        # Temporal proximity score: sustained proximity over time [0..1]
        temp_score = min(1.0, stats.duration / max(self.min_attended_dur, 0.1))

        # Motion consistency score: co-movement [0..1]
        motion_score = max(0.0, stats.average_motion_similarity)

        total_score = (
            self.w_dist * dist_score
            + self.w_temp * temp_score
            + self.w_motion * motion_score
        )
        return float(total_score), float(dist_score), float(temp_score), float(motion_score)

    def update(
        self,
        person_tracks: List[Track],
        bag_tracks: List[Track],
        timestamp: float,
    ) -> Dict[int, BagOwnershipRecord]:
        """Update association for all bags given current frame's tracks.

        Args:
            person_tracks: Active person tracks.
            bag_tracks: Active bag tracks.
            timestamp: Video timestamp in seconds.

        Returns:
            Dictionary mapping bag_id to BagOwnershipRecord.
        """
        person_map = {p.track_id: p for p in person_tracks}

        for bag in bag_tracks:
            bid = bag.track_id
            if bid not in self.bag_records:
                self.bag_records[bid] = BagOwnershipRecord(bag_id=bid)

            record = self.bag_records[bid]

            # Case A: Owner not yet confirmed -> evaluate candidate persons
            if not record.is_owner_confirmed:
                best_candidate_id: Optional[int] = None
                best_score = -1.0

                for person in person_tracks:
                    pid = person.track_id
                    dist = euclidean_distance(bag.center, person.center)

                    if dist <= self.max_dist:
                        # Candidate is in proximity
                        if pid not in record.candidates:
                            record.candidates[pid] = CandidateOwnerStats(
                                person_id=pid,
                                first_seen_timestamp=timestamp,
                                last_seen_timestamp=timestamp,
                            )

                        cand = record.candidates[pid]
                        cand.last_seen_timestamp = timestamp
                        cand.proximity_frames += 1
                        cand.distance_samples.append(dist)
                        if len(cand.distance_samples) > self.history_len:
                            cand.distance_samples.pop(0)

                        motion_sim = motion_cosine_similarity(bag.velocity, person.velocity)
                        cand.motion_sim_samples.append(motion_sim)
                        if len(cand.motion_sim_samples) > self.history_len:
                            cand.motion_sim_samples.pop(0)

                        tot_score, _, _, _ = self.calculate_candidate_score(cand, dist)
                        if tot_score > best_score:
                            best_score = tot_score
                            best_candidate_id = pid

                # If highest scoring candidate meets confirmation threshold
                if (
                    best_candidate_id is not None
                    and best_score >= self.min_score
                    and record.candidates[best_candidate_id].duration >= self.min_attended_dur
                ):
                    record.confirmed_owner_id = best_candidate_id
                    record.is_owner_confirmed = True
                    record.confirmation_timestamp = timestamp

            # Case B: Owner is confirmed -> monitor distance and relative motion
            if record.is_owner_confirmed and record.confirmed_owner_id is not None:
                oid = record.confirmed_owner_id
                if oid in person_map:
                    # Owner is visible in current frame
                    owner_track = person_map[oid]
                    record.is_owner_present_in_frame = True
                    record.owner_centroid = owner_track.center

                    curr_dist = euclidean_distance(bag.center, owner_track.center)

                    # Compute relative separation speed (rate of distance increase)
                    if record.owner_distance_history:
                        prev_time, prev_dist = record.owner_distance_history[-1]
                        dt = max(timestamp - prev_time, 1e-4)
                        record.owner_relative_speed = (curr_dist - prev_dist) / dt
                    else:
                        record.owner_relative_speed = 0.0

                    record.last_owner_distance = curr_dist
                    record.owner_distance_history.append((timestamp, curr_dist))
                    if len(record.owner_distance_history) > 100:
                        record.owner_distance_history.pop(0)
                else:
                    # Owner walked out of frame -> definitely separated
                    record.is_owner_present_in_frame = False
                    # Retain last distance or designate as far out of view
                    record.last_owner_distance = max(record.last_owner_distance, 999.0)
                    record.owner_relative_speed = 0.0

            elif not record.is_owner_confirmed:
                # If no owner confirmed yet, find nearest person if any
                if person_tracks:
                    dists = [euclidean_distance(bag.center, p.center) for p in person_tracks]
                    min_idx = int(np.argmin(dists))
                    record.last_owner_distance = float(dists[min_idx])
                    record.owner_centroid = person_tracks[min_idx].center
                else:
                    record.last_owner_distance = 999.0
                    record.owner_centroid = None

        return self.bag_records
