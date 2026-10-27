"""ByteTrack implementation with class-consistent tracking and velocity estimation."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from ..detection.detector import Detection
from .tracker import BaseTracker
from .track_state import Track

logger = logging.getLogger("AbandonedObjectDetection.ByteTrack")


def calculate_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute Intersection over Union (IoU) between two boxes [x1, y1, x2, y2]."""
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    union_area = box_a_area + box_b_area - intersection_area
    if union_area <= 0:
        return 0.0

    return float(intersection_area / union_area)


def compute_iou_cost_matrix(tracks: List[Track], detections: List[Detection]) -> np.ndarray:
    """Compute 1 - IoU cost matrix between tracks and detections."""
    if not tracks or not detections:
        return np.empty((len(tracks), len(detections)), dtype=np.float32)

    cost_matrix = np.ones((len(tracks), len(detections)), dtype=np.float32)
    for i, trk in enumerate(tracks):
        for j, det in enumerate(detections):
            iou = calculate_iou(trk.bbox, det.bbox)
            cost_matrix[i, j] = 1.0 - iou
    return cost_matrix


class ByteTracker(BaseTracker):
    """Class-consistent ByteTrack algorithm."""

    def __init__(
        self,
        track_thresh: float = 0.35,
        match_thresh: float = 0.80,
        track_buffer: int = 60,
        min_box_area: float = 100.0,
        frame_rate: int = 30,
    ):
        """Initialize ByteTracker.

        Args:
            track_thresh: High-confidence detection threshold.
            match_thresh: IoU matching threshold (max 1 - IoU distance).
            track_buffer: Max missed frames before dropping track.
            min_box_area: Minimum bbox area to track.
            frame_rate: Video frame rate.
        """
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.min_box_area = min_box_area
        self.frame_rate = frame_rate

        self.next_id = 1
        self.tracked_tracks: List[Track] = []
        self.lost_tracks: List[Track] = []
        self.frame_count = 0

    def reset(self):
        """Reset tracker state."""
        self.next_id = 1
        self.tracked_tracks.clear()
        self.lost_tracks.clear()
        self.frame_count = 0

    def _associate_class(
        self,
        tracks: List[Track],
        dets_high: List[Detection],
        dets_low: List[Detection],
        timestamp: float,
    ) -> Tuple[List[Track], List[Track], List[Detection]]:
        """Perform two-stage association for a single object class."""
        matched_tracks: List[Track] = []
        unmatched_tracks: List[Track] = []
        unmatched_high_dets: List[Detection] = []

        # --- Stage 1: Associate with High-Confidence Detections ---
        if tracks and dets_high:
            cost_matrix = compute_iou_cost_matrix(tracks, dets_high)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            matched_trk_idx = set()
            matched_det_idx = set()

            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < self.match_thresh:
                    trk = tracks[r]
                    det = dets_high[c]
                    trk.update_detection(det.bbox, det.confidence, timestamp)
                    matched_tracks.append(trk)
                    matched_trk_idx.add(r)
                    matched_det_idx.add(c)

            unmatched_tracks = [t for i, t in enumerate(tracks) if i not in matched_trk_idx]
            unmatched_high_dets = [d for j, d in enumerate(dets_high) if j not in matched_det_idx]
        else:
            unmatched_tracks = list(tracks)
            unmatched_high_dets = list(dets_high)

        # --- Stage 2: Associate Unmatched Tracks with Low-Confidence Detections ---
        still_unmatched_tracks: List[Track] = []
        if unmatched_tracks and dets_low:
            cost_matrix_low = compute_iou_cost_matrix(unmatched_tracks, dets_low)
            row_ind, col_ind = linear_sum_assignment(cost_matrix_low)

            matched_trk_idx = set()
            for r, c in zip(row_ind, col_ind):
                # Slightly looser IoU threshold for second stage
                if cost_matrix_low[r, c] < (self.match_thresh + 0.1):
                    trk = unmatched_tracks[r]
                    det = dets_low[c]
                    trk.update_detection(det.bbox, det.confidence, timestamp)
                    matched_tracks.append(trk)
                    matched_trk_idx.add(r)

            still_unmatched_tracks = [
                t for i, t in enumerate(unmatched_tracks) if i not in matched_trk_idx
            ]
        else:
            still_unmatched_tracks = list(unmatched_tracks)

        return matched_tracks, still_unmatched_tracks, unmatched_high_dets

    def update(self, detections: List[Detection], timestamp: float) -> List[Track]:
        """Update tracker with frame detections.

        Args:
            detections: List of current frame detections.
            timestamp: Current timestamp in seconds.

        Returns:
            List of active Track objects.
        """
        self.frame_count += 1

        # Filter detections by area
        valid_dets = [d for d in detections if d.area >= self.min_box_area]

        # Separate detections and existing tracks by class (0: person, 1: bag)
        active_classes = {0, 1}
        all_updated_tracks: List[Track] = []

        for cls_id in active_classes:
            cls_tracks = [t for t in self.tracked_tracks if t.class_id == cls_id]
            effective_thresh = min(self.track_thresh, 0.20) if cls_id == 1 else self.track_thresh
            cls_dets_high = [
                d for d in valid_dets if d.class_id == cls_id and d.confidence >= effective_thresh
            ]
            cls_dets_low = [
                d for d in valid_dets if d.class_id == cls_id and 0.05 <= d.confidence < effective_thresh
            ]

            matched, unmatched_trks, new_dets = self._associate_class(
                cls_tracks, cls_dets_high, cls_dets_low, timestamp
            )

            # Update tracks that were matched
            all_updated_tracks.extend(matched)

            # Handle unmatched tracks (mark missed)
            for trk in unmatched_trks:
                trk.mark_missed()
                if trk.time_since_update <= self.track_buffer:
                    all_updated_tracks.append(trk)

            # Initialize new tracks for unmatched high confidence detections
            for det in new_dets:
                new_track = Track(
                    track_id=self.next_id,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    last_timestamp=timestamp,
                )
                self.next_id += 1
                all_updated_tracks.append(new_track)

        self.tracked_tracks = all_updated_tracks
        # Return only currently active tracks (not long-missed)
        return [t for t in self.tracked_tracks if t.time_since_update <= 5]
