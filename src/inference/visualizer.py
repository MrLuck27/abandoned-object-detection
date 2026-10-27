"""Visualizer for overlaying tracks, associations, state badges, and alerts."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from ..tracking.track_state import Track
from ..association.owner_associator import BagOwnershipRecord
from ..temporal.states import AbandonmentState, AbandonmentAlert
from ..temporal.state_machine import BagTrackState


# BGR Color Palette
COLORS = {
    "person": (255, 140, 0),          # Cyan / Light Blue
    AbandonmentState.UNKNOWN: (180, 180, 180),           # Gray
    AbandonmentState.ATTENDED: (50, 205, 50),            # Lime Green
    AbandonmentState.OWNER_MOVING_AWAY: (0, 165, 255),   # Orange
    AbandonmentState.SEPARATED: (0, 215, 255),           # Amber Yellow
    AbandonmentState.STATIONARY: (0, 215, 255),          # Amber
    AbandonmentState.SUSPECTED_ABANDONED: (0, 69, 255),  # Red-Orange
    AbandonmentState.ABANDONED: (0, 0, 255),             # Bright Red
    AbandonmentState.RECOVERED: (255, 191, 0),           # Deep Sky Blue
}


class PipelineVisualizer:
    """Renders professional surveillance overlays on OpenCV video frames."""

    def __init__(
        self,
        show_boxes: bool = True,
        show_ids: bool = True,
        show_association: bool = True,
        show_states: bool = True,
        show_timer: bool = True,
        show_fps: bool = True,
    ):
        self.show_boxes = show_boxes
        self.show_ids = show_ids
        self.show_association = show_association
        self.show_states = show_states
        self.show_timer = show_timer
        self.show_fps = show_fps

    def draw_rounded_rectangle(
        self,
        img: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int = 2,
    ):
        """Draw bounding box with corner highlights."""
        x1, y1 = pt1
        x2, y2 = pt2
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Highlight corners
        corner_len = min(15, (x2 - x1) // 4, (y2 - y1) // 4)
        if corner_len > 3:
            thick = thickness + 2
            cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, thick)
            cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, thick)
            cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, thick)
            cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, thick)
            cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, thick)
            cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, thick)
            cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, thick)
            cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, thick)

    def draw_badge(
        self,
        img: np.ndarray,
        text: str,
        org: Tuple[int, int],
        bg_color: Tuple[int, int, int],
        text_color: Tuple[int, int, int] = (255, 255, 255),
        font_scale: float = 0.5,
        thickness: int = 1,
    ):
        """Draw a text badge with filled background rectangle."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x, y = org
        y_top = max(0, y - h - 6)
        cv2.rectangle(img, (x, y_top), (x + w + 8, y + baseline), bg_color, -1)
        cv2.putText(img, text, (x + 4, y - 2), font, font_scale, text_color, thickness, cv2.LINE_AA)

    def render(
        self,
        frame: np.ndarray,
        person_tracks: List[Track],
        bag_tracks: List[Track],
        ownership_records: Dict[int, BagOwnershipRecord],
        bag_states: Dict[int, BagTrackState],
        active_alerts: List[AbandonmentAlert],
        fps: float = 0.0,
        latency_ms: float = 0.0,
    ) -> np.ndarray:
        """Render all detection, tracking, association, state and alert overlays.

        Args:
            frame: Input BGR frame.
            person_tracks: Active person tracks.
            bag_tracks: Active bag tracks.
            ownership_records: Dict of BagOwnershipRecord.
            bag_states: Dict of BagTrackState.
            active_alerts: Current frame alerts.
            fps: Current processing frames per second.
            latency_ms: Latency in milliseconds.

        Returns:
            Annotated BGR image frame.
        """
        out = frame.copy()
        h, w = out.shape[:2]

        # 1. Draw Person Tracks
        for p in person_tracks:
            x1, y1, x2, y2 = map(int, p.bbox)
            if self.show_boxes:
                self.draw_rounded_rectangle(out, (x1, y1), (x2, y2), COLORS["person"], 2)
            if self.show_ids:
                label = f"P{p.track_id:02d} ({p.confidence:.2f})"
                self.draw_badge(out, label, (x1, y1), COLORS["person"], (0, 0, 0))

        # 2. Draw Bag Tracks and Associated Owners
        for b in bag_tracks:
            bid = b.track_id
            state_info = bag_states.get(bid)
            state = state_info.current_state if state_info else AbandonmentState.UNKNOWN
            color = COLORS.get(state, (0, 255, 255))

            bx1, by1, bx2, by2 = map(int, b.bbox)
            bcx, bcy = int(b.center[0]), int(b.center[1])

            # Draw Bag Bounding Box
            if self.show_boxes:
                thick = 3 if state == AbandonmentState.ABANDONED else 2
                self.draw_rounded_rectangle(out, (bx1, by1), (bx2, by2), color, thick)

            # Draw Owner Link & Distance Vector
            rec = ownership_records.get(bid)
            if rec and self.show_association and rec.owner_centroid is not None:
                ocx, ocy = int(rec.owner_centroid[0]), int(rec.owner_centroid[1])
                # Draw connecting dashed or colored line
                cv2.line(out, (bcx, bcy), (ocx, ocy), color, 2, cv2.LINE_AA)

                # Distance text mid-line
                mid_x = (bcx + ocx) // 2
                mid_y = (bcy + ocy) // 2
                dist_txt = f"{rec.last_owner_distance:.0f}px"
                self.draw_badge(out, dist_txt, (mid_x, mid_y), (40, 40, 40), (255, 255, 255), 0.45)

            # Draw State Badge and Timer on Bag
            owner_label = f"Owner: P{rec.confirmed_owner_id:02d}" if (rec and rec.is_owner_confirmed) else "Owner: Looking..."
            timer_text = ""
            if state_info and state_info.separation_start_time is not None and self.show_timer:
                timer_text = f" | Timer: {state_info.separation_duration:.1f}s"

            main_label = f"Bag B{bid:02d} [{state.value}]"
            sub_label = f"{owner_label}{timer_text}"

            self.draw_badge(out, main_label, (bx1, by1), color, (255, 255, 255) if state == AbandonmentState.ABANDONED else (0, 0, 0))
            self.draw_badge(out, sub_label, (bx1, by2 + 18), (30, 30, 30), (220, 220, 220), 0.42)

        # 3. Draw Alert Banner if any object is ABANDONED
        abandoned_bags = [bid for bid, st in bag_states.items() if st.current_state == AbandonmentState.ABANDONED]
        if abandoned_bags:
            banner_h = 55
            overlay = out.copy()
            cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 200), -1)
            cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)

            bag_ids_str = ", ".join([f"B{bid:02d}" for bid in abandoned_bags])
            alert_text = f"ALERT: UNATTENDED OBJECT DETECTED! Bag(s): {bag_ids_str}"
            cv2.putText(out, alert_text, (20, 36), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

        # 4. Top Information Bar (FPS, Latency, Stats)
        if self.show_fps:
            stat_bar = f"FPS: {fps:.1f} | Latency: {latency_ms:.1f}ms | Persons: {len(person_tracks)} | Bags: {len(bag_tracks)}"
            self.draw_badge(out, stat_bar, (w - 380, 24), (20, 20, 20), (0, 255, 255), 0.48)

        return out
