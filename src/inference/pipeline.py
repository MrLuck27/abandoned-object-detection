"""End-to-End Surveillance Pipeline orchestrating Detection, Tracking, Association, and Temporal Reasoning."""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import cv2
import numpy as np

from ..detection.detector import BaseDetector, Detection
from ..detection.yolo_detector import YOLODetector
from ..tracking.tracker import BaseTracker
from ..tracking.byte_tracker import ByteTracker
from ..tracking.track_state import Track
from ..association.owner_associator import OwnerAssociator, BagOwnershipRecord
from ..temporal.state_machine import TemporalStateMachine, BagTrackState
from ..temporal.states import AbandonmentAlert
from ..utils.video_utils import VideoReader, VideoWriter
from ..utils.power_monitor import PowerMonitor
from .visualizer import PipelineVisualizer

logger = logging.getLogger("AbandonedObjectDetection.Pipeline")


@dataclass
class PipelineFrameResult:
    """Output for a single processed frame."""

    frame_index: int
    timestamp: float
    annotated_frame: np.ndarray
    person_tracks: List[Track]
    bag_tracks: List[Track]
    ownership_records: Dict[int, BagOwnershipRecord]
    bag_states: Dict[int, BagTrackState]
    alerts: List[AbandonmentAlert]
    fps: float
    latency_ms: float
    hardware_metrics: Dict[str, any]


class SurveillancePipeline:
    """Modular End-to-End Pipeline for Unattended Object Detection."""

    def __init__(self, config: Dict[str, any]):
        """Initialize pipeline components from configuration dictionary."""
        self.config = config

        # 1. Detector
        det_cfg = config.get("detection", {})
        self.detector: BaseDetector = YOLODetector(
            model_path=det_cfg.get("model_path", "yolov8n.pt"),
            confidence_threshold=det_cfg.get("confidence_threshold", 0.35),
            iou_threshold=det_cfg.get("iou_threshold", 0.45),
            device=det_cfg.get("device", "0"),
            coco_mapping=det_cfg.get("coco_mapping", None),
            imgsz=det_cfg.get("imgsz", 480),
        )

        # 2. Tracker
        trk_cfg = config.get("tracking", {})
        self.tracker: BaseTracker = ByteTracker(
            track_thresh=trk_cfg.get("track_thresh", 0.35),
            match_thresh=trk_cfg.get("match_thresh", 0.80),
            track_buffer=trk_cfg.get("track_buffer", 60),
            min_box_area=trk_cfg.get("min_box_area", 100.0),
            frame_rate=trk_cfg.get("frame_rate", 30),
        )

        # 3. Owner-Bag Associator
        asc_cfg = config.get("association", {})
        weights = asc_cfg.get("weights", {})
        self.associator = OwnerAssociator(
            weight_distance=weights.get("distance", 0.45),
            weight_temporal=weights.get("temporal_proximity", 0.35),
            weight_motion=weights.get("motion_consistency", 0.20),
            max_association_distance=asc_cfg.get("max_association_distance", 250.0),
            min_attended_duration=asc_cfg.get("min_attended_duration", 1.0),
            min_association_score=asc_cfg.get("min_association_score", 0.40),
            candidate_history_len=asc_cfg.get("candidate_history_len", 45),
        )

        # 4. Temporal State Machine
        tmp_cfg = config.get("temporal", {})
        self.state_machine = TemporalStateMachine(
            abandonment_time_threshold=tmp_cfg.get("abandonment_time_threshold", 15.0),
            owner_distance_threshold=tmp_cfg.get("owner_distance_threshold", 180.0),
            owner_moving_away_speed=tmp_cfg.get("owner_moving_away_speed", 1.5),
            object_movement_threshold=tmp_cfg.get("object_movement_threshold", 25.0),
            stationary_window_seconds=tmp_cfg.get("stationary_window_seconds", 3.0),
            occlusion_grace_period=tmp_cfg.get("occlusion_grace_period", 3.0),
            owner_return_distance=tmp_cfg.get("owner_return_distance", 120.0),
            alert_cooldown_seconds=tmp_cfg.get("alert_cooldown_seconds", 30.0),
        )

        # 5. Visualizer
        vis_cfg = config.get("visualization", {})
        self.visualizer = PipelineVisualizer(
            show_boxes=vis_cfg.get("show_bounding_boxes", True),
            show_ids=vis_cfg.get("show_track_ids", True),
            show_association=vis_cfg.get("show_owner_association", True),
            show_states=vis_cfg.get("show_state_banner", True),
            show_timer=vis_cfg.get("show_timer", True),
            show_fps=vis_cfg.get("show_fps", True),
        )

        # 6. Hardware Monitor
        self.power_monitor = PowerMonitor(gpu_device_id=0)

        # Rolling latency buffer
        self._latencies: List[float] = []
        self.total_processed_frames = 0
        self.start_wall_time = time.time()

    def reset(self):
        """Reset internal pipeline tracker, associator, and state machine."""
        self.tracker.reset()
        self.associator.reset()
        self.state_machine.reset()
        self._latencies.clear()
        self.total_processed_frames = 0
        self.start_wall_time = time.time()

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        frame_index: int = 0,
        render_annotated: bool = True,
    ) -> PipelineFrameResult:
        """Process a single video frame through all pipeline stages."""
        t_start = time.perf_counter()

        # Step 1: Object Detection
        detections = self.detector.detect(frame)

        # Step 2: Multi-Object Tracking
        active_tracks = self.tracker.update(detections, timestamp)
        person_tracks = [t for t in active_tracks if t.class_id == 0]
        bag_tracks = [t for t in active_tracks if t.class_id == 1]

        # Step 3: Owner-Bag Association
        ownership_records = self.associator.update(person_tracks, bag_tracks, timestamp)

        # Step 4: Temporal State Machine Reasoning
        bag_states, new_alerts = self.state_machine.update(bag_tracks, ownership_records, timestamp)

        # Latency & FPS calculation
        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0
        self._latencies.append(latency_ms)
        if len(self._latencies) > 30:
            self._latencies.pop(0)

        avg_latency = sum(self._latencies) / len(self._latencies)
        current_fps = 1000.0 / max(avg_latency, 1.0)
        self.total_processed_frames += 1

        # Query hardware metrics
        hw_metrics = self.power_monitor.get_metrics()

        # Step 5: Visual Rendering
        annotated_frame = frame
        if render_annotated:
            annotated_frame = self.visualizer.render(
                frame=frame,
                person_tracks=person_tracks,
                bag_tracks=bag_tracks,
                ownership_records=ownership_records,
                bag_states=bag_states,
                active_alerts=new_alerts,
                fps=current_fps,
                latency_ms=latency_ms,
            )

        return PipelineFrameResult(
            frame_index=frame_index,
            timestamp=timestamp,
            annotated_frame=annotated_frame,
            person_tracks=person_tracks,
            bag_tracks=bag_tracks,
            ownership_records=ownership_records,
            bag_states=bag_states,
            alerts=new_alerts,
            fps=current_fps,
            latency_ms=latency_ms,
            hardware_metrics=hw_metrics,
        )

    def process_video(
        self,
        input_path: str | Path,
        output_path: Optional[str | Path] = None,
        max_frames: Optional[int] = None,
        frame_stride: int = 1,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> Dict[str, any]:
        """Run pipeline on an entire video file."""
        self.reset()
        video_reader = VideoReader(input_path)
        total_frames = video_reader.total_frames
        if max_frames:
            total_frames = min(total_frames, max_frames)

        video_writer = None
        if output_path:
            video_writer = VideoWriter(
                output_path=output_path,
                fps=video_reader.fps / frame_stride,
                width=video_reader.width,
                height=video_reader.height,
            )

        collected_alerts: List[AbandonmentAlert] = []
        frame_idx = 0
        processed_count = 0
        start_time = time.time()
        should_render = output_path is not None

        try:
            for f_idx, timestamp, frame in video_reader:
                if max_frames and f_idx >= max_frames:
                    break

                if f_idx % frame_stride != 0:
                    continue

                res = self.process_frame(frame, timestamp, frame_index=f_idx, render_annotated=should_render)

                if video_writer:
                    video_writer.write(res.annotated_frame)

                if res.alerts:
                    collected_alerts.extend(res.alerts)

                frame_idx = f_idx
                processed_count += 1
                if progress_callback and (processed_count % 30 == 0 or f_idx >= total_frames - 1):
                    progress_callback(processed_count, total_frames // frame_stride, res.fps)

        finally:
            video_reader.release()
            if video_writer:
                video_writer.release()

        elapsed = time.time() - start_time
        power_summary = self.power_monitor.get_power_summary(frame_idx, elapsed)

        return {
            "total_frames": frame_idx,
            "elapsed_seconds": round(elapsed, 2),
            "average_fps": round(frame_idx / max(elapsed, 1e-4), 2),
            "total_alerts": len(collected_alerts),
            "alerts": collected_alerts,
            "transition_history": self.state_machine.transition_history,
            "power_summary": power_summary,
        }
