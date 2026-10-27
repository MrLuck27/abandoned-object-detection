"""Evaluation module for object detection, tracking, and event-level verification."""

from .metrics import (
    EventEvaluationMetrics,
    compute_precision_recall_f1,
    compute_false_alarms_per_hour,
)
from .event_evaluator import EventEvaluator
from .detection_evaluator import DetectionEvaluator

__all__ = [
    "EventEvaluationMetrics",
    "compute_precision_recall_f1",
    "compute_false_alarms_per_hour",
    "EventEvaluator",
    "DetectionEvaluator",
]
