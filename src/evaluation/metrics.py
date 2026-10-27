"""Metric calculation utilities for detection, tracking, and event-level evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


@dataclass
class EventEvaluationMetrics:
    """Event-level evaluation metrics."""

    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    false_alarms_per_hour: float
    mean_time_to_alert_seconds: float
    total_monitoring_hours: float

    def to_dict(self) -> Dict[str, any]:
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "false_alarms_per_hour": round(self.false_alarms_per_hour, 2),
            "mean_time_to_alert_seconds": round(self.mean_time_to_alert_seconds, 2),
            "total_monitoring_hours": round(self.total_monitoring_hours, 4),
        }


def compute_precision_recall_f1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Calculate precision, recall, and F1 score with zero-division safety."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_false_alarms_per_hour(false_positives: int, total_duration_seconds: float) -> float:
    """Calculate normalized rate of false alarms per hour."""
    hours = max(total_duration_seconds / 3600.0, 1e-6)
    return float(false_positives / hours)
