"""Event-level Ground Truth evaluation engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from .metrics import (
    EventEvaluationMetrics,
    compute_precision_recall_f1,
    compute_false_alarms_per_hour,
)
from ..temporal.states import AbandonmentAlert


class EventEvaluator:
    """Evaluates predicted abandonment alerts against ground truth event annotations."""

    def __init__(self, temporal_tolerance_seconds: float = 5.0):
        """Initialize EventEvaluator.

        Args:
            temporal_tolerance_seconds: Max acceptable difference between ground truth
                                        abandonment time and detected alert time.
        """
        self.tolerance = temporal_tolerance_seconds

    def evaluate_video(
        self,
        predicted_alerts: List[AbandonmentAlert],
        ground_truth_events: List[Dict[str, any]],
        video_duration_seconds: float,
    ) -> Dict[str, any]:
        """Evaluate a single video's predictions against ground truth events."""
        gt_abandoned = [e for e in ground_truth_events if e.get("event_type", "ABANDONED") == "ABANDONED"]
        matched_gt_indices = set()
        matched_pred_indices = set()
        time_to_alerts: List[float] = []

        # Sort predicted alerts chronologically
        sorted_preds = sorted(predicted_alerts, key=lambda a: a.timestamp)

        # Match each prediction with nearest valid GT event
        for p_idx, alert in enumerate(sorted_preds):
            for g_idx, gt in enumerate(gt_abandoned):
                if g_idx in matched_gt_indices:
                    continue

                gt_time = gt.get("abandoned_at", gt.get("separation_start", 0.0))
                # Check if alert occurred around or after abandonment time
                time_diff = alert.timestamp - gt_time

                if -self.tolerance <= time_diff <= (self.tolerance + 15.0):
                    matched_gt_indices.add(g_idx)
                    matched_pred_indices.add(p_idx)
                    time_to_alerts.append(max(0.0, alert.timestamp - gt.get("separation_start", gt_time)))
                    break

        tp = len(matched_gt_indices)
        fp = len(sorted_preds) - len(matched_pred_indices)
        fn = len(gt_abandoned) - tp

        prf = compute_precision_recall_f1(tp, fp, fn)
        fa_per_hour = compute_false_alarms_per_hour(fp, video_duration_seconds)
        mean_tta = (sum(time_to_alerts) / len(time_to_alerts)) if time_to_alerts else 0.0

        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": prf["precision"],
            "recall": prf["recall"],
            "f1": prf["f1"],
            "false_alarms_per_hour": fa_per_hour,
            "mean_time_to_alert": mean_tta,
            "duration_seconds": video_duration_seconds,
        }

    def evaluate_dataset(
        self,
        results_by_video: Dict[str, Dict[str, any]],
    ) -> EventEvaluationMetrics:
        """Aggregate results across multiple evaluated videos.

        Args:
            results_by_video: Dict mapping video_id to dict with:
                - 'predicted_alerts': List[AbandonmentAlert]
                - 'ground_truth_events': List[Dict]
                - 'duration_seconds': float
        """
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_duration = 0.0
        all_ttas = []

        for vid, vdata in results_by_video.items():
            eval_res = self.evaluate_video(
                predicted_alerts=vdata["predicted_alerts"],
                ground_truth_events=vdata["ground_truth_events"],
                video_duration_seconds=vdata["duration_seconds"],
            )
            total_tp += eval_res["tp"]
            total_fp += eval_res["fp"]
            total_fn += eval_res["fn"]
            total_duration += eval_res["duration_seconds"]
            if eval_res["mean_time_to_alert"] > 0:
                all_ttas.append(eval_res["mean_time_to_alert"])

        prf = compute_precision_recall_f1(total_tp, total_fp, total_fn)
        fa_hr = compute_false_alarms_per_hour(total_fp, total_duration)
        mean_tta = (sum(all_ttas) / len(all_ttas)) if all_ttas else 0.0
        total_hours = total_duration / 3600.0

        return EventEvaluationMetrics(
            true_positives=total_tp,
            false_positives=total_fp,
            false_negatives=total_fn,
            precision=prf["precision"],
            recall=prf["recall"],
            f1_score=prf["f1"],
            false_alarms_per_hour=fa_hr,
            mean_time_to_alert_seconds=mean_tta,
            total_monitoring_hours=total_hours,
        )
