"""Detector Evaluation Script.

Evaluates trained or baseline YOLO detector on held-out test set
and produces results/detection_metrics.json and confusion matrix.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.detection_evaluator import DetectionEvaluator

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger("EvaluateDetector")


def run_detector_evaluation(
    model_path: str = "models/weights/best.pt",
    data_yaml: str = "data/processed/yolo_dataset/dataset.yaml",
    split: str = "test",
    output_json: str = "results/metrics/detection_metrics.json",
    device: str = "0",
):
    """Run evaluation and save metrics."""
    m_path = Path(model_path)
    if not m_path.exists():
        logger.warning(f"Specified model '{model_path}' not found. Falling back to base pretrained 'yolov8n.pt'.")
        m_path = Path("yolov8n.pt")

    d_path = Path(data_yaml)
    if not d_path.exists():
        logger.error(
            f"Dataset YAML not found at: {d_path.resolve()}.\n"
            f"Please prepare the dataset before evaluating detector."
        )
    import torch
    if device != "cpu" and not torch.cuda.is_available():
        logger.warning(f"CUDA device '{device}' requested but torch.cuda.is_available() is False. Using 'cpu'.")
        device = "cpu"

    evaluator = DetectionEvaluator(model_path=str(m_path))
    metrics = evaluator.evaluate(
        data_yaml=str(d_path),
        split=split,
        device=device,
    )

    out_file = Path(output_json)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(
        f"Detector evaluation complete!\n"
        f"  mAP@0.5: {metrics.get('map50', 0.0):.4f}\n"
        f"  mAP@0.5:0.95: {metrics.get('map50_95', 0.0):.4f}\n"
        f"  Precision: {metrics.get('precision', 0.0):.4f}\n"
        f"  Recall: {metrics.get('recall', 0.0):.4f}\n"
        f"  Saved to: {out_file}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLO detector")
    parser.add_argument("--model", type=str, default="models/weights/best.pt", help="Path to model weights")
    parser.add_argument("--data", type=str, default="data/processed/yolo_dataset/dataset.yaml", help="Dataset yaml")
    parser.add_argument("--split", type=str, default="test", help="Evaluation split ('val' or 'test')")
    parser.add_argument("--output", type=str, default="results/metrics/detection_metrics.json", help="Output JSON path")
    parser.add_argument("--device", type=str, default="0", help="CUDA device or 'cpu'")
    args = parser.parse_args()

    run_detector_evaluation(
        model_path=args.model,
        data_yaml=args.data,
        split=args.split,
        output_json=args.output,
        device=args.device,
    )
