"""CLI Inference Runner for Unattended Object Detection Pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_yaml
from src.inference.pipeline import SurveillancePipeline

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger("RunInference")


def main():
    parser = argparse.ArgumentParser(description="Run unattended object detection on a video stream")
    parser.add_argument("--source", type=str, required=True, help="Path to input video or webcam ID (e.g. 0)")
    parser.add_argument("--config", type=str, default="configs/system.yaml", help="Path to system config")
    parser.add_argument("--output", type=str, default="results/annotated_output.mp4", help="Path to save output video")
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--abandon_time", type=float, default=None, help="Override abandonment time threshold (seconds)")
    parser.add_argument("--owner_dist", type=float, default=None, help="Override owner separation distance (pixels)")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.abandon_time is not None:
        cfg["temporal"]["abandonment_time_threshold"] = args.abandon_time
    if args.owner_dist is not None:
        cfg["temporal"]["owner_distance_threshold"] = args.owner_dist

    # Parse source
    src = int(args.source) if args.source.isdigit() else args.source

    logger.info(f"Initializing Surveillance Pipeline with config: {args.config}")
    pipeline = SurveillancePipeline(cfg)

    def progress_callback(current, total, fps):
        logger.info(f"Processing: frame {current}/{total} | Current Speed: {fps:.1f} FPS")

    logger.info(f"Processing input source: {src} ...")
    summary = pipeline.process_video(
        input_path=src,
        output_path=args.output,
        max_frames=args.max_frames,
        progress_callback=progress_callback,
    )

    logger.info(
        f"\n" + "=" * 60 + "\n"
        f"INFERENCE RUN COMPLETE:\n"
        f"  Total Frames Processed: {summary['total_frames']}\n"
        f"  Elapsed Time:           {summary['elapsed_seconds']} s\n"
        f"  Average FPS:            {summary['average_fps']} FPS\n"
        f"  Total Alerts Triggered: {summary['total_alerts']}\n"
        f"  Annotated Video Saved:  {args.output}\n"
        + "=" * 60
    )


if __name__ == "__main__":
    main()
