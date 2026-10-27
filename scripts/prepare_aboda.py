"""ABODA (Abandoned Object Detection in Airport) Dataset Preparation Script.

This script parses ABODA video sequences and ground truth annotations,
extracts frames, converts bounding box annotations into standardized YOLO format,
and performs SEQUENCE-LEVEL splitting (train/val/test) to prevent temporal data leakage.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import yaml

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger("PrepareABODA")


def extract_frames_from_video(
    video_path: Path, output_dir: Path, sample_rate: int = 5
) -> List[Tuple[int, Path, int, int]]:
    """Extract frames from a video at a specified frame stride.

    Returns:
        List of tuples: (frame_idx, frame_path, width, height)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Failed to open video: {video_path}")
        return []

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_idx = 0
    extracted_frames = []

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if frame_idx % sample_rate == 0:
            frame_filename = f"{video_path.stem}_frame_{frame_idx:06d}.jpg"
            frame_path = output_dir / frame_filename
            cv2.imwrite(str(frame_path), frame)
            extracted_frames.append((frame_idx, frame_path, width, height))

        frame_idx += 1

    cap.release()
    return extracted_frames


def parse_aboda_xml_annotation(xml_path: Path) -> Dict[int, List[Tuple[int, float, float, float, float]]]:
    """Parse ABODA XML annotation file if available.

    Returns mapping of frame_number -> list of (class_id, x_center, y_center, width, height).
    """
    annotations = {}
    if not xml_path.is_file():
        return annotations

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for track in root.findall(".//track"):
            label = track.attrib.get("label", "").lower()
            # Map labels: 0: person, 1: bag/luggage
            cls_id = 0 if "person" in label else 1

            for box in track.findall("box"):
                frame_num = int(box.attrib.get("frame", 0))
                xtl = float(box.attrib.get("xtl", 0))
                ytl = float(box.attrib.get("ytl", 0))
                xbr = float(box.attrib.get("xbr", 0))
                ybr = float(box.attrib.get("ybr", 0))

                if frame_num not in annotations:
                    annotations[frame_num] = []
                annotations[frame_num].append((cls_id, xtl, ytl, xbr, ybr))

    except Exception as e:
        logger.error(f"Error parsing XML {xml_path}: {e}")

    return annotations


def convert_bbox_to_yolo(
    box: Tuple[int, float, float, float, float], img_w: int, img_h: int
) -> str:
    """Convert (cls_id, x1, y1, x2, y2) into normalized YOLO string."""
    cls_id, x1, y1, x2, y2 = box
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0

    cx_n = cx / img_w
    cy_n = cy / img_h
    w_n = w / img_w
    h_n = h / img_h

    return f"{cls_id} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}\n"


def prepare_aboda_dataset(
    raw_dir: Path,
    output_dir: Path,
    sample_rate: int = 5,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
):
    """Main pipeline for ABODA preparation with SEQUENCE-LEVEL splits."""
    logger.info(f"Checking ABODA raw data directory: {raw_dir}")

    # Look for video files in raw_dir
    video_extensions = [".avi", ".mp4", ".mkv", ".mov"]
    video_files = [f for f in raw_dir.glob("**/*") if f.suffix.lower() in video_extensions]

    if not video_files:
        logger.warning(
            f"No raw video files found in '{raw_dir}'.\n"
            f"Please download the ABODA dataset and place the video files into '{raw_dir}'.\n"
            f"Expected structure:\n"
            f"  data/raw/aboda/\n"
            f"    ├── video1.avi\n"
            f"    ├── video1.xml\n"
            f"    ├── video2.avi\n"
            f"    └── ...\n"
        )
        return

    logger.info(f"Found {len(video_files)} video sequences. Partitioning sequences...")

    # Sequence-level splitting to prevent temporal leakage
    # Sort for deterministic splitting
    video_files.sort(key=lambda x: x.name)
    n_videos = len(video_files)
    n_train = max(1, int(n_videos * train_ratio))
    n_val = max(1, int(n_videos * val_ratio))

    train_videos = video_files[:n_train]
    val_videos = video_files[n_train : n_train + n_val]
    test_videos = video_files[n_train + n_val :]
    if not test_videos:
        test_videos = val_videos  # Fallback if few sequences

    splits = {"train": train_videos, "val": val_videos, "test": test_videos}

    for split_name, v_list in splits.items():
        img_out = output_dir / "images" / split_name
        lbl_out = output_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing split '{split_name}' with {len(v_list)} sequences...")
        for v in v_list:
            logger.info(f"Extracting frames for {v.name}...")
            frames = extract_frames_from_video(v, img_out, sample_rate=sample_rate)

            # Look for XML annotation
            xml_candidate = v.with_suffix(".xml")
            annotations = parse_aboda_xml_annotation(xml_candidate)

            for f_idx, f_path, w, h in frames:
                lbl_file = lbl_out / f"{f_path.stem}.txt"
                if f_idx in annotations:
                    lines = [convert_bbox_to_yolo(b, w, h) for b in annotations[f_idx]]
                    lbl_file.write_text("".join(lines), encoding="utf-8")
                else:
                    # Write empty label file if no objects present
                    lbl_file.write_text("", encoding="utf-8")

    # Write data.yaml for Ultralytics YOLO
    data_yaml = output_dir / "dataset.yaml"
    yaml_content = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "person", 1: "bag"},
    }
    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    logger.info(f"ABODA dataset preparation complete! Dataset YAML written to {data_yaml}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare ABODA dataset into YOLO format")
    parser.add_argument("--raw_dir", type=str, default="data/raw/aboda", help="Raw ABODA directory")
    parser.add_argument("--output_dir", type=str, default="data/processed/yolo_dataset", help="Output directory")
    parser.add_argument("--sample_rate", type=int, default=5, help="Frame stride for extraction")
    args = parser.parse_args()

    prepare_aboda_dataset(
        raw_dir=Path(args.raw_dir),
        output_dir=Path(args.output_dir),
        sample_rate=args.sample_rate,
    )
