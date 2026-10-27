"""Dataset Statistics and Visual Analysis Generator.

Inspects images, label text files, class distributions, and bounding box ratios
across dataset splits and saves statistical summaries and plots.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger("AnalyzeDataset")


def analyze_dataset_split(split_img_dir: Path, split_lbl_dir: Path) -> Dict[str, any]:
    """Analyze single split directory."""
    img_files = list(split_img_dir.glob("*.jpg")) + list(split_img_dir.glob("*.png"))
    num_images = len(img_files)

    class_counts = {0: 0, 1: 0}  # 0: person, 1: bag
    widths = []
    heights = []
    box_areas = []

    for img_path in img_files:
        try:
            with Image.open(img_path) as im:
                w, h = im.size
                widths.append(w)
                heights.append(h)
        except Exception:
            pass

        lbl_path = split_lbl_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            lines = lbl_path.read_text(encoding="utf-8").strip().splitlines()
            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    box_w = float(parts[3])
                    box_h = float(parts[4])
                    class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
                    box_areas.append(box_w * box_h)

    return {
        "num_images": num_images,
        "person_instances": class_counts.get(0, 0),
        "bag_instances": class_counts.get(1, 0),
        "avg_width": float(np.mean(widths)) if widths else 0.0,
        "avg_height": float(np.mean(heights)) if heights else 0.0,
        "avg_box_norm_area": float(np.mean(box_areas)) if box_areas else 0.0,
    }


def generate_analysis_report(dataset_dir: Path, output_fig: Path, output_json: Path):
    """Scan train/val/test splits and create plots and JSON metrics."""
    splits = ["train", "val", "test"]
    stats_by_split = {}
    total_persons = 0
    total_bags = 0
    total_images = 0

    for s in splits:
        img_dir = dataset_dir / "images" / s
        lbl_dir = dataset_dir / "labels" / s
        if img_dir.exists():
            s_stats = analyze_dataset_split(img_dir, lbl_dir)
            stats_by_split[s] = s_stats
            total_images += s_stats["num_images"]
            total_persons += s_stats["person_instances"]
            total_bags += s_stats["bag_instances"]
        else:
            stats_by_split[s] = {
                "num_images": 0,
                "person_instances": 0,
                "bag_instances": 0,
                "avg_width": 0.0,
                "avg_height": 0.0,
            }

    summary = {
        "total_images": total_images,
        "total_person_instances": total_persons,
        "total_bag_instances": total_bags,
        "splits": stats_by_split,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Dataset summary JSON saved to {output_json}")

    # Plot class distributions and splits
    output_fig.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Plot 1: Instances per Split
    split_names = list(stats_by_split.keys())
    persons = [stats_by_split[s]["person_instances"] for s in split_names]
    bags = [stats_by_split[s]["bag_instances"] for s in split_names]

    x = np.arange(len(split_names))
    bar_w = 0.35

    axes[0].bar(x - bar_w / 2, persons, bar_w, label="Person (0)", color="#1f77b4")
    axes[0].bar(x + bar_w / 2, bags, bar_w, label="Bag (1)", color="#ff7f0e")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([s.upper() for s in split_names])
    axes[0].set_ylabel("Instance Count")
    axes[0].set_title("Class Instances by Split")
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # Plot 2: Total Class Distribution
    total_counts = [total_persons, total_bags]
    labels = ["Person", "Bag"]
    if sum(total_counts) > 0:
        axes[1].pie(total_counts, labels=labels, autopct="%1.1f%%", colors=["#1f77b4", "#ff7f0e"], startangle=90)
        axes[1].set_title("Overall Class Distribution")
    else:
        axes[1].text(0.5, 0.5, "No annotations found yet", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(str(output_fig), dpi=200)
    plt.close()

    logger.info(f"Dataset analysis figure saved to {output_fig}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze YOLO dataset")
    parser.add_argument("--dataset_dir", type=str, default="data/processed/yolo_dataset")
    parser.add_argument("--output_fig", type=str, default="results/figures/dataset_analysis.png")
    parser.add_argument("--output_json", type=str, default="results/metrics/dataset_statistics.json")
    args = parser.parse_args()

    generate_analysis_report(
        Path(args.dataset_dir), Path(args.output_fig), Path(args.output_json)
    )
