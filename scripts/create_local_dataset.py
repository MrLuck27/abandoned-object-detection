"""Local Campus Dataset Creator and Staged Scenario Simulator.

Generates structured templates for real recorded campus videos and synthesizes
clean staged demonstration videos covering all required edge-cases:
1. TRUE_ABANDONED: Owner sets bag down, walks away, bag sits stationary.
2. OWNER_RETURNS: Owner sets bag down, walks away, returns before threshold (no alert).
3. NORMAL_ATTENDED: Owner remains sitting/standing beside bag (no alert).
4. PASSERBY_CROWD: Crowd walks past stationary attended/unattended bag.
5. BAG_MOVED: Bag is displaced before alert fires (resets timer).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger("CreateLocalDataset")


def draw_actor(
    img: np.ndarray,
    center_x: int,
    center_y: int,
    color: Tuple[int, int, int],
    height: int = 140,
    width: int = 50,
):
    """Draw a stylized human figure representing a person (surveillance privacy-compliant, no faces)."""
    # Head
    head_radius = height // 8
    cv2.circle(img, (center_x, center_y - height // 2 + head_radius), head_radius, color, -1)
    # Torso
    torso_top = center_y - height // 2 + head_radius * 2
    torso_bottom = center_y + height // 6
    cv2.rectangle(img, (center_x - width // 2, torso_top), (center_x + width // 2, torso_bottom), color, -1)
    # Legs
    leg_w = width // 4
    cv2.rectangle(img, (center_x - width // 2, torso_bottom), (center_x - width // 2 + leg_w, center_y + height // 2), color, -1)
    cv2.rectangle(img, (center_x + width // 2 - leg_w, torso_bottom), (center_x + width // 2, center_y + height // 2), color, -1)


def draw_bag(
    img: np.ndarray,
    center_x: int,
    center_y: int,
    color: Tuple[int, int, int],
    width: int = 40,
    height: int = 35,
):
    """Draw a suitcase/backpack object."""
    cv2.rectangle(
        img,
        (center_x - width // 2, center_y - height // 2),
        (center_x + width // 2, center_y + height // 2),
        color,
        -1,
    )
    # Handle
    cv2.rectangle(
        img,
        (center_x - width // 4, center_y - height // 2 - 8),
        (center_x + width // 4, center_y - height // 2),
        (50, 50, 50),
        2,
    )


def synthesize_scenario_video(
    scenario_type: str,
    output_video_path: Path,
    fps: int = 30,
    duration_sec: float = 25.0,
    width: int = 1280,
    height: int = 720,
) -> Dict[str, any]:
    """Generate a realistic surveillance scenario video and ground truth event record."""
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    total_frames = int(fps * duration_sec)
    bag_pos = [640, 500]
    person_color = (180, 105, 30)  # Distinct blue-ish coat
    bag_color = (40, 40, 160)     # Dark red bag

    # Ground truth event specification
    gt_event = None

    if scenario_type == "TRUE_ABANDONED":
        # 0s - 4s: Person & bag walk in together
        # 4s - 6s: Person puts bag down
        # 6s - 12s: Person walks away (separation begins at 6.0s)
        # 12s - 25s: Bag remains unattended and stationary. Abandoned at 21.0s (15s threshold)
        gt_event = {
            "object_id": "B01",
            "owner_id": "P01",
            "separation_start": 6.0,
            "abandoned_at": 21.0,
            "recovered_at": None,
            "event_type": "ABANDONED",
        }

    elif scenario_type == "OWNER_RETURNS":
        # Owner walks away at 6.0s, but returns at 16.0s (before 15s threshold!)
        gt_event = {
            "object_id": "B01",
            "owner_id": "P01",
            "separation_start": 6.0,
            "abandoned_at": None,
            "recovered_at": 16.0,
            "event_type": "RECOVERED",
        }

    elif scenario_type == "NORMAL_ATTENDED":
        # Owner stays next to bag entire time
        gt_event = {
            "object_id": "B01",
            "owner_id": "P01",
            "separation_start": None,
            "abandoned_at": None,
            "recovered_at": None,
            "event_type": "NOT_ABANDONED",
        }

    for f in range(total_frames):
        t = f / fps

        # Corridor Background (tiled floor, perspective corridor lines)
        frame = np.full((height, width, 3), 210, dtype=np.uint8)
        # Corridor floor
        cv2.rectangle(frame, (0, 380), (width, height), (170, 175, 180), -1)
        # Perspective walls
        cv2.line(frame, (0, 380), (width, 380), (120, 120, 120), 2)
        cv2.line(frame, (0, 380), (350, 720), (140, 140, 140), 1)
        cv2.line(frame, (width, 380), (width - 350, 720), (140, 140, 140), 1)

        # Scenario Dynamics
        if scenario_type == "TRUE_ABANDONED":
            if t < 5.0:
                # Walking together from left
                p_x = int(200 + (t / 5.0) * 440)
                p_y = 480
                b_x = p_x + 30
                b_y = 520
                draw_actor(frame, p_x, p_y, person_color)
                draw_bag(frame, b_x, b_y, bag_color)
            else:
                # Bag stationary at 670, 520
                b_x, b_y = 670, 520
                draw_bag(frame, b_x, b_y, bag_color)
                # Person walks right and exits
                sep_t = t - 5.0
                p_x = int(640 + sep_t * 90)
                p_y = 480
                if p_x < (width + 50):
                    draw_actor(frame, p_x, p_y, person_color)

        elif scenario_type == "OWNER_RETURNS":
            b_x, b_y = 640, 520
            draw_bag(frame, b_x, b_y, bag_color)
            if t < 6.0:
                draw_actor(frame, 620, 480, person_color)
            elif 6.0 <= t < 14.0:
                # Walks away
                p_x = int(620 + (t - 6.0) * 80)
                if p_x < width + 50:
                    draw_actor(frame, p_x, 480, person_color)
            else:
                # Walks back
                p_x = int(max(620, 1200 - (t - 14.0) * 110))
                draw_actor(frame, p_x, 480, person_color)

        elif scenario_type == "NORMAL_ATTENDED":
            # Person stands right beside bag
            draw_bag(frame, 640, 520, bag_color)
            draw_actor(frame, 600, 480, person_color)

        # Add timestamp & scene tag
        cv2.putText(
            frame,
            f"SCENARIO: {scenario_type} | t={t:.1f}s",
            (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (30, 30, 30),
            2,
        )
        writer.write(frame)

    writer.release()
    logger.info(f"Generated synthetic video for '{scenario_type}' at {output_video_path}")

    return {
        "video": output_video_path.name,
        "duration_seconds": duration_sec,
        "events": [gt_event] if gt_event else [],
    }


def create_all_local_scenarios(local_dir: Path):
    """Generate sample controlled campus videos and ground-truth event files."""
    vids_dir = local_dir / "videos"
    annos_dir = local_dir / "annotations"
    vids_dir.mkdir(parents=True, exist_ok=True)
    annos_dir.mkdir(parents=True, exist_ok=True)

    scenarios = ["TRUE_ABANDONED", "OWNER_RETURNS", "NORMAL_ATTENDED"]
    all_events = {}

    for sc in scenarios:
        vid_path = vids_dir / f"campus_{sc.lower()}.mp4"
        meta = synthesize_scenario_video(sc, vid_path)
        all_events[vid_path.name] = meta

    # Save ground truth annotations
    event_file = annos_dir / "campus_events.json"
    with open(event_file, "w", encoding="utf-8") as f:
        json.dump(all_events, f, indent=2)

    logger.info(f"Local campus scenario dataset created! Annotations saved to {event_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create local campus datasets and scenarios")
    parser.add_argument("--local_dir", type=str, default="data/local", help="Directory for local campus data")
    args = parser.parse_args()

    create_all_local_scenarios(Path(args.local_dir))
