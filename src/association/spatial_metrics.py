"""Spatial and kinematic relationship metrics between persons and objects."""

from __future__ import annotations

import numpy as np


def euclidean_distance(pt1: np.ndarray, pt2: np.ndarray) -> float:
    """Calculate Euclidean distance between two 2D points."""
    return float(np.linalg.norm(pt1 - pt2))


def normalized_distance(person_bbox: np.ndarray, bag_bbox: np.ndarray) -> float:
    """Calculate distance between centers normalized by person height.

    Normalizing by person height creates a camera-depth-invariant proxy for real-world distance.
    """
    p_h = max(person_bbox[3] - person_bbox[1], 10.0)
    p_center = np.array([(person_bbox[0] + person_bbox[2]) / 2.0, (person_bbox[1] + person_bbox[3]) / 2.0])
    b_center = np.array([(bag_bbox[0] + bag_bbox[2]) / 2.0, (bag_bbox[1] + bag_bbox[3]) / 2.0])

    pixel_dist = np.linalg.norm(p_center - b_center)
    return float(pixel_dist / p_h)


def box_edge_distance(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calculate minimum distance between boundaries of two axis-aligned bounding boxes."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    dx = max(0.0, max(x1_min - x2_max, x2_min - x1_max))
    dy = max(0.0, max(y1_min - y2_max, y2_min - y1_max))

    return float(np.sqrt(dx * dx + dy * dy))


def motion_cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two velocity vectors.

    Returns value between -1.0 (opposite directions) and 1.0 (identical direction).
    If either vector is stationary, returns 0.5 (neutral).
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 < 2.0 or norm2 < 2.0:
        return 0.5  # Neutral when one or both are stationary

    cos_sim = np.dot(v1, v2) / (norm1 * norm2)
    return float(np.clip(cos_sim, -1.0, 1.0))
