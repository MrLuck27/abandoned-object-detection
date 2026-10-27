"""Unit tests for spatial distance and kinematic metrics."""

import numpy as np
import pytest
from src.association.spatial_metrics import (
    euclidean_distance,
    box_edge_distance,
    normalized_distance,
    motion_cosine_similarity,
)


def test_euclidean_distance():
    p1 = np.array([0.0, 0.0])
    p2 = np.array([30.0, 40.0])
    dist = euclidean_distance(p1, p2)
    assert np.isclose(dist, 50.0)


def test_box_edge_distance_non_overlapping():
    box1 = np.array([0, 0, 10, 10])
    box2 = np.array([20, 0, 30, 10])
    dist = box_edge_distance(box1, box2)
    assert np.isclose(dist, 10.0)


def test_box_edge_distance_overlapping():
    box1 = np.array([0, 0, 20, 20])
    box2 = np.array([10, 10, 30, 30])
    dist = box_edge_distance(box1, box2)
    assert np.isclose(dist, 0.0)


def test_normalized_distance():
    person_box = np.array([100, 100, 150, 250])  # height = 150, center = (125, 175)
    bag_box = np.array([120, 220, 140, 240])     # center = (130, 230)
    norm_dist = normalized_distance(person_box, bag_box)
    assert norm_dist > 0.0
    assert norm_dist < 1.0  # Close proximity normalized by person height


def test_motion_cosine_similarity():
    # Co-moving parallel vectors
    v1 = np.array([10.0, 0.0])
    v2 = np.array([20.0, 0.0])
    assert np.isclose(motion_cosine_similarity(v1, v2), 1.0)

    # Opposing vectors
    v3 = np.array([-10.0, 0.0])
    assert np.isclose(motion_cosine_similarity(v1, v3), -1.0)

    # Stationary vector
    v_stat = np.array([0.0, 0.0])
    assert np.isclose(motion_cosine_similarity(v1, v_stat), 0.5)
