"""Shared fixtures for reposition geometry tests."""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Simple rectangular room (4 m × 3 m, floor at y = 1.6 m)
# Camera at origin (0, 0, 0).  JSON / xyz2json coordinate frame.
# ---------------------------------------------------------------------------
@pytest.fixture
def simple_room_data() -> dict:
    """4 m × 3 m room centered on camera in XZ, floor at y = 1.6 m."""
    camera_height = 1.6
    hw, hd = 2.0, 1.5  # half-width, half-depth

    pts = [
        {"xyz": [-hw, camera_height,  hd]},   # 0: back-left
        {"xyz": [ hw, camera_height,  hd]},   # 1: back-right
        {"xyz": [ hw, camera_height, -hd]},   # 2: front-right
        {"xyz": [-hw, camera_height, -hd]},   # 3: front-left
    ]

    walls = []
    # We iterate corners to build walls with inward normals.
    for i in range(4):
        j = (i + 1) % 4
        a = np.array(pts[i]["xyz"], dtype=np.float64)
        b = np.array(pts[j]["xyz"], dtype=np.float64)
        # Compute horizontal inward normal (pointing toward origin in XZ).
        wall_dir = b - a
        # Right-hand perpendicular on XZ plane
        normal_2d = np.array([-wall_dir[2], wall_dir[0]], dtype=np.float64)  # (-dz, dx)
        normal_2d /= np.linalg.norm(normal_2d)
        # Make sure it points toward origin
        midpoint_xz = (a[[0, 2]] + b[[0, 2]]) / 2.0
        if np.dot(normal_2d, -midpoint_xz) < 0:  # -midpoint points toward origin
            normal_2d = -normal_2d
        normal = np.array([normal_2d[0], 0.0, normal_2d[1]], dtype=np.float64)
        d = -float(np.dot(normal, a))
        walls.append({
            "pointsIdx": [i, j],
            "planeEquation": [float(normal[0]), float(normal[1]), float(normal[2]), d],
        })

    return {
        "cameraHeight": camera_height,
        "cameraCeilingHeight": 1.0,
        "layoutHeight": 2.6,
        "layoutPoints": {"points": pts},
        "layoutWalls": {"walls": walls},
    }


# ---------------------------------------------------------------------------
# Small sample depth array and masks
# ---------------------------------------------------------------------------
@pytest.fixture
def dummy_depth() -> np.ndarray:
    """256-bin depth: constant wall distance = 2.0 m (plan_y units)."""
    # wall_distance_from_depth does abs(depth[idx]) * cameraHeight
    # So depth = wall_dist / cameraHeight
    return np.full(256, 2.0 / 1.6, dtype=np.float64)


@pytest.fixture
def small_binary_mask() -> np.ndarray:
    """A 20×40 mask with a rectangular blob near centre."""
    mask = np.zeros((512, 1024), dtype=bool)
    mask[200:300, 400:600] = True
    return mask


@pytest.fixture
def empty_mask() -> np.ndarray:
    return np.zeros((512, 1024), dtype=bool)
