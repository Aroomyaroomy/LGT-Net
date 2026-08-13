from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from reposition.reposition import (
    default_standing_rotation,
    freestanding_floor_translation,
    get_masks,
    placements_from_mask_dir,
    resolve_mesh_scale,
    resolve_standing_pose,
    resolve_standing_rotation,
)
from reposition.lgt_utils import uv2equirectangular
from reposition.sam_utils import uprightness, uprightness_metrics


def _square_room(half: float = 2.0, camera_height: float = 1.6) -> dict:
    """Axis-aligned square footprint centered on the camera (JSON frame)."""
    pts = [
        [half, 0.0, -half],
        [half, 0.0, half],
        [-half, 0.0, half],
        [-half, 0.0, -half],
    ]
    walls = [
        {
            "normal": [1.0, 0.0, 0.0],
            "planeEquation": [1.0, 0.0, 0.0, -half],
            "pointsIdx": [0, 1],
        },
        {
            "normal": [0.0, 0.0, 1.0],
            "planeEquation": [0.0, 0.0, 1.0, -half],
            "pointsIdx": [1, 2],
        },
        {
            "normal": [-1.0, 0.0, 0.0],
            "planeEquation": [-1.0, 0.0, 0.0, -half],
            "pointsIdx": [2, 3],
        },
        {
            "normal": [0.0, 0.0, -1.0],
            "planeEquation": [0.0, 0.0, -1.0, -half],
            "pointsIdx": [3, 0],
        },
    ]
    return {
        "cameraHeight": camera_height,
        "layoutHeight": camera_height + 1.4,
        "layoutPoints": {
            "num": 4,
            "points": [{"xyz": p} for p in pts],
        },
        "layoutWalls": {"num": 4, "walls": walls},
    }


def _blob_mask(h: int = 64, w: int = 128, *, v0: float = 0.55, v1: float = 0.85) -> np.ndarray:
    """Compact boolean mask in the lower pano half (away from pure nadir)."""
    mask = np.zeros((h, w), dtype=bool)
    y0, y1 = int(v0 * h), int(v1 * h)
    x0, x1 = w // 2 - 4, w // 2 + 4
    mask[y0:y1, x0:x1] = True
    return mask


def _write_mask_png(path: Path, binary: np.ndarray) -> None:
    Image.fromarray((binary.astype(np.uint8) * 255)).save(path)


def _is_renderable(placement: dict) -> bool:
    """Renderer-safe: finite translation, resolved scale, and not QC-rejected."""
    if placement.get("error"):
        return False
    if (placement.get("quality") or {}).get("keep") is False:
        return False
    t = placement.get("translation")
    s = placement.get("scale")
    if t is None or not isinstance(s, dict) or s.get("factor") is None:
        return False
    arr = np.asarray(t, dtype=np.float64).reshape(-1)
    factor = float(s["factor"])
    return (
        arr.shape == (3,)
        and np.all(np.isfinite(arr))
        and np.isfinite(factor)
        and factor > 0.0
    )


class CrashSafetyTests(unittest.TestCase):
    def test_empty_mask_pose_is_all_nones(self):
        """Failed pose resolution must be a full None tuple, never a partial pose."""
        data = _square_room()
        empty = np.zeros((64, 128), dtype=bool)
        out = resolve_standing_pose(empty, data, depth=np.ones(256))
        self.assertEqual(out[:5], (None, None, None, None, None))
        self.assertIn("translation: no_contact_uvs", out[5])

    def test_failed_poses_never_reach_renderer_payload(self):
        """
        Skipped objects may be logged with an error, but must not look renderable.
        failure counts every skipped object (I/O, empty, unresolved translation/scale).
        """
        data = _square_room()
        with tempfile.TemporaryDirectory() as tmp:
            mask_dir = Path(tmp)
            _write_mask_png(mask_dir / "mask_0.png", _blob_mask())
            _write_mask_png(mask_dir / "mask_1.png", np.zeros((32, 64), dtype=bool))

            with patch(
                "reposition.reposition.resolve_standing_pose",
                return_value=(None, None, None, None, None, "unresolved: translation: no_contact_uvs"),
            ):
                placements, success, failure = placements_from_mask_dir(
                    mask_dir,
                    data,
                    depth=np.ones(256),
                    do_manhattan=False,
                )

        self.assertEqual(success, 0)
        self.assertEqual(failure, 2)
        self.assertFalse(any(_is_renderable(p) for p in placements))
        for p in placements:
            self.assertIsNone(p.get("translation"))
            self.assertIn("error", p)

    def test_get_masks_rejects_missing_directory(self):
        with self.assertRaises(NotADirectoryError):
            get_masks(Path("definitely_not_a_mask_dir_xyz"))


class CoreLogicTests(unittest.TestCase):
    def test_freestanding_floor_translation_depth_gate_and_placement(self):
        data = _square_room()
        binary = _blob_mask()
        uv = np.array([0.5, 0.7], dtype=np.float64)
        direction = uv2equirectangular(uv)
        origin = np.zeros(3, dtype=np.float64)

        self.assertEqual(
            freestanding_floor_translation(
                origin, direction, uv, binary, data, depth=None
            ),
            (None, None, "no_depth"),
        )

        depth = np.full(256, 2.0 / data["cameraHeight"], dtype=np.float64)
        p, method, err = freestanding_floor_translation(
            origin,
            direction,
            uv,
            binary,
            data,
            depth,
            object_height=0.9,
        )
        self.assertIsNotNone(p)
        self.assertIsNone(err)
        self.assertIn(method, {"floor", "angular", "wall_fraction"})
        self.assertAlmostEqual(float(p[1]), data["cameraHeight"], places=5)
        self.assertLess(float(np.linalg.norm(p[[0, 2]])), 2.0)

    def test_mesh_scale_resolved_or_none(self):
        binary = _blob_mask()
        translation = np.array([0.0, 1.6, 1.0], dtype=np.float64)

        room = resolve_mesh_scale(
            binary, translation, "floor", "wall_fraction", object_height=1.0
        )
        self.assertEqual(room["method"], "room_distance")
        self.assertGreater(room["factor"], 0.0)

        angular = resolve_mesh_scale(
            binary, translation, "floor", "angular", object_height=1.0
        )
        self.assertEqual(angular["method"], "angular")
        self.assertEqual(angular["factor"], 1.0)

        # Wall: native SAM3D size is intentional.
        wall = resolve_mesh_scale(
            binary, translation, "wall", None, object_height=1.0
        )
        self.assertEqual(wall["method"], "sam3d")
        self.assertEqual(wall["factor"], 1.0)

        # Floor without a derivable scale must not invent factor=1.0.
        self.assertIsNone(
            resolve_mesh_scale(binary, translation, "floor", "wall_fraction")
        )

    def test_rotation_fallback_does_not_skip(self):
        """Rotation failure alone → default upright orientation, still placed."""
        data = _square_room()
        binary = _blob_mask()
        depth = np.full(256, 2.0 / data["cameraHeight"], dtype=np.float64)
        translation = np.array([0.5, 1.6, 0.8], dtype=np.float64)

        fallback = default_standing_rotation(translation)
        self.assertEqual(fallback["source"], "default")
        self.assertEqual(np.asarray(fallback["matrix"]).shape, (3, 3))

        with patch(
            "reposition.reposition.resolve_standing_rotation",
            return_value=(None, None, None),
        ):
            t, uv, surface, rotation, scale, err = resolve_standing_pose(
                binary,
                data,
                depth=depth,
                object_height=0.8,
                num_samples=3,
            )

        self.assertIsNotNone(t)
        self.assertIsNone(err)
        self.assertEqual(surface, "floor")
        self.assertIsNotNone(scale)
        self.assertEqual(rotation["source"], "default")
        self.assertIn("yaw", rotation)

    def test_standing_pose_and_rotation_on_floor(self):
        data = _square_room()
        binary = _blob_mask()
        depth = np.full(256, 2.0 / data["cameraHeight"], dtype=np.float64)

        translation, contact_uv, surface, rotation, scale, err = resolve_standing_pose(
            binary,
            data,
            depth=depth,
            object_height=0.8,
            num_samples=3,
        )
        self.assertIsNotNone(translation)
        self.assertIsNone(err)
        self.assertIsNotNone(contact_uv)
        self.assertEqual(surface, "floor")
        self.assertIsInstance(rotation, dict)
        self.assertIn("yaw", rotation)
        self.assertEqual(np.asarray(rotation["matrix"]).shape, (3, 3))
        self.assertIsInstance(scale, dict)
        self.assertIn(scale["method"], {"room_distance", "angular"})

        yaw, matrix, normal = resolve_standing_rotation(translation, data)
        self.assertIsNotNone(yaw)
        self.assertEqual(matrix.shape, (3, 3))
        self.assertAlmostEqual(float(np.linalg.norm(normal[[0, 2]])), 1.0, places=5)

    def test_unresolved_scale_skips_and_counts_failure(self):
        """Scale unresolved → skip object; counted as failure, not renderable."""
        data = _square_room()
        with tempfile.TemporaryDirectory() as tmp:
            mask_dir = Path(tmp)
            _write_mask_png(mask_dir / "mask_0.png", _blob_mask())

            fake_t = np.array([0.4, 1.6, 0.5], dtype=np.float64)
            fake_uv = np.array([0.5, 0.7], dtype=np.float64)
            with patch(
                "reposition.reposition.resolve_standing_pose",
                return_value=(
                    fake_t,
                    fake_uv,
                    "floor",
                    default_standing_rotation(fake_t),
                    None,  # scale unresolved
                    "unresolved: scale: missing_object_height",
                ),
            ):
                placements, success, failure = placements_from_mask_dir(
                    mask_dir,
                    data,
                    depth=np.ones(256),
                    do_manhattan=False,
                )

        self.assertEqual(success, 0)
        self.assertEqual(failure, 1)
        self.assertFalse(any(_is_renderable(p) for p in placements))
        self.assertIn("scale", placements[0]["error"])


    def test_failure_causes_are_specific(self):
        data = _square_room()
        binary = _blob_mask()
        depth = np.full(256, 2.0 / data["cameraHeight"], dtype=np.float64)

        t, *_, err = resolve_standing_pose(binary, data, depth=depth)
        self.assertIsNone(t)
        self.assertIn("scale: missing_object_height", err)

        t, *_, err = resolve_standing_pose(binary, {}, depth=depth)
        self.assertIsNone(t)
        self.assertIn("translation: missing_camera_height", err)

        t, *_, err = resolve_standing_pose(
            binary, {"cameraHeight": 1.6}, depth=depth
        )
        self.assertIsNone(t)
        self.assertIn("translation: missing_layout", err)


class UprightnessEvalTests(unittest.TestCase):
    """Evaluation-only metric — does not affect placement."""

    def test_identity_is_perfectly_upright(self):
        self.assertAlmostEqual(uprightness(None), 1.0, places=6)
        self.assertAlmostEqual(uprightness(np.eye(3)), 1.0, places=6)
        m = uprightness_metrics(np.eye(3))
        self.assertAlmostEqual(m["tilt_deg"], 0.0, places=5)
        self.assertAlmostEqual(m["upright_01"], 1.0, places=6)

    def test_sideways_and_upside_down(self):
        # 90° about +X → local +Y maps to -Z (horizontal)
        r_side = np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        self.assertAlmostEqual(uprightness(r_side), 0.0, places=6)

        # 180° about +X → upside down
        r_flip = np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            dtype=np.float64,
        )
        self.assertAlmostEqual(uprightness(r_flip), -1.0, places=6)
        self.assertAlmostEqual(uprightness_metrics(r_flip)["tilt_deg"], 180.0, places=5)

    def test_yaw_only_stays_upright(self):
        # Pure yaw about world +Y must keep uprightness = 1
        yaw = 0.7
        c, s = np.cos(yaw), np.sin(yaw)
        r_yaw = np.array(
            [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
            dtype=np.float64,
        )
        self.assertAlmostEqual(uprightness({"matrix": r_yaw.tolist()}), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
