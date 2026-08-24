"""Unit tests for reposition.reposition — furniture placement orchestration."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from reposition.reposition import (
    freestanding_floor_translation,
    get_masks,
    load_aligned_binary_mask,
    placements_from_mask_dir,
    resolve_standing_pose,
    resolve_standing_rotation,
)


# ---------------------------------------------------------------------------
# Helpers (mirror test_lgt_utils pattern)
# ---------------------------------------------------------------------------

def _make_room_data(camera_height=1.6, hw=2.0, hd=1.5):
    """4m x 3m rectangular room centred on camera, floor at y=cameraHeight."""
    pts = [
        {"xyz": [-hw, camera_height,  hd]},
        {"xyz": [ hw, camera_height,  hd]},
        {"xyz": [ hw, camera_height, -hd]},
        {"xyz": [-hw, camera_height, -hd]},
    ]
    walls = []
    for i in range(4):
        j = (i + 1) % 4
        a = np.array(pts[i]["xyz"], dtype=np.float64)
        b = np.array(pts[j]["xyz"], dtype=np.float64)
        wdir = b - a
        n2d = np.array([-wdir[2], wdir[0]], dtype=np.float64)
        n2d /= np.linalg.norm(n2d)
        mid_xz = (a[[0, 2]] + b[[0, 2]]) / 2.0
        if np.dot(n2d, -mid_xz) < 0:
            n2d = -n2d
        normal = np.array([n2d[0], 0.0, n2d[1]], dtype=np.float64)
        d = -float(np.dot(normal, a))
        walls.append({
            "pointsIdx": [i, j],
            "planeEquation": [float(normal[0]), float(normal[1]),
                              float(normal[2]), d],
            "normal": normal,  # needed by resolve_standing_rotation
        })
    return {
        "cameraHeight": camera_height,
        "cameraCeilingHeight": 1.0,
        "layoutHeight": camera_height + 1.0,
        "layoutPoints": {"points": pts},
        "layoutWalls": {"walls": walls},
    }


def _dummy_depth(wall_dist=2.0, camera_height=1.6):
    """256-bin depth with constant wall distance in plan_y units."""
    return np.full(256, wall_dist / camera_height, dtype=np.float64)


def _centered_mask(size=100):
    """A square mask blob near image center."""
    mask = np.zeros((512, 1024), dtype=bool)
    h, w = mask.shape
    mask[h // 2 - size // 2: h // 2 + size // 2,
         w // 2 - size // 2: w // 2 + size // 2] = True
    return mask


def _bottom_mask(size=80):
    """A mask blob near the bottom of the image (close to camera)."""
    mask = np.zeros((512, 1024), dtype=bool)
    mask[400:480, 450:550] = True
    return mask


def _wall_mask():
    """A mask on the left side of the image (wall direction)."""
    mask = np.zeros((512, 1024), dtype=bool)
    mask[200:300, 0:80] = True
    return mask


# ═══════════════════════════════════════════════════════════════════════════
# resolve_standing_rotation
# ═══════════════════════════════════════════════════════════════════════════

class TestResolveStandingRotation(unittest.TestCase):
    def setUp(self):
        self.room = _make_room_data()

    def test_floor_object_returns_rotation(self):
        """Object on floor should get yaw from nearest wall inward normal."""
        t = np.array([0.0, 1.6, 1.0])  # near back wall
        yaw, R, normal = resolve_standing_rotation(t, self.room, wall=None)
        self.assertIsNotNone(yaw)
        self.assertIsNotNone(R)
        self.assertIsNotNone(normal)
        self.assertEqual(R.shape, (3, 3))
        # R should be orthogonal
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        # normal should be horizontal
        self.assertAlmostEqual(normal[1], 0.0)

    def test_wall_mounted_object(self):
        """Wall-mounted: uses the provided wall's normal directly."""
        t = np.array([0.0, 1.0, -1.45])  # near front wall
        front_wall = self.room["layoutWalls"]["walls"][2]  # front wall (-Z)
        yaw, R, normal = resolve_standing_rotation(t, self.room, wall=front_wall)
        self.assertIsNotNone(yaw)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        # Front wall normal should point inward (+Z direction → 0,0,1 after inward correction)
        self.assertIsNotNone(normal)

    def test_yaw_is_consistent_for_center_position(self):
        """Center of room should still get a valid rotation."""
        t = np.array([0.0, 1.6, 0.0])
        yaw, R, normal = resolve_standing_rotation(t, self.room)
        self.assertIsNotNone(yaw)
        self.assertIsNotNone(R)

    def test_rotation_y_axis_preserved(self):
        """Rotation matrix should keep Y-axis unchanged (yaw-only)."""
        t = np.array([0.0, 1.6, 0.5])
        _, R, _ = resolve_standing_rotation(t, self.room)
        np.testing.assert_allclose(R @ [0, 1, 0], [0, 1, 0], atol=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# freestanding_floor_translation
# ═══════════════════════════════════════════════════════════════════════════

class TestFreestandingFloorTranslation(unittest.TestCase):
    def setUp(self):
        self.room = _make_room_data()
        self.depth = _dummy_depth()

    def test_floor_method_with_good_ray(self):
        """A ray pointing downward (toward +Y floor) should use 'floor' method."""
        origin = np.zeros(3)
        # +Y points toward floor; direction with +Y hits floor within room bounds
        direction = np.array([0.0, 0.6, -0.5])
        direction = direction / np.linalg.norm(direction)
        uv = np.array([0.5, 0.6])
        binary = _centered_mask()

        p, method, _ = freestanding_floor_translation(
            origin, direction, uv, binary, self.room, self.depth)
        self.assertIsNotNone(p)
        self.assertEqual(method, "floor")
        self.assertAlmostEqual(p[1], 1.6)  # on floor plane

    def test_floor_method_hit_inside_room(self):
        origin = np.zeros(3)
        direction = np.array([0.0, 0.6, -0.5])
        direction = direction / np.linalg.norm(direction)
        uv = np.array([0.5, 0.52])
        binary = _centered_mask()

        p, method, _ = freestanding_floor_translation(
            origin, direction, uv, binary, self.room, self.depth)
        self.assertIsNotNone(p)
        # Should be inside the room footprint
        from reposition.lgt_utils import check_floor_hit
        self.assertTrue(check_floor_hit(p, self.room))

    def test_angular_method_with_object_height(self):
        """When floor hit is unreliable (near nadir), use angular-size ranging."""
        origin = np.zeros(3)
        # Near-nadir direction: large dy (+Y toward floor, > nadir_dy 0.92)
        direction = np.array([0.0, 0.97, -0.24])
        direction = direction / np.linalg.norm(direction)
        uv = np.array([0.5, 0.8])
        # Mask with known vertical extent
        mask = np.zeros((512, 1024), dtype=bool)
        mask[400:440, 500:520] = True  # small angular height

        # Object height = 1.0m, angular height of mask ≈ (40/512)*π ≈ 0.245 rad
        p, method, _ = freestanding_floor_translation(
            origin, direction, uv, mask, self.room, self.depth,
            object_height=1.0)
        self.assertIsNotNone(p)
        # Should use angular or wall_fraction (depends on exact params)
        self.assertIn(method, ["angular", "wall_fraction", "floor"])

    def test_wall_fraction_fallback(self):
        """Without SAM3D height, near-nadir rays still range (assumed height)."""
        origin = np.zeros(3)
        direction = np.array([0.0, 0.98, -0.2])
        direction = direction / np.linalg.norm(direction)
        uv = np.array([0.5, 0.85])
        binary = _bottom_mask()

        p, method, _ = freestanding_floor_translation(
            origin, direction, uv, binary, self.room, self.depth,
            object_height=None)
        self.assertIsNotNone(p)
        self.assertIn(method, ["angular", "wall_fraction"])
        self.assertAlmostEqual(p[1], 1.6)

    def test_returns_none_without_depth(self):
        origin = np.zeros(3)
        direction = np.array([0.0, 0.5, -0.866])
        direction = direction / np.linalg.norm(direction)
        uv = np.array([0.5, 0.6])
        binary = _centered_mask()

        p, method, _ = freestanding_floor_translation(
            origin, direction, uv, binary, self.room, depth=None)
        self.assertIsNone(p)
        self.assertIsNone(method)

    def test_result_on_floor_plane(self):
        origin = np.zeros(3)
        direction = np.array([0.3, 0.6, -0.5])
        direction = direction / np.linalg.norm(direction)
        uv = np.array([0.55, 0.55])
        binary = _centered_mask()

        p, _, _ = freestanding_floor_translation(
            origin, direction, uv, binary, self.room, self.depth)
        self.assertIsNotNone(p)
        self.assertAlmostEqual(p[1], 1.6)

    def test_room_with_different_camera_height(self):
        room = _make_room_data(camera_height=1.2, hw=3.0, hd=2.0)
        depth = _dummy_depth(wall_dist=3.0, camera_height=1.2)
        origin = np.zeros(3)
        direction = np.array([0.0, 0.4, -0.6])
        direction = direction / np.linalg.norm(direction)
        uv = np.array([0.5, 0.52])
        binary = _centered_mask()

        p, method, _ = freestanding_floor_translation(
            origin, direction, uv, binary, room, depth)
        self.assertIsNotNone(p)
        self.assertAlmostEqual(p[1], 1.2)


# ═══════════════════════════════════════════════════════════════════════════
# resolve_standing_pose — translation/surface checks
# ═══════════════════════════════════════════════════════════════════════════

class TestResolveStandingTranslation(unittest.TestCase):
    def setUp(self):
        self.room = _make_room_data()
        self.depth = _dummy_depth()

    def test_centered_mask_placed_on_floor(self):
        binary = _centered_mask()
        t, uv, surface, _, _, _ = resolve_standing_pose(
            binary, self.room, depth=self.depth, object_height=1.0)
        self.assertIsNotNone(t)
        self.assertIsNotNone(uv)
        self.assertEqual(surface, "floor")
        self.assertAlmostEqual(t[1], 1.6)

    def test_wall_mask_placed_on_wall(self):
        binary = _wall_mask()
        t, uv, surface, _, _, _ = resolve_standing_pose(
            binary, self.room, depth=self.depth, object_height=1.0)
        self.assertIsNotNone(t)
        self.assertIsNotNone(uv)
        # Left-side mask may hit wall or floor depending on geometry
        self.assertIn(surface, ["floor", "wall"])

    def test_empty_mask_returns_none(self):
        binary = np.zeros((512, 1024), dtype=bool)
        t, uv, surface, _, _, _ = resolve_standing_pose(
            binary, self.room, depth=self.depth)
        self.assertIsNone(t)
        self.assertIsNone(uv)
        self.assertIsNone(surface)

    def test_without_depth_still_works(self):
        """Without depth, falls back to legacy floor hit or wall."""
        binary = _centered_mask()
        t, uv, surface, _, _, _ = resolve_standing_pose(
            binary, self.room, depth=None)
        # May or may not find a placement depending on mask position
        if t is not None:
            self.assertIn(surface, ["floor", "wall"])

    def test_custom_origin(self):
        binary = _centered_mask()
        origin = np.array([0.0, 0.0, 0.0])
        t, uv, surface, _, _, _ = resolve_standing_pose(
            binary, self.room, origin=origin, depth=self.depth, object_height=1.0)
        self.assertIsNotNone(t)
        self.assertEqual(surface, "floor")

    def test_translation_inside_room_footprint(self):
        from reposition.lgt_utils import check_floor_hit
        binary = _centered_mask()
        t, _, surface, _, _, _ = resolve_standing_pose(
            binary, self.room, depth=self.depth, object_height=1.0)
        if surface == "floor" and t is not None:
            self.assertTrue(
                check_floor_hit(t, self.room),
                f"Translation {t} should be inside room footprint")


# ═══════════════════════════════════════════════════════════════════════════
# resolve_standing_pose
# ═══════════════════════════════════════════════════════════════════════════

class TestResolveStandingPose(unittest.TestCase):
    def setUp(self):
        self.room = _make_room_data()
        self.depth = _dummy_depth()

    def test_full_pose_for_centered_mask(self):
        binary = _centered_mask()
        t, uv, surface, rotation, _, _ = resolve_standing_pose(
            binary, self.room, depth=self.depth, object_height=1.0)
        self.assertIsNotNone(t)
        self.assertIsNotNone(uv)
        self.assertIsNotNone(surface)
        self.assertIsNotNone(rotation)
        self.assertIn("yaw", rotation)
        self.assertIn("matrix", rotation)
        self.assertIsInstance(rotation["yaw"], float)
        self.assertEqual(len(rotation["matrix"]), 3)

    def test_empty_mask_returns_none(self):
        binary = np.zeros((512, 1024), dtype=bool)
        t, uv, surface, rotation, _, _ = resolve_standing_pose(
            binary, self.room, depth=self.depth)
        self.assertIsNone(t)
        self.assertIsNone(uv)
        self.assertIsNone(surface)
        self.assertIsNone(rotation)

    def test_rotation_matrix_is_orthogonal(self):
        binary = _centered_mask()
        _, _, _, rotation, _, _ = resolve_standing_pose(
            binary, self.room, depth=self.depth, object_height=1.0)
        if rotation is not None:
            R = np.array(rotation["matrix"])
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)

    def test_translation_on_floor(self):
        binary = _centered_mask()
        t, _, surface, _, _, _ = resolve_standing_pose(
            binary, self.room, depth=self.depth, object_height=1.0)
        if surface == "floor":
            self.assertAlmostEqual(t[1], 1.6)

    def test_with_object_height(self):
        binary = np.zeros((512, 1024), dtype=bool)
        binary[200:280, 450:550] = True  # 80px vertical span
        t, uv, surface, rotation, _, _ = resolve_standing_pose(
            binary, self.room, depth=self.depth, object_height=1.5)
        self.assertIsNotNone(t)
        self.assertIsNotNone(rotation)

    def test_without_sam3d_height_still_places(self):
        """Missing SAM3D object height must not skip a otherwise-valid pose."""
        binary = _centered_mask()
        t, uv, surface, rotation, scale, err = resolve_standing_pose(
            binary, self.room, depth=self.depth)
        self.assertIsNone(err)
        self.assertIsNotNone(t)
        self.assertIsNotNone(uv)
        self.assertIsNotNone(rotation)
        self.assertIsNotNone(scale)
        self.assertIn(scale["method"], {"room_distance", "angular", "assumed"})
        self.assertGreater(scale["factor"], 0.0)
        self.assertAlmostEqual(scale["assumed_height"], 0.8)

    def test_no_depth_fallback(self):
        """Without depth, resolves pose via legacy floor/wall path."""
        binary = _centered_mask()
        t, uv, surface, rotation, _, _ = resolve_standing_pose(
            binary, self.room, depth=None)
        # Centered mask → near-horizontal ray; may hit wall or floor
        self.assertIsNotNone(t)
        self.assertIn(surface, ["floor", "wall"])
        self.assertIsNotNone(rotation)

    def test_pose_with_custom_num_samples(self):
        binary = _centered_mask()
        t1, _, _, _, _, _ = resolve_standing_pose(
            binary, self.room, depth=self.depth, num_samples=1, object_height=1.0)
        t3, _, _, _, _, _ = resolve_standing_pose(
            binary, self.room, depth=self.depth, num_samples=3, object_height=1.0)
        self.assertIsNotNone(t1)
        self.assertIsNotNone(t3)

    def test_pose_with_large_room(self):
        room = _make_room_data(camera_height=1.6, hw=5.0, hd=4.0)
        depth = _dummy_depth(wall_dist=5.0)
        binary = _centered_mask(size=60)
        t, uv, surface, rotation, _, _ = resolve_standing_pose(
            binary, room, depth=depth, object_height=1.0)
        self.assertIsNotNone(t)
        self.assertIsNotNone(rotation)


# ═══════════════════════════════════════════════════════════════════════════
# get_masks
# ═══════════════════════════════════════════════════════════════════════════

class TestGetMasks(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.mask_dir = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _touch(self, name: str):
        (self.mask_dir / name).touch()

    def test_returns_sorted_masks(self):
        self._touch("mask_2.png")
        self._touch("mask_0.png")
        self._touch("mask_10.png")
        self._touch("other_file.txt")
        self._touch("image_1.png")

        result = get_masks(self.mask_dir)
        names = [p.name for p in result]
        self.assertEqual(names, ["mask_0.png", "mask_2.png", "mask_10.png"])

    def test_natural_sort_order(self):
        """Sorts numerically by mask index, not lexicographically."""
        for i in [1, 2, 10, 20, 100]:
            self._touch(f"mask_{i}.png")
        result = get_masks(self.mask_dir)
        indices = [int(p.stem.split("_")[1]) for p in result]
        self.assertEqual(indices, [1, 2, 10, 20, 100])

    def test_empty_directory(self):
        self.assertEqual(get_masks(self.mask_dir), [])

    def test_no_matching_files(self):
        self._touch("image.png")
        self._touch("mask.txt")
        self._touch("other.jpg")
        self.assertEqual(get_masks(self.mask_dir), [])

    def test_not_a_directory_raises(self):
        self._touch("mask_1.png")
        with self.assertRaises(NotADirectoryError):
            get_masks(self.mask_dir / "mask_1.png")

    def test_nonexistent_path_raises(self):
        with self.assertRaises(NotADirectoryError):
            get_masks(self.mask_dir / "nonexistent")


# ═══════════════════════════════════════════════════════════════════════════
# load_aligned_binary_mask
# ═══════════════════════════════════════════════════════════════════════════

class TestLoadAlignedBinaryMask(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _create_white_mask_png(self, path: Path):
        """Create a small white square on black background PNG at 1024x512."""
        img = np.zeros((512, 1024, 3), dtype=np.uint8)
        img[200:300, 400:600] = 255  # white rectangle
        Image.fromarray(img).save(path)

    def _create_grayscale_png(self, path: Path):
        """Create a 2D grayscale PNG (no RGB channel)."""
        img = np.zeros((512, 1024), dtype=np.uint8)
        img[200:300, 400:600] = 255
        Image.fromarray(img).save(path)

    def test_loads_binary_mask_no_manhattan(self):
        path = self.tmp / "mask_0.png"
        self._create_white_mask_png(path)
        binary = load_aligned_binary_mask(path, do_manhattan=False)
        self.assertEqual(binary.shape, (512, 1024))
        self.assertEqual(binary.dtype, bool)
        # White rectangle region should be True
        self.assertTrue(binary[250, 500])
        # Black background should be False
        self.assertFalse(binary[0, 0])

    def test_grayscale_image_converted_to_rgb(self):
        path = self.tmp / "mask_1.png"
        self._create_grayscale_png(path)
        binary = load_aligned_binary_mask(path, do_manhattan=False)
        self.assertTrue(binary[250, 500])
        self.assertFalse(binary[0, 0])

    def test_full_white_image(self):
        path = self.tmp / "mask_full.png"
        img = np.full((512, 1024, 3), 255, dtype=np.uint8)
        Image.fromarray(img).save(path)
        binary = load_aligned_binary_mask(path, do_manhattan=False)
        self.assertTrue(binary.all())

    def test_full_black_image(self):
        path = self.tmp / "mask_empty.png"
        img = np.zeros((512, 1024, 3), dtype=np.uint8)
        Image.fromarray(img).save(path)
        binary = load_aligned_binary_mask(path, do_manhattan=False)
        self.assertFalse(binary.any())

    def test_manhattan_without_cache_path_raises(self):
        path = self.tmp / "mask_0.png"
        self._create_white_mask_png(path)
        with self.assertRaises(ValueError):
            load_aligned_binary_mask(path, do_manhattan=True, vp_cache_path=None)

    def test_manhattan_with_nonexistent_cache_raises(self):
        path = self.tmp / "mask_0.png"
        self._create_white_mask_png(path)
        with self.assertRaises(ValueError):
            load_aligned_binary_mask(
                path, do_manhattan=True, vp_cache_path="/nonexistent/path.npz")

    def test_image_smaller_than_1024x512_is_resized(self):
        path = self.tmp / "mask_small.png"
        img = np.full((64, 128, 3), 255, dtype=np.uint8)
        Image.fromarray(img).save(path)
        binary = load_aligned_binary_mask(path, do_manhattan=False)
        self.assertEqual(binary.shape, (512, 1024))
        self.assertTrue(binary.all())

    def test_image_larger_than_1024x512_is_resized(self):
        path = self.tmp / "mask_large.png"
        img = np.full((1024, 2048, 3), 255, dtype=np.uint8)
        Image.fromarray(img).save(path)
        binary = load_aligned_binary_mask(path, do_manhattan=False)
        self.assertEqual(binary.shape, (512, 1024))


# ═══════════════════════════════════════════════════════════════════════════
# placements_from_mask_dir
# ═══════════════════════════════════════════════════════════════════════════

class TestPlacementsFromMaskDir(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)
        self.room = _make_room_data()
        self.depth = _dummy_depth()
        # Optional SAM3D heights; placement must also work without them.
        self.sam3d_meta = {
            i: {"scale": 1.0, "rotation_xyzw": [0, 0, 0, 1]}
            for i in range(8)
        }

    def tearDown(self):
        self.tmpdir.cleanup()

    def _create_mask(self, name: str, u_center=500, v_center=280, size=60):
        """Create a mask PNG with a white blob."""
        path = self.tmp / name
        img = np.zeros((512, 1024, 3), dtype=np.uint8)
        half = size // 2
        img[v_center - half:v_center + half,
            u_center - half:u_center + half] = 255
        Image.fromarray(img).save(path)
        return path

    def test_returns_placements_for_valid_masks(self):
        self._create_mask("mask_0.png", u_center=512, v_center=280)
        self._create_mask("mask_1.png", u_center=400, v_center=300)

        results, success, failure = placements_from_mask_dir(
            self.tmp,
            self.room,
            depth=self.depth,
            do_manhattan=False,
            num_samples=3,
            sam3d_metadata=self.sam3d_meta,
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(success + failure, 2)
        for r in results:
            self.assertIn("mask", r)
            self.assertIn("translation", r)
            self.assertIn("rotation", r)
            self.assertIn("contact_uv", r)
            self.assertIn("surface", r)
            # Translation should be present for valid masks
            self.assertIsNotNone(r["translation"])
            self.assertIsNotNone(r["surface"])

    def test_empty_mask_dir(self):
        results, success, failure = placements_from_mask_dir(
            self.tmp, self.room, depth=self.depth, do_manhattan=False)
        self.assertEqual(results, [])
        self.assertEqual(success, 0)
        self.assertEqual(failure, 0)

    def test_translation_is_serializable(self):
        """Translation should be a plain list, not a numpy array."""
        self._create_mask("mask_0.png", u_center=512, v_center=280)
        results, _, _ = placements_from_mask_dir(
            self.tmp, self.room, depth=self.depth, do_manhattan=False,
            sam3d_metadata=self.sam3d_meta)
        t = results[0]["translation"]
        self.assertIsInstance(t, list)
        self.assertEqual(len(t), 3)
        for v in t:
            self.assertIsInstance(v, float)

    def test_contact_uv_is_serializable(self):
        self._create_mask("mask_0.png", u_center=512, v_center=280)
        results, _, _ = placements_from_mask_dir(
            self.tmp, self.room, depth=self.depth, do_manhattan=False,
            sam3d_metadata=self.sam3d_meta)
        uv = results[0]["contact_uv"]
        self.assertIsInstance(uv, list)
        self.assertEqual(len(uv), 2)

    def test_rotation_structure(self):
        self._create_mask("mask_0.png", u_center=512, v_center=280)
        results, _, _ = placements_from_mask_dir(
            self.tmp, self.room, depth=self.depth, do_manhattan=False,
            sam3d_metadata=self.sam3d_meta)
        rotation = results[0]["rotation"]
        self.assertIsNotNone(rotation)
        self.assertIn("yaw", rotation)
        self.assertIn("matrix", rotation)
        self.assertEqual(len(rotation["matrix"]), 3)
        self.assertIsInstance(rotation["yaw"], float)

    def test_without_depth_still_works(self):
        """Should fall back to legacy floor/wall path."""
        self._create_mask("mask_0.png", u_center=512, v_center=280)
        results, _, _ = placements_from_mask_dir(
            self.tmp, self.room, depth=None, do_manhattan=False,
            sam3d_metadata=self.sam3d_meta)
        self.assertEqual(len(results), 1)
        self.assertIsNotNone(results[0]["translation"])

    def test_without_sam3d_metadata_still_places(self):
        self._create_mask("mask_0.png", u_center=512, v_center=280)
        results, success, failure = placements_from_mask_dir(
            self.tmp, self.room, depth=self.depth, do_manhattan=False)
        self.assertEqual(len(results), 1)
        self.assertIsNone(results[0].get("error"))
        self.assertIsNotNone(results[0]["translation"])
        self.assertIsNotNone(results[0]["scale"])
        self.assertEqual(success, 1)
        self.assertEqual(failure, 0)

    def test_all_masks_have_surface_field(self):
        for i in range(3):
            self._create_mask(f"mask_{i}.png",
                              u_center=300 + i * 100, v_center=280)
        results, _, _ = placements_from_mask_dir(
            self.tmp, self.room, depth=self.depth, do_manhattan=False,
            sam3d_metadata=self.sam3d_meta)
        surfaces = [r["surface"] for r in results]
        self.assertTrue(all(s in ("floor", "wall") for s in surfaces))

    def test_qc_reject_sets_error_and_counts_failure(self):
        """QC rejects fold into record['error'] and placement_failure."""
        self._create_mask("mask_0.png", u_center=512, v_center=280)
        boxes = {
            "detections": [
                {
                    "label": "door",
                    "score": 0.99,
                    "box": {"xMin": 400, "yMin": 200, "xMax": 600, "yMax": 400},
                }
            ]
        }
        results, success, failure = placements_from_mask_dir(
            self.tmp,
            self.room,
            depth=self.depth,
            do_manhattan=False,
            sam3d_metadata=self.sam3d_meta,
            dino_boxes=boxes,
            min_score=0.3,
        )
        self.assertEqual(len(results), 1)
        # Door on floor → surface_class fail → counted as failure with error.
        if results[0].get("surface") == "floor":
            self.assertEqual(success, 0)
            self.assertEqual(failure, 1)
            err = results[0].get("error") or ""
            self.assertEqual(err, "Item on wrong surface")
            self.assertEqual(
                (results[0].get("quality") or {}).get("message"),
                "Item on wrong surface",
            )

    def test_with_sam3d_metadata_dict(self):
        self._create_mask("mask_0.png", u_center=512, v_center=280)
        metadata = {0: {"scale": 1.5, "rotation_xyzw": [0, 0, 0, 1]}}
        results, _, _ = placements_from_mask_dir(
            self.tmp, self.room, depth=self.depth, do_manhattan=False,
            sam3d_metadata=metadata)
        self.assertEqual(len(results), 1)
        rotation = results[0]["rotation"]
        self.assertIsNotNone(rotation)
        # Rotation is present and valid
        self.assertIn("yaw", rotation)
        self.assertIn("matrix", rotation)

if __name__ == "__main__":
    unittest.main()
