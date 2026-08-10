"""Unit tests for reposition.sam_utils — quaternion / rotation / SAM3D helpers."""

import json
import os
import tempfile
import unittest

import numpy as np

from reposition.sam_utils import (
    _horizontal_dir,
    load_sam3d_metadata,
    mask_index_from_name,
    merge_sam3d_orientations_into_placements,
    object_height_from_sam3d,
    quat_xyzw_to_rotation_matrix,
    rotation_about_axis,
    rotation_aligning_vectors,
    sam3d_orientation_to_obj3d,
    signed_angle_about_axis,
)


# ═══════════════════════════════════════════════════════════════════════════
# quat_xyzw_to_rotation_matrix
# ═══════════════════════════════════════════════════════════════════════════

class TestQuatToRotationMatrix(unittest.TestCase):
    def test_identity(self):
        R = quat_xyzw_to_rotation_matrix([0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_90_degree_z(self):
        angle = np.pi / 2
        q = [0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)]
        R = quat_xyzw_to_rotation_matrix(q)
        np.testing.assert_allclose(
            R @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-10)

    def test_90_degree_y(self):
        angle = np.pi / 2
        q = [0.0, np.sin(angle / 2), 0.0, np.cos(angle / 2)]
        R = quat_xyzw_to_rotation_matrix(q)
        np.testing.assert_allclose(
            R @ [1.0, 0.0, 0.0], [0.0, 0.0, -1.0], atol=1e-10)

    def test_90_degree_x(self):
        angle = np.pi / 2
        q = [np.sin(angle / 2), 0.0, 0.0, np.cos(angle / 2)]
        R = quat_xyzw_to_rotation_matrix(q)
        np.testing.assert_allclose(
            R @ [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], atol=1e-10)

    def test_180_degree(self):
        q = [0.0, 1.0, 0.0, 0.0]
        R = quat_xyzw_to_rotation_matrix(q)
        np.testing.assert_allclose(
            R @ [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(
            R @ [0.0, 0.0, 1.0], [0.0, 0.0, -1.0], atol=1e-10)

    def test_is_orthogonal(self):
        q = [0.3, 0.5, 0.1, np.sqrt(1 - 0.09 - 0.25 - 0.01)]
        R = quat_xyzw_to_rotation_matrix(q)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_non_unit_normalized(self):
        q = [0.0, 0.0, 2.0, 0.0]
        R = quat_xyzw_to_rotation_matrix(q)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_near_zero_norm_raises(self):
        with self.assertRaises(ValueError):
            quat_xyzw_to_rotation_matrix([0.0, 0.0, 0.0, 0.0])

    def test_list_input(self):
        R = quat_xyzw_to_rotation_matrix([0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_tuple_input(self):
        R = quat_xyzw_to_rotation_matrix((0.0, 0.0, 0.0, 1.0))
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# rotation_aligning_vectors
# ═══════════════════════════════════════════════════════════════════════════

class TestRotationAligningVectors(unittest.TestCase):
    def test_same_direction_identity(self):
        R = rotation_aligning_vectors(
            np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_opposite_direction_180(self):
        R = rotation_aligning_vectors(
            np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]))
        np.testing.assert_allclose(
            R @ [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], atol=1e-10)

    def test_90_degree_rotation(self):
        R = rotation_aligning_vectors(
            np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(
            R @ [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], atol=1e-10)

    def test_result_is_orthogonal(self):
        R = rotation_aligning_vectors(
            np.array([0.6, 0.8, 0.0]),
            np.array([-0.3, 0.4, np.sqrt(1 - 0.25)]))
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_unnormalized_inputs(self):
        R = rotation_aligning_vectors(
            np.array([5.0, 0.0, 0.0]), np.array([0.0, 3.0, 0.0]))
        np.testing.assert_allclose(
            R @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# signed_angle_about_axis
# ═══════════════════════════════════════════════════════════════════════════

class TestSignedAngleAboutAxis(unittest.TestCase):
    def test_zero_angle(self):
        a = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(
            signed_angle_about_axis(a, a, [0.0, 1.0, 0.0]), 0.0)

    def test_positive_90_around_y(self):
        ang = signed_angle_about_axis(
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 1.0, 0.0]))
        self.assertAlmostEqual(ang, -np.pi / 2)

    def test_negative_90_around_y(self):
        ang = signed_angle_about_axis(
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]))
        self.assertAlmostEqual(ang, np.pi / 2)

    def test_parallel_to_axis_ignored(self):
        ang = signed_angle_about_axis(
            np.array([1.0, 5.0, 0.0]),
            np.array([0.0, 3.0, 1.0]),
            np.array([0.0, 1.0, 0.0]))
        self.assertAlmostEqual(ang, -np.pi / 2)


# ═══════════════════════════════════════════════════════════════════════════
# rotation_about_axis
# ═══════════════════════════════════════════════════════════════════════════

class TestRotationAboutAxis(unittest.TestCase):
    def test_zero_angle_identity(self):
        R = rotation_about_axis(np.array([1.0, 0.0, 0.0]), 0.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_90_around_y(self):
        R = rotation_about_axis(np.array([0.0, 1.0, 0.0]), np.pi / 2)
        np.testing.assert_allclose(
            R @ [1.0, 0.0, 0.0], [0.0, 0.0, -1.0], atol=1e-10)

    def test_90_around_z(self):
        R = rotation_about_axis(np.array([0.0, 0.0, 1.0]), np.pi / 2)
        np.testing.assert_allclose(
            R @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-10)

    def test_is_orthogonal(self):
        import math
        for angle in [0.0, 0.5, 1.0, -2.0, math.pi]:
            for axis in [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 2, 3]]:
                R = rotation_about_axis(np.array(axis, dtype=float), angle)
                np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)

    def test_unnormalized_axis(self):
        R = rotation_about_axis(np.array([0.0, 3.0, 0.0]), np.pi / 2)
        np.testing.assert_allclose(
            R @ [1.0, 0.0, 0.0], [0.0, 0.0, -1.0], atol=1e-10)

    def test_composes_correctly(self):
        R1 = rotation_about_axis(np.array([0, 1, 0]), np.deg2rad(30))
        R2 = rotation_about_axis(np.array([0, 1, 0]), np.deg2rad(60))
        R90 = rotation_about_axis(np.array([0, 1, 0]), np.deg2rad(90))
        np.testing.assert_allclose(R2 @ R1, R90, atol=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# sam3d_orientation_to_obj3d
# ═══════════════════════════════════════════════════════════════════════════

class TestSam3DOrientationToObj3D(unittest.TestCase):
    def test_identity_quaternion(self):
        R, info = sam3d_orientation_to_obj3d([0.0, 0.0, 0.0, 1.0])
        self.assertEqual(info["source"], "sam3d")
        self.assertAlmostEqual(info["yaw"], 0.0, delta=1e-10)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_yaw_only_is_preserved(self):
        yaw = np.deg2rad(45)
        q = [0.0, np.sin(yaw / 2), 0.0, np.cos(yaw / 2)]
        R, info = sam3d_orientation_to_obj3d(q)
        self.assertAlmostEqual(info["yaw"], yaw)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)

    def test_forward_is_horizontal_only(self):
        pitch = np.deg2rad(30)
        q_x = [np.sin(pitch / 2), 0.0, 0.0, np.cos(pitch / 2)]
        _, info = sam3d_orientation_to_obj3d(q_x)
        self.assertAlmostEqual(info["yaw"], 0.0, delta=1e-10)

    def test_result_is_y_axis_rotation(self):
        q = [0.1, 0.2, 0.3, np.sqrt(1 - 0.01 - 0.04 - 0.09)]
        R, _ = sam3d_orientation_to_obj3d(q)
        np.testing.assert_allclose(
            R @ [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], atol=1e-10)

    def test_no_forward_component(self):
        q = [np.sin(np.pi / 4), 0.0, 0.0, np.cos(np.pi / 4)]
        R, info = sam3d_orientation_to_obj3d(q)
        self.assertEqual(info["source"], "sam3d")
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# mask_index_from_name
# ═══════════════════════════════════════════════════════════════════════════

class TestMaskIndexFromName(unittest.TestCase):
    def test_standard_name(self):
        self.assertEqual(mask_index_from_name("mask_12.png"), 12)

    def test_zero(self):
        self.assertEqual(mask_index_from_name("mask_0.png"), 0)

    def test_large_index(self):
        self.assertEqual(mask_index_from_name("mask_999.png"), 999)

    def test_path_with_dir(self):
        self.assertEqual(mask_index_from_name("/some/dir/mask_42.png"), 42)

    def test_wrong_prefix(self):
        self.assertIsNone(mask_index_from_name("image_12.png"))

    def test_no_extension(self):
        self.assertIsNone(mask_index_from_name("mask_7"))

    def test_wrong_extension(self):
        self.assertIsNone(mask_index_from_name("mask_3.jpg"))

    def test_non_numeric_index(self):
        self.assertIsNone(mask_index_from_name("mask_abc.png"))

    def test_uppercase(self):
        self.assertEqual(mask_index_from_name("MASK_42.PNG"), 42)


# ═══════════════════════════════════════════════════════════════════════════
# load_sam3d_metadata
# ═══════════════════════════════════════════════════════════════════════════

class TestLoadSam3DMetadata(unittest.TestCase):
    def test_basic_parsing(self):
        payload = [
            {"object_index": 0, "rotation": [0.1, 0.2, 0.3, 0.4], "scale": 1.5},
            {"object_index": 1, "rotation": [0.5, 0.6, 0.7, 0.8]},
            {"object_index": 3},
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(payload, f)
            tmp = f.name

        try:
            meta = load_sam3d_metadata(tmp)
            self.assertEqual(meta[0]["rotation_xyzw"], [0.1, 0.2, 0.3, 0.4])
            self.assertEqual(meta[1]["rotation_xyzw"], [0.5, 0.6, 0.7, 0.8])
            self.assertIsNone(meta[3]["rotation_xyzw"])
            self.assertEqual(meta[0]["scale"], 1.5)
        finally:
            os.unlink(tmp)

    def test_missing_object_index(self):
        payload = [{"name": "chair"}]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(payload, f)
            tmp = f.name

        try:
            meta = load_sam3d_metadata(tmp)
            self.assertIn(0, meta)
            self.assertIsNone(meta[0]["rotation_xyzw"])
        finally:
            os.unlink(tmp)

    def test_not_a_list_raises(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump({"not": "a list"}, f)
            tmp = f.name

        try:
            with self.assertRaises(ValueError):
                load_sam3d_metadata(tmp)
        finally:
            os.unlink(tmp)

    def test_nested_rotation_list(self):
        payload = [{"object_index": 0, "rotation": [[0.0, 0.0, 0.0, 1.0]]}]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(payload, f)
            tmp = f.name

        try:
            meta = load_sam3d_metadata(tmp)
            self.assertEqual(meta[0]["rotation_xyzw"], [0.0, 0.0, 0.0, 1.0])
        finally:
            os.unlink(tmp)


# ═══════════════════════════════════════════════════════════════════════════
# merge_sam3d_orientations_into_placements
# ═══════════════════════════════════════════════════════════════════════════

class TestMergeSam3DOrientationsIntoPlacements(unittest.TestCase):
    @staticmethod
    def _sample_placements():
        return [
            {"mask": "mask_0.png", "translation": [1.0, 1.6, 2.0],
             "rotation": {"yaw": 0.5, "source": "lgt"}},
            {"mask": "mask_1.png", "translation": [0.0, 1.6, 1.0],
             "rotation": None},
            {"mask": "mask_2.png", "translation": [-1.0, 1.6, 1.5]},
        ]

    @staticmethod
    def _sample_metadata():
        return {
            0: {"rotation_xyzw": [0.0, 0.0, 0.0, 1.0]},
            1: {"rotation_xyzw": [0.0, np.sin(np.pi / 8), 0.0, np.cos(np.pi / 8)]},
        }

    def test_replaces_lgt_rotation(self):
        merged = merge_sam3d_orientations_into_placements(
            self._sample_placements(), self._sample_metadata())
        self.assertEqual(merged[0]["rotation"]["source"], "sam3d")
        self.assertEqual(merged[0]["rotation"]["frame"], "obj3d")
        self.assertIn("yaw", merged[0]["rotation"])

    def test_adds_rotation_to_null(self):
        merged = merge_sam3d_orientations_into_placements(
            self._sample_placements(), self._sample_metadata())
        self.assertIsNotNone(merged[1]["rotation"])
        self.assertEqual(merged[1]["rotation"]["source"], "sam3d")

    def test_keeps_missing_rotation(self):
        merged = merge_sam3d_orientations_into_placements(
            self._sample_placements(), self._sample_metadata())
        self.assertNotIn("rotation", merged[2])

    def test_preserves_translation(self):
        merged = merge_sam3d_orientations_into_placements(
            self._sample_placements(), self._sample_metadata())
        for i, orig in enumerate(self._sample_placements()):
            if orig["translation"] is not None:
                np.testing.assert_allclose(
                    merged[i]["translation"], orig["translation"])

    def test_none_metadata_passthrough(self):
        merged = merge_sam3d_orientations_into_placements(
            self._sample_placements(), None)
        self.assertEqual(merged, self._sample_placements())

    def test_metadata_path(self):
        payload = [{"object_index": 0, "rotation": [0.0, 0.0, 0.0, 1.0]}]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(payload, f)
            tmp = f.name

        try:
            placements = [{"mask": "mask_0.png", "translation": [1, 1.6, 2]}]
            merged = merge_sam3d_orientations_into_placements(placements, tmp)
            self.assertEqual(merged[0]["rotation"]["source"], "sam3d")
        finally:
            os.unlink(tmp)


# ═══════════════════════════════════════════════════════════════════════════
# object_height_from_sam3d
# ═══════════════════════════════════════════════════════════════════════════

class TestObjectHeightFromSam3D(unittest.TestCase):
    def test_returns_none_with_no_args(self):
        self.assertIsNone(object_height_from_sam3d(0))

    def test_metadata_scale(self):
        metadata = {0: {"scale": 1.8}}
        self.assertAlmostEqual(
            object_height_from_sam3d(0, metadata=metadata), 1.8)

    def test_metadata_nested_scale(self):
        metadata = {5: {"scale": [[2.5]]}}
        self.assertAlmostEqual(
            object_height_from_sam3d(5, metadata=metadata), 2.5)

    def test_missing_object_index(self):
        metadata = {0: {"scale": 1.0}}
        self.assertIsNone(object_height_from_sam3d(99, metadata=metadata))

    def test_zero_scale_returns_none(self):
        metadata = {0: {"scale": 0.0}}
        self.assertIsNone(object_height_from_sam3d(0, metadata=metadata))

    def test_no_scale_key(self):
        metadata = {0: {"name": "chair"}}
        self.assertIsNone(object_height_from_sam3d(0, metadata=metadata))

    def test_nonexistent_furniture_dir(self):
        self.assertIsNone(
            object_height_from_sam3d(0, furniture_dir="/nonexistent/path"))


# ═══════════════════════════════════════════════════════════════════════════
# _horizontal_dir
# ═══════════════════════════════════════════════════════════════════════════

class TestHorizontalDir(unittest.TestCase):
    def test_already_horizontal(self):
        d = _horizontal_dir(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
        self.assertIsNotNone(d)
        np.testing.assert_allclose(d, [1.0, 0.0, 0.0], atol=1e-10)

    def test_vertical_component_removed(self):
        d = _horizontal_dir(np.array([1.0, 3.0, 0.0]), np.array([0.0, 1.0, 0.0]))
        self.assertIsNotNone(d)
        self.assertAlmostEqual(d[1], 0.0)
        np.testing.assert_allclose(np.linalg.norm(d), 1.0, atol=1e-10)

    def test_unit_length(self):
        d = _horizontal_dir(np.array([3.0, 7.0, 0.0]), np.array([0.0, 1.0, 0.0]))
        self.assertIsNotNone(d)
        np.testing.assert_allclose(np.linalg.norm(d), 1.0, atol=1e-10)

    def test_custom_up_axis(self):
        d = _horizontal_dir(np.array([1.0, 0.0, 3.0]), np.array([0.0, 0.0, 1.0]))
        self.assertIsNotNone(d)
        self.assertAlmostEqual(d[2], 0.0)
        self.assertAlmostEqual(d[1], 0.0)

    def test_pure_vertical_returns_none(self):
        d = _horizontal_dir(np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0]))
        self.assertIsNone(d)

    def test_zero_vector_returns_none(self):
        d = _horizontal_dir(np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
        self.assertIsNone(d)

    def test_oblique_up_axis(self):
        d = _horizontal_dir(
            np.array([1.0, 1.0, 0.0]),
            np.array([0.0, 1.0, 1.0]) / np.sqrt(2))
        self.assertIsNotNone(d)
        # Should be orthogonal to the up axis
        up = np.array([0.0, 1.0, 1.0]) / np.sqrt(2)
        self.assertAlmostEqual(np.dot(d, up), 0.0, delta=1e-10)


if __name__ == "__main__":
    unittest.main()
