"""Unit tests for reposition.lgt_utils — pure geometry functions."""

import numpy as np
import unittest

from reposition.lgt_utils import (
    _normalize,
    _point_on_segment_xz,
    _wall_y_bounds,
    azimuth_unit_xz,
    check_floor_hit,
    closer_than_wall,
    contact_uvs_from_mask,
    floor_polygon_from_layout,
    horizontal_range,
    horizontal_wall_normal,
    inward_wall_normal,
    json_frame_to_obj3d,
    json_rotation_to_obj3d,
    mask_angular_height,
    nearest_wall,
    patch_index_from_uv,
    place_on_floor_at_range,
    placement_to_obj3d_frame,
    point_in_floor_polygon,
    range_from_angular_size,
    ray_intersect_floor,
    ray_intersect_wall,
    rotation_matrix_yaw,
    shrink_range_into_footprint,
    solve_t,
    solve_t_plane,
    to_json_frame,
    uv2equirectangular,
    wall_distance_from_depth,
    wall_fraction_from_elevation,
    yaw_from_wall_normal,
)


# ---------------------------------------------------------------------------
# Helpers
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
        })
    return {
        "cameraHeight": camera_height,
        "cameraCeilingHeight": 1.0,
        "layoutHeight": 2.6,
        "layoutPoints": {"points": pts},
        "layoutWalls": {"walls": walls},
    }


def _dummy_depth():
    """256-bin depth; wall at 2.0 m (plan_y units)."""
    return np.full(256, 2.0 / 1.6, dtype=np.float64)


def _small_mask():
    mask = np.zeros((512, 1024), dtype=bool)
    mask[200:300, 400:600] = True
    return mask


def _empty_mask():
    return np.zeros((512, 1024), dtype=bool)


# ═══════════════════════════════════════════════════════════════════════════
# _normalize
# ═══════════════════════════════════════════════════════════════════════════

class TestNormalize(unittest.TestCase):
    def test_unit_vector_unchanged(self):
        v = np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(_normalize(v), v)

    def test_scale_invariant(self):
        np.testing.assert_allclose(
            _normalize(np.array([3.0, 4.0, 0.0])), [0.6, 0.8, 0.0])

    def test_negative_vector(self):
        r = _normalize(np.array([-1.0, -1.0, -1.0]))
        np.testing.assert_allclose(np.linalg.norm(r), 1.0)
        self.assertTrue((r < 0).all())

    def test_near_zero_raises(self):
        with self.assertRaises(ValueError):
            _normalize(np.array([0.0, 0.0, 0.0]))

    def test_very_small_raises(self):
        with self.assertRaises(ValueError):
            _normalize(np.array([1e-13, 1e-13, 1e-13]))


# ═══════════════════════════════════════════════════════════════════════════
# to_json_frame
# ═══════════════════════════════════════════════════════════════════════════

class TestToJsonFrame(unittest.TestCase):
    def test_forward_direction(self):
        np.testing.assert_allclose(
            to_json_frame(np.array([0.0, 0.0, 1.0])), [0.0, 0.0, -1.0], atol=1e-6)

    def test_right_direction(self):
        np.testing.assert_allclose(
            to_json_frame(np.array([1.0, 0.0, 0.0])), [1.0, 0.0, 0.0], atol=1e-6)

    def test_y_preserved(self):
        np.testing.assert_allclose(
            to_json_frame(np.array([0.0, 2.5, 0.0])), [0.0, 2.5, 0.0], atol=1e-10)

    def test_output_is_unit_when_input_is_unit(self):
        v = np.array([0.6, 0.0, 0.8])
        r = to_json_frame(v)
        np.testing.assert_allclose(np.linalg.norm(r), 1.0, atol=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# json_frame_to_obj3d / json_rotation_to_obj3d / placement_to_obj3d_frame
# ═══════════════════════════════════════════════════════════════════════════

class TestJsonFrameToObj3D(unittest.TestCase):
    def test_origin(self):
        np.testing.assert_allclose(
            json_frame_to_obj3d(np.array([0.0, 0.0, 0.0])), [0.0, 0.0, 0.0])

    def test_x_axis_maps_to_z(self):
        np.testing.assert_allclose(
            json_frame_to_obj3d(np.array([1.0, 0.0, 0.0])), [0.0, 0.0, 1.0])

    def test_y_axis_maps_to_negative_y(self):
        np.testing.assert_allclose(
            json_frame_to_obj3d(np.array([0.0, 2.0, 0.0])), [0.0, -2.0, 0.0])

    def test_z_axis_maps_to_negative_x(self):
        np.testing.assert_allclose(
            json_frame_to_obj3d(np.array([0.0, 0.0, 3.0])), [-3.0, 0.0, 0.0])


class TestJsonRotationToObj3D(unittest.TestCase):
    def test_identity_remains_identity(self):
        R = np.eye(3)
        np.testing.assert_allclose(
            json_rotation_to_obj3d(R), np.eye(3), atol=1e-10)

    def test_is_conjugate(self):
        rng = np.random.RandomState(42)
        R = rotation_matrix_yaw(float(rng.uniform(-np.pi, np.pi)))
        T = np.array([[0, 0, -1], [0, -1, 0], [1, 0, 0]], dtype=np.float64)
        expected = T @ R @ T.T
        np.testing.assert_allclose(json_rotation_to_obj3d(R), expected, atol=1e-10)


class TestPlacementToObj3DFrame(unittest.TestCase):
    def test_translation_only(self):
        t, r = placement_to_obj3d_frame(
            np.array([1.0, 1.6, 2.0]), rotation=None)
        np.testing.assert_allclose(t, [-2.0, -1.6, 1.0])
        self.assertIsNone(r)

    def test_with_rotation(self):
        R_json = rotation_matrix_yaw(0.0)
        t, r = placement_to_obj3d_frame(
            np.array([0.0, 1.6, 0.0]), rotation=R_json)
        np.testing.assert_allclose(t, [0.0, -1.6, 0.0])
        np.testing.assert_allclose(r, np.eye(3), atol=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# solve_t / solve_t_plane
# ═══════════════════════════════════════════════════════════════════════════

class TestSolveT(unittest.TestCase):
    def test_horizontal_ray_hits_floor(self):
        t = solve_t(0.0, np.array([0.0, 2.0, 0.0]), np.array([0.0, -1.0, 0.0]))
        self.assertAlmostEqual(t, 2.0)

    def test_ray_from_origin(self):
        t = solve_t(1.6, np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
        self.assertAlmostEqual(t, 1.6)

    def test_ray_parallel_to_floor_raises(self):
        with self.assertRaises(ValueError):
            solve_t(0.0, np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))

    def test_ray_away_from_floor_raises(self):
        with self.assertRaises(ValueError):
            solve_t(1.6, np.array([0.0, 2.0, 0.0]), np.array([0.0, 1.0, 0.0]))


class TestSolveTPlane(unittest.TestCase):
    def test_vertical_plane(self):
        t = solve_t_plane(
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0, -3.0]))
        self.assertAlmostEqual(t, 3.0)

    def test_oblique_ray(self):
        t = solve_t_plane(
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 0.0]),
            np.array([1.0, 0.0, 0.0, -5.0]))
        self.assertAlmostEqual(t, 5.0)

    def test_ray_parallel_to_plane_raises(self):
        with self.assertRaises(ValueError):
            solve_t_plane(
                np.array([0.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([1.0, 0.0, 0.0, -1.0]))

    def test_ray_away_from_plane_raises(self):
        with self.assertRaises(ValueError):
            solve_t_plane(
                np.array([5.0, 0.0, 0.0]),
                np.array([1.0, 0.0, 0.0]),
                np.array([1.0, 0.0, 0.0, -3.0]))


# ═══════════════════════════════════════════════════════════════════════════
# horizontal_range
# ═══════════════════════════════════════════════════════════════════════════

class TestHorizontalRange(unittest.TestCase):
    def test_origin(self):
        self.assertEqual(horizontal_range(np.array([0.0, 5.0, 0.0])), 0.0)

    def test_unit_xz(self):
        self.assertAlmostEqual(
            horizontal_range(np.array([3.0, 999.0, 4.0])), 5.0)

    def test_negative_coordinates(self):
        self.assertAlmostEqual(
            horizontal_range(np.array([-3.0, 0.0, -4.0])), 5.0)


# ═══════════════════════════════════════════════════════════════════════════
# horizontal_wall_normal / inward_wall_normal / yaw_from_wall_normal
# ═══════════════════════════════════════════════════════════════════════════

class TestHorizontalWallNormal(unittest.TestCase):
    def test_already_horizontal(self):
        n = np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(horizontal_wall_normal(n), n)

    def test_vertical_component_removed(self):
        r = horizontal_wall_normal(np.array([1.0, 2.0, 0.0]))
        np.testing.assert_allclose(r, [1.0, 0.0, 0.0])

    def test_pure_vertical_raises(self):
        with self.assertRaises(ValueError):
            horizontal_wall_normal(np.array([0.0, 1.0, 0.0]))

    def test_zero_vector_raises(self):
        with self.assertRaises(ValueError):
            horizontal_wall_normal(np.array([0.0, 0.0, 0.0]))


class TestInwardWallNormal(unittest.TestCase):
    def test_already_inward(self):
        r = inward_wall_normal(
            np.array([-1.0, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0]),
            np.zeros(3))
        np.testing.assert_allclose(r, [-1.0, 0.0, 0.0])

    def test_outward_flipped(self):
        r = inward_wall_normal(
            np.array([1.0, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0]),
            np.zeros(3))
        np.testing.assert_allclose(r, [-1.0, 0.0, 0.0])

    def test_custom_room_center(self):
        r = inward_wall_normal(
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 5.0]),
            np.array([1.0, 0.0, 3.0]))
        np.testing.assert_allclose(r, [0.0, 0.0, -1.0])

    def test_vertical_component_stripped(self):
        r = inward_wall_normal(
            np.array([0.0, 999.0, 1.0]),
            np.array([0.0, 0.0, 5.0]),
            np.zeros(3))
        np.testing.assert_allclose(r, [0.0, 0.0, -1.0])


class TestYawFromWallNormal(unittest.TestCase):
    def test_front_wall(self):
        self.assertAlmostEqual(
            yaw_from_wall_normal(np.array([0.0, 0.0, 1.0])), 0.0)

    def test_left_wall(self):
        self.assertAlmostEqual(
            yaw_from_wall_normal(np.array([1.0, 0.0, 0.0])), np.pi / 2)

    def test_right_wall(self):
        self.assertAlmostEqual(
            yaw_from_wall_normal(np.array([-1.0, 0.0, 0.0])), -np.pi / 2)

    def test_back_wall(self):
        yaw = yaw_from_wall_normal(np.array([0.0, 0.0, -1.0]))
        self.assertLess(abs(abs(yaw) - np.pi), 1e-10)

    def test_vertical_component_ignored(self):
        self.assertAlmostEqual(
            yaw_from_wall_normal(np.array([0.0, 5.0, 1.0])), 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# rotation_matrix_yaw
# ═══════════════════════════════════════════════════════════════════════════

class TestRotationMatrixYaw(unittest.TestCase):
    def test_zero_yaw_is_identity(self):
        np.testing.assert_allclose(
            rotation_matrix_yaw(0.0), np.eye(3), atol=1e-10)

    def test_orthogonal(self):
        for yaw in [0.0, np.pi / 4, np.pi / 2, -np.pi / 3, np.pi]:
            R = rotation_matrix_yaw(yaw)
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
            np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_pi_over_2(self):
        R = rotation_matrix_yaw(np.pi / 2)
        np.testing.assert_allclose(R @ [1, 0, 0], [0, 0, -1], atol=1e-10)
        np.testing.assert_allclose(R @ [0, 0, 1], [1, 0, 0], atol=1e-10)

    def test_y_axis_fixed(self):
        for yaw in [0.0, 0.5, 1.0, -2.0]:
            R = rotation_matrix_yaw(yaw)
            np.testing.assert_allclose(R @ [0, 1, 0], [0, 1, 0], atol=1e-10)

    def test_composes_correctly(self):
        R1 = rotation_matrix_yaw(0.3)
        R2 = rotation_matrix_yaw(0.5)
        R_sum = rotation_matrix_yaw(0.8)
        np.testing.assert_allclose(R2 @ R1, R_sum, atol=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# mask_angular_height / range_from_angular_size
# ═══════════════════════════════════════════════════════════════════════════

class TestMaskAngularHeight(unittest.TestCase):
    def test_full_image_spans_pi(self):
        mask = np.ones((512, 1024), dtype=bool)
        self.assertAlmostEqual(mask_angular_height(mask), np.pi, delta=0.01)

    def test_half_image(self):
        mask = np.zeros((512, 1024), dtype=bool)
        mask[:256, :] = True
        self.assertAlmostEqual(mask_angular_height(mask), np.pi / 2, delta=0.01)

    def test_empty_mask(self):
        self.assertEqual(
            mask_angular_height(np.zeros((512, 1024), dtype=bool)), 0.0)

    def test_single_row(self):
        mask = np.zeros((512, 1024), dtype=bool)
        mask[256, :] = True
        self.assertLess(mask_angular_height(mask), 0.01)


class TestRangeFromAngularSize(unittest.TestCase):
    def test_known_case(self):
        alpha = 2.0 * np.arctan(0.5)
        self.assertAlmostEqual(range_from_angular_size(alpha, 1.0), 1.0, delta=0.01)

    def test_double_height_double_range(self):
        alpha = 0.5
        r1 = range_from_angular_size(alpha, 1.0)
        r2 = range_from_angular_size(alpha, 2.0)
        self.assertAlmostEqual(r2, 2.0 * r1, delta=1e-6)

    def test_small_angle(self):
        alpha = 0.01
        r = range_from_angular_size(alpha, 1.0)
        self.assertAlmostEqual(r, 100.0, delta=1.0)

    def test_eps_clamping(self):
        r = range_from_angular_size(0.0, 1.0)
        self.assertTrue(np.isfinite(r) and r > 100.0)


# ═══════════════════════════════════════════════════════════════════════════
# wall_fraction_from_elevation
# ═══════════════════════════════════════════════════════════════════════════

class TestWallFractionFromElevation(unittest.TestCase):
    def test_nadir_returns_near(self):
        frac = wall_fraction_from_elevation(np.array([0.0, -0.99, 0.1]))
        self.assertAlmostEqual(frac, 0.28, delta=0.05)

    def test_horizon_returns_far(self):
        frac = wall_fraction_from_elevation(np.array([1.0, 0.0, 0.0]))
        self.assertAlmostEqual(frac, 0.82, delta=0.05)

    def test_monotonic(self):
        f1 = wall_fraction_from_elevation(np.array([0.7, -0.7, 0.1]))
        f2 = wall_fraction_from_elevation(np.array([0.7, -0.3, 0.1]))
        self.assertLess(f1, f2)


# ═══════════════════════════════════════════════════════════════════════════
# azimuth_unit_xz
# ═══════════════════════════════════════════════════════════════════════════

class TestAzimuthUnitXZ(unittest.TestCase):
    def test_east_direction(self):
        az = azimuth_unit_xz(np.array([1.0, 0.2, 0.0]), np.array([0.75, 0.5]))
        np.testing.assert_allclose(az, [1.0, 0.0, 0.0], atol=1e-10)

    def test_north_direction(self):
        az = azimuth_unit_xz(np.array([0.0, 0.2, 1.0]), np.array([0.5, 0.5]))
        np.testing.assert_allclose(az, [0.0, 0.0, 1.0], atol=1e-10)

    def test_unit_length(self):
        az = azimuth_unit_xz(np.array([3.0, 0.5, 4.0]), np.array([0.5, 0.5]))
        np.testing.assert_allclose(np.linalg.norm(az), 1.0, atol=1e-10)

    def test_nadir_fallback(self):
        az = azimuth_unit_xz(np.array([0.0, -1.0, 0.0]), np.array([0.5, 0.9]))
        self.assertIsNotNone(az)
        np.testing.assert_allclose(np.linalg.norm(az), 1.0, atol=1e-10)
        self.assertEqual(az[1], 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# place_on_floor_at_range
# ═══════════════════════════════════════════════════════════════════════════

class TestPlaceOnFloorAtRange(unittest.TestCase):
    def setUp(self):
        self.data = _make_room_data()

    def test_forward_placement(self):
        p = place_on_floor_at_range(
            np.zeros(3), np.array([0.0, 0.0, 1.0]), 2.0, self.data)
        np.testing.assert_allclose(p, [0.0, 1.6, 2.0])

    def test_diagonal_placement(self):
        p = place_on_floor_at_range(
            np.zeros(3), np.array([1.0, 0.0, 0.0]), 3.0, self.data)
        np.testing.assert_allclose(p, [3.0, 1.6, 0.0])

    def test_zero_range(self):
        p = place_on_floor_at_range(
            np.zeros(3), np.array([1.0, 0.0, 0.0]), 0.0, self.data)
        np.testing.assert_allclose(p, [0.0, 1.6, 0.0])

    def test_azimuth_vertical_component_stripped(self):
        p = place_on_floor_at_range(
            np.zeros(3), np.array([1.0, 5.0, 0.0]), 1.0, self.data)
        np.testing.assert_allclose(p, [1.0, 1.6, 0.0], atol=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# patch_index_from_uv
# ═══════════════════════════════════════════════════════════════════════════

class TestPatchIndexFromUV(unittest.TestCase):
    def test_start_of_range(self):
        idx = patch_index_from_uv(np.array([0.001953125, 0.5]))
        self.assertGreaterEqual(idx, 0)
        self.assertLess(idx, 256)

    def test_middle_of_range(self):
        self.assertEqual(patch_index_from_uv(np.array([0.5, 0.5])), 128)

    def test_end_wraps(self):
        self.assertEqual(patch_index_from_uv(np.array([1.0, 0.5])), 0)

    def test_custom_patch_num(self):
        self.assertEqual(
            patch_index_from_uv(np.array([0.5, 0.5]), patch_num=4), 2)


# ═══════════════════════════════════════════════════════════════════════════
# wall_distance_from_depth / closer_than_wall
# ═══════════════════════════════════════════════════════════════════════════

class TestWallDistanceFromDepth(unittest.TestCase):
    def test_constant_depth(self):
        d = wall_distance_from_depth(_dummy_depth(), np.array([0.5, 0.5]))
        self.assertAlmostEqual(d, 2.0)

    def test_scales_with_camera_height(self):
        d = wall_distance_from_depth(
            _dummy_depth(), np.array([0.5, 0.5]), camera_height=1.0)
        self.assertAlmostEqual(d, 2.0 / 1.6)


class TestCloserThanWall(unittest.TestCase):
    def setUp(self):
        self.room = _make_room_data()
        self.depth = _dummy_depth()

    def test_point_inside(self):
        p = np.array([0.0, 1.6, 1.0])
        self.assertTrue(
            closer_than_wall(p, np.array([0.25, 0.5]), self.depth, self.room))

    def test_point_outside(self):
        p = np.array([0.0, 1.6, 5.0])
        self.assertFalse(
            closer_than_wall(p, np.array([0.25, 0.5]), self.depth, self.room))


# ═══════════════════════════════════════════════════════════════════════════
# point_in_floor_polygon / floor_polygon_from_layout / check_floor_hit
# ═══════════════════════════════════════════════════════════════════════════

class TestPointInFloorPolygon(unittest.TestCase):
    @staticmethod
    def _square():
        return np.array([[0, 0], [4, 0], [4, 3], [0, 3]], dtype=np.float64)

    def test_inside(self):
        self.assertTrue(point_in_floor_polygon(np.array([2.0, 1.5]), self._square()))

    def test_outside(self):
        self.assertFalse(
            point_in_floor_polygon(np.array([10.0, 10.0]), self._square()))

    def test_on_vertex(self):
        self.assertTrue(point_in_floor_polygon(np.array([0.0, 0.0]), self._square()))

    def test_on_edge(self):
        self.assertTrue(point_in_floor_polygon(np.array([2.0, 0.0]), self._square()))

    def test_xyz_input_drops_y(self):
        self.assertTrue(
            point_in_floor_polygon(np.array([2.0, 999.0, 1.5]), self._square()))


class TestFloorPolygonFromLayout(unittest.TestCase):
    def test_returns_n_by_2(self):
        poly = floor_polygon_from_layout(_make_room_data())
        self.assertEqual(poly.shape, (4, 2))
        self.assertEqual(poly.dtype, np.float64)


class TestCheckFloorHit(unittest.TestCase):
    def setUp(self):
        self.room = _make_room_data()

    def test_inside_hit(self):
        self.assertTrue(check_floor_hit(np.array([0.0, 1.6, 0.0]), self.room))

    def test_outside_not_hit(self):
        self.assertFalse(
            check_floor_hit(np.array([100.0, 1.6, 100.0]), self.room))


# ═══════════════════════════════════════════════════════════════════════════
# contact_uvs_from_mask
# ═══════════════════════════════════════════════════════════════════════════

class TestContactUVsFromMask(unittest.TestCase):
    def test_centroid_first(self):
        uvs = contact_uvs_from_mask(_small_mask(), num_samples=5)
        self.assertEqual(len(uvs), 5)
        self.assertGreater(uvs[0][0], 0.4)
        self.assertLess(uvs[0][0], 0.6)

    def test_all_in_0_1_range(self):
        uvs = contact_uvs_from_mask(_small_mask(), num_samples=3)
        for uv in uvs:
            self.assertTrue(0.0 <= uv[0] <= 1.0)
            self.assertTrue(0.0 <= uv[1] <= 1.0)

    def test_single_sample(self):
        uvs = contact_uvs_from_mask(_small_mask(), num_samples=1)
        self.assertEqual(len(uvs), 1)

    def test_empty_mask(self):
        self.assertEqual(contact_uvs_from_mask(_empty_mask()), [])

    def test_samples_sorted_by_centroid_closeness(self):
        uvs = contact_uvs_from_mask(_small_mask(), num_samples=50)
        # First sample should have the smallest (or tied-smallest)
        # distance from its own u, since it IS the centroid sample
        centroid_u = uvs[0][0]
        # All UVs are within the mask's span
        for uv in uvs:
            self.assertGreaterEqual(uv[0], 0.3,
                                    msg="U should be within mask range")
            self.assertLessEqual(uv[0], 0.7,
                                msg="U should be within mask range")


# ═══════════════════════════════════════════════════════════════════════════
# nearest_wall
# ═══════════════════════════════════════════════════════════════════════════

class TestNearestWall(unittest.TestCase):
    def setUp(self):
        self.room = _make_room_data()

    def test_point_near_back_wall(self):
        p = np.array([0.0, 1.6, 1.45])
        wall, dist, inward = nearest_wall(p, self.room)
        self.assertIsNotNone(wall)
        np.testing.assert_allclose(inward, [0.0, 0.0, -1.0], atol=1e-10)

    def test_point_near_left_wall(self):
        p = np.array([-1.95, 1.6, 0.0])
        wall, dist, inward = nearest_wall(p, self.room)
        self.assertIsNotNone(wall)
        np.testing.assert_allclose(inward, [1.0, 0.0, 0.0], atol=1e-10)

    def test_point_at_center(self):
        p = np.array([0.0, 1.6, 0.0])
        wall, dist, inward = nearest_wall(p, self.room)
        self.assertIsNotNone(wall)
        self.assertIsNotNone(inward)
        self.assertNotEqual(dist, float("inf"))

    def test_inward_point_toward_origin(self):
        for ox, oz in [(0, 1.4), (-1.9, 0), (1.9, 0), (0, -1.4)]:
            p = np.array([ox, 1.6, oz], dtype=np.float64)
            _, _, inward = nearest_wall(p, self.room)
            self.assertIsNotNone(inward)
            to_origin = np.array([-p[0], 0.0, -p[2]])
            if np.linalg.norm(to_origin) > 1e-6:
                self.assertGreater(
                    np.dot(inward, to_origin), -1e-10,
                    f"Failed for offset ({ox}, {oz})")


# ═══════════════════════════════════════════════════════════════════════════
# shrink_range_into_footprint
# ═══════════════════════════════════════════════════════════════════════════

class TestShrinkRangeIntoFootprint(unittest.TestCase):
    def setUp(self):
        self.room = _make_room_data()

    def test_inside_unchanged(self):
        p = shrink_range_into_footprint(
            np.zeros(3), np.array([0.0, 0.0, 1.0]), 1.0, self.room)
        self.assertIsNotNone(p)
        np.testing.assert_allclose(p, [0.0, 1.6, 1.0])

    def test_shrinks_into_room(self):
        """A point outside the room should be pulled back inside."""
        p = shrink_range_into_footprint(
            np.zeros(3), np.array([0.0, 0.0, 1.0]), 3.0, self.room, steps=20)
        self.assertIsNotNone(p)
        self.assertTrue(check_floor_hit(p, self.room))

    def test_returns_none_when_outside_from_far_origin(self):
        p = shrink_range_into_footprint(
            np.array([10.0, 0.0, 10.0]),
            np.array([0.0, 0.0, 1.0]),
            0.3, self.room, min_range=0.2, steps=3)
        self.assertIsNone(p)


# ═══════════════════════════════════════════════════════════════════════════
# uv2equirectangular
# ═══════════════════════════════════════════════════════════════════════════

class TestUv2Equirectangular(unittest.TestCase):
    def test_returns_unit_vector(self):
        for u, v in [(0.5, 0.5), (0.0, 0.5), (0.75, 0.25), (0.5, 0.0)]:
            d = uv2equirectangular(np.array([u, v]))
            np.testing.assert_allclose(np.linalg.norm(d), 1.0, atol=1e-10)

    def test_center_is_forward(self):
        d = uv2equirectangular(np.array([0.5, 0.5]))
        np.testing.assert_allclose(d[:2], [0.0, 0.0], atol=1e-6)
        self.assertLess(d[2], 0)  # forward in JSON is -Z


# ═══════════════════════════════════════════════════════════════════════════
# _wall_y_bounds
# ═══════════════════════════════════════════════════════════════════════════

class TestWallYBounds(unittest.TestCase):
    def test_with_explicit_ceiling(self):
        data = {"cameraHeight": 1.6, "cameraCeilingHeight": 1.0, "layoutHeight": 2.6}
        y_lo, y_hi = _wall_y_bounds(data)
        self.assertAlmostEqual(y_lo, -1.0)
        self.assertAlmostEqual(y_hi, 1.6)

    def test_fallback_to_layout_height(self):
        data = {"cameraHeight": 1.6, "layoutHeight": 3.0}
        y_lo, y_hi = _wall_y_bounds(data)
        self.assertAlmostEqual(y_lo, -1.4)  # -(3.0 - 1.6)
        self.assertAlmostEqual(y_hi, 1.6)

    def test_symmetric_room(self):
        data = {"cameraHeight": 1.2, "cameraCeilingHeight": 1.2, "layoutHeight": 2.4}
        y_lo, y_hi = _wall_y_bounds(data)
        self.assertAlmostEqual(y_lo, -1.2)
        self.assertAlmostEqual(y_hi, 1.2)


# ═══════════════════════════════════════════════════════════════════════════
# _point_on_segment_xz
# ═══════════════════════════════════════════════════════════════════════════

class TestPointOnSegmentXZ(unittest.TestCase):
    def setUp(self):
        self.a = np.array([0.0, 99.0, 0.0], dtype=np.float64)
        self.b = np.array([4.0, 99.0, 0.0], dtype=np.float64)

    def test_point_on_segment_midpoint(self):
        self.assertTrue(
            _point_on_segment_xz(np.array([2.0, 0.0, 0.0]), self.a, self.b))

    def test_point_on_segment_endpoint(self):
        self.assertTrue(
            _point_on_segment_xz(np.array([0.0, 0.0, 0.0]), self.a, self.b))
        self.assertTrue(
            _point_on_segment_xz(np.array([4.0, 0.0, 0.0]), self.a, self.b))

    def test_point_before_segment_start(self):
        self.assertFalse(
            _point_on_segment_xz(np.array([-0.5, 0.0, 0.0]), self.a, self.b))

    def test_point_after_segment_end(self):
        self.assertFalse(
            _point_on_segment_xz(np.array([5.0, 0.0, 0.0]), self.a, self.b))

    def test_point_off_axis(self):
        self.assertFalse(
            _point_on_segment_xz(np.array([2.0, 0.0, 1.0]), self.a, self.b))

    def test_within_tolerance(self):
        # Just barely on the segment within default tol
        self.assertTrue(
            _point_on_segment_xz(
                np.array([2.0, 0.0, 0.04]), self.a, self.b, tol=0.05))

    def test_outside_tolerance(self):
        self.assertFalse(
            _point_on_segment_xz(
                np.array([2.0, 0.0, 0.1]), self.a, self.b, tol=0.05))

    def test_zero_length_segment(self):
        a = np.array([1.0, 0.0, 1.0])
        # Point exactly at zero-length segment position — always True
        self.assertTrue(_point_on_segment_xz(
            np.array([1.0, 0.0, 1.0]), a, a, tol=0.01))
        self.assertTrue(_point_on_segment_xz(
            np.array([1.0, 0.0, 1.0]), a, a, tol=0.0))
        # Point away from zero-length segment, tight tolerance — False
        self.assertFalse(_point_on_segment_xz(
            np.array([1.1, 0.0, 1.1]), a, a, tol=0.01))

    def test_diagonal_segment(self):
        a = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        b = np.array([3.0, 0.0, 4.0], dtype=np.float64)
        self.assertTrue(
            _point_on_segment_xz(np.array([1.5, 0.0, 2.0]), a, b))

    def test_y_coordinate_ignored(self):
        # Y values differ greatly; XZ projection is what matters
        self.assertTrue(
            _point_on_segment_xz(
                np.array([2.0, 50.0, 0.0]),
                np.array([0.0, -10.0, 0.0]),
                np.array([4.0, 30.0, 0.0])))


# ═══════════════════════════════════════════════════════════════════════════
# ray_intersect_floor
# ═══════════════════════════════════════════════════════════════════════════

class TestRayIntersectFloor(unittest.TestCase):
    def setUp(self):
        self.room = _make_room_data()  # cameraHeight=1.6

    def test_ray_from_origin_to_floor(self):
        origin = np.array([0.0, 0.0, 0.0])
        direction = np.array([0.0, 0.8, -0.6])  # downward-forward
        direction = direction / np.linalg.norm(direction)
        p = ray_intersect_floor(origin, direction, self.room)
        self.assertAlmostEqual(p[1], 1.6)  # floor height

    def test_ray_from_above_floor(self):
        origin = np.array([0.0, 3.0, 0.0])
        direction = np.array([0.0, -1.0, 0.0])
        p = ray_intersect_floor(origin, direction, self.room)
        self.assertAlmostEqual(p[1], 1.6)
        self.assertAlmostEqual(p[0], 0.0)
        self.assertAlmostEqual(p[2], 0.0)

    def test_point_is_on_ray_line(self):
        origin = np.array([0.5, 0.3, 0.0])
        direction = np.array([0.1, 0.7, 0.3])
        direction = direction / np.linalg.norm(direction)
        p = ray_intersect_floor(origin, direction, self.room)
        # Reconstruct t and verify
        t_est = (1.6 - origin[1]) / direction[1]
        np.testing.assert_allclose(p, origin + t_est * direction, atol=1e-10)

    def test_ray_parallel_to_floor_raises(self):
        origin = np.array([0.0, 2.0, 0.0])
        direction = np.array([1.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            ray_intersect_floor(origin, direction, self.room)

    def test_ray_away_from_floor_raises(self):
        origin = np.array([0.0, 2.0, 0.0])
        direction = np.array([0.0, 1.0, 0.0])
        with self.assertRaises(ValueError):
            ray_intersect_floor(origin, direction, self.room)


# ═══════════════════════════════════════════════════════════════════════════
# ray_intersect_wall
# ═══════════════════════════════════════════════════════════════════════════

class TestRayIntersectWall(unittest.TestCase):
    def setUp(self):
        self.room = _make_room_data()  # 4m x 3m room, camera at origin

    def test_hits_front_wall(self):
        origin = np.zeros(3)
        direction = np.array([0.0, 0.0, -1.0])  # toward -Z
        hit = ray_intersect_wall(origin, direction, self.room)
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(hit[2], -1.5, delta=0.05)
        self.assertAlmostEqual(hit[0], 0.0, delta=0.05)

    def test_hits_back_wall(self):
        origin = np.zeros(3)
        direction = np.array([0.0, 0.0, 1.0])  # toward +Z
        hit = ray_intersect_wall(origin, direction, self.room)
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(hit[2], 1.5, delta=0.05)

    def test_hits_left_wall(self):
        origin = np.zeros(3)
        direction = np.array([-1.0, 0.0, 0.0])  # toward -X
        hit = ray_intersect_wall(origin, direction, self.room)
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(hit[0], -2.0, delta=0.05)

    def test_hits_right_wall(self):
        origin = np.zeros(3)
        direction = np.array([1.0, 0.0, 0.0])  # toward +X
        hit = ray_intersect_wall(origin, direction, self.room)
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(hit[0], 2.0, delta=0.05)

    def test_diagonal_hit(self):
        origin = np.zeros(3)
        direction = np.array([1.0, 0.0, -1.0])
        direction = direction / np.linalg.norm(direction)
        hit = ray_intersect_wall(origin, direction, self.room)
        self.assertIsNotNone(hit)

    def test_ray_from_offset_origin(self):
        origin = np.array([1.0, 0.0, 0.0])
        direction = np.array([1.0, 0.0, 0.0])  # toward right wall
        hit = ray_intersect_wall(origin, direction, self.room)
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(hit[0], 2.0, delta=0.05)

    def test_returns_none_when_no_hit(self):
        # Point the ray where there is no wall (room has only 4 walls)
        origin = np.array([10.0, 0.0, 10.0])
        direction = np.array([1.0, 0.0, 1.0])
        direction = direction / np.linalg.norm(direction)
        hit = ray_intersect_wall(origin, direction, self.room)
        self.assertIsNone(hit)

    def test_return_wall_true(self):
        origin = np.zeros(3)
        direction = np.array([0.0, 0.0, -1.0])
        hit, wall = ray_intersect_wall(origin, direction, self.room, return_wall=True)
        self.assertIsNotNone(hit)
        self.assertIsNotNone(wall)
        self.assertIn("planeEquation", wall)
        self.assertIn("pointsIdx", wall)

    def test_return_wall_true_none(self):
        origin = np.array([10.0, 0.0, 10.0])
        direction = np.array([1.0, 0.0, 1.0])
        direction = direction / np.linalg.norm(direction)
        hit, wall = ray_intersect_wall(origin, direction, self.room, return_wall=True)
        self.assertIsNone(hit)
        self.assertIsNone(wall)

    def test_hits_nearest_wall(self):
        """When ray hits multiple walls, return the nearest one."""
        origin = np.array([0.0, 0.0, 0.0])
        # Diagonal toward a corner — should hit one wall first
        direction = np.array([0.5, 0.0, -0.5])
        direction = direction / np.linalg.norm(direction)
        hit = ray_intersect_wall(origin, direction, self.room)
        self.assertIsNotNone(hit)
        # Should be closer than the far corner
        dist = np.linalg.norm(hit)
        self.assertLess(dist, 5.0)

    def test_ray_above_ceiling_no_hit(self):
        origin = np.array([0.0, 3.0, 0.0])
        direction = np.array([1.0, 0.0, 0.0])
        hit = ray_intersect_wall(origin, direction, self.room)
        self.assertIsNone(hit)

    def test_ray_below_floor_no_hit(self):
        origin = np.array([0.0, -3.0, 0.0])
        direction = np.array([1.0, 0.0, 0.0])
        hit = ray_intersect_wall(origin, direction, self.room)
        self.assertIsNone(hit)

    def test_ray_hits_within_height_bounds(self):
        """Ray at camera height (1.6m) hits wall segment that spans floor-to-ceiling."""
        origin = np.zeros(3)
        direction = np.array([0.0, 0.1, -1.0])
        direction = direction / np.linalg.norm(direction)
        hit = ray_intersect_wall(origin, direction, self.room)
        # Slight upward angle at origin — still within wall height at hit distance
        self.assertIsNotNone(hit)


if __name__ == "__main__":
    unittest.main()
