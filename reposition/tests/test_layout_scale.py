"""Unit tests for layout JSON scale helpers."""

import unittest

import numpy as np

from utils.writer import xyz2json


class TestXyz2JsonCameraHeight(unittest.TestCase):
    def test_layout_height_matches_camera_times_ratio(self):
        xyz = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]], dtype=np.float32)
        data = xyz2json(xyz, ratio=1.0, camera_height=1.3)
        self.assertAlmostEqual(data["cameraHeight"], 1.3)
        self.assertAlmostEqual(data["cameraCeilingHeight"], 1.3)
        self.assertAlmostEqual(data["layoutHeight"], 2.6)

    def test_points_scale_with_camera_height(self):
        xyz = np.array([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0], [-2.0, 1.0, 0.0]], dtype=np.float32)
        a = xyz2json(xyz, ratio=0.625, camera_height=1.6)
        b = xyz2json(xyz, ratio=0.625, camera_height=0.8)
        pa = np.array(a["layoutPoints"]["points"][0]["xyz"])
        pb = np.array(b["layoutPoints"]["points"][0]["xyz"])
        np.testing.assert_allclose(pb, pa * 0.5, atol=1e-6)
        self.assertAlmostEqual(b["layoutHeight"], a["layoutHeight"] * 0.5)

    def test_room_height_replaces_metric_scale(self):
        xyz = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]], dtype=np.float32)
        data = xyz2json(xyz, ratio=0.625, room_height=2.6)
        self.assertAlmostEqual(data["cameraHeight"], 1.6)
        self.assertAlmostEqual(data["cameraCeilingHeight"], 1.0)
        self.assertAlmostEqual(data["layoutHeight"], 2.6)


class TestLayoutHull(unittest.TestCase):
    def test_hull_height_matches_layout(self):
        from visualization.compare_layout_furniture import layout_json_to_hull

        xyz = np.array(
            [[2.0, 1.0, 1.5], [2.0, 1.0, -1.5], [-2.0, 1.0, -1.5], [-2.0, 1.0, 1.5]],
            dtype=np.float32,
        )
        data = xyz2json(xyz, ratio=0.625, camera_height=1.6)
        mesh, edges = layout_json_to_hull(data)
        extent = np.asarray(mesh.get_axis_aligned_bounding_box().get_extent())
        self.assertGreater(len(mesh.triangles), 4)
        self.assertAlmostEqual(float(extent[1]), float(data["layoutHeight"]), places=3)
        self.assertGreater(len(edges.lines), 0)

        scaled, _ = layout_json_to_hull(data, room_height=2.5)
        self.assertAlmostEqual(
            float(scaled.get_axis_aligned_bounding_box().get_extent()[1]), 2.5, places=3
        )


if __name__ == "__main__":
    unittest.main()
