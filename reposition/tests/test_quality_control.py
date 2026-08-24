"""Unit tests for Plan A / Plan B placement quality control."""

from __future__ import annotations

import unittest

import numpy as np

from reposition.quality_control import (
    WALL_MOUNTED_LABELS,
    aabb_overlap_ratio,
    aabb_overlap_volume,
    apply_inter_object_clipping_quality,
    apply_layout_geometry_quality,
    apply_placement_quality,
    apply_surface_class_quality,
    check_height_band,
    check_inside_footprint,
    check_surface_class_consistency,
    check_wall_attachment,
    check_wall_penetration,
    expected_surface_for_label,
    filter_kept_placements,
    label_map_from_dino_boxes,
    mark_quality_rejections,
    normalize_dino_label,
    quality_check_message,
    typical_height_for_label,
)


def _boxes(*items: tuple[str, float]) -> dict:
    detections = []
    for i, (label, score) in enumerate(items):
        detections.append(
            {
                "detection_id": f"det-{i}",
                "label": label,
                "score": score,
                "box": {
                    "xMin": i * 10,
                    "yMin": 0,
                    "xMax": i * 10 + 8,
                    "yMax": 8,
                },
            }
        )
    return {
        "num_detections": len(detections),
        "detections": detections,
        "box_prompts": [d["box"] for d in detections],
    }


def _square_room(half: float = 2.0, camera_height: float = 1.6) -> dict:
    """Axis-aligned square room centered on the camera (JSON frame)."""
    pts = [
        {"xyz": [-half, camera_height, half]},
        {"xyz": [half, camera_height, half]},
        {"xyz": [half, camera_height, -half]},
        {"xyz": [-half, camera_height, -half]},
    ]
    walls = []
    for i in range(4):
        j = (i + 1) % 4
        a = np.array(pts[i]["xyz"], dtype=np.float64)
        b = np.array(pts[j]["xyz"], dtype=np.float64)
        wall_dir = b - a
        normal_2d = np.array([-wall_dir[2], wall_dir[0]], dtype=np.float64)
        normal_2d /= np.linalg.norm(normal_2d)
        midpoint_xz = (a[[0, 2]] + b[[0, 2]]) / 2.0
        if np.dot(normal_2d, -midpoint_xz) < 0:
            normal_2d = -normal_2d
        normal = np.array([normal_2d[0], 0.0, normal_2d[1]], dtype=np.float64)
        d = -float(np.dot(normal, a))
        walls.append(
            {
                "pointsIdx": [i, j],
                "planeEquation": [
                    float(normal[0]),
                    float(normal[1]),
                    float(normal[2]),
                    d,
                ],
                "normal": normal.tolist(),
            }
        )
    return {
        "cameraHeight": camera_height,
        "cameraCeilingHeight": 1.0,
        "layoutHeight": camera_height + 1.0,
        "layoutPoints": {"points": pts},
        "layoutWalls": {"walls": walls},
    }


class TestNormalizeDinoLabel(unittest.TestCase):
    def test_strips_score_and_case(self):
        self.assertEqual(normalize_dino_label("Painting(0.82)"), "painting")

    def test_joins_words(self):
        self.assertEqual(normalize_dino_label("book shelf"), "bookshelf")

    def test_empty(self):
        self.assertIsNone(normalize_dino_label(""))
        self.assertIsNone(normalize_dino_label(None))


class TestTypicalHeightForLabel(unittest.TestCase):
    def test_known_classes(self):
        self.assertAlmostEqual(typical_height_for_label("chair"), 0.866, places=2)
        self.assertAlmostEqual(typical_height_for_label("table"), 0.747, places=2)
        self.assertAlmostEqual(typical_height_for_label("bed"), 0.55, places=2)
        self.assertAlmostEqual(typical_height_for_label("sofa"), 0.762, places=2)

    def test_concatenated_dino_tokens(self):
        self.assertAlmostEqual(typical_height_for_label("sofachairbed"), 0.866, places=2)
        self.assertAlmostEqual(typical_height_for_label("cabinetshelf"), 0.90)

    def test_unknown_uses_default(self):
        self.assertAlmostEqual(typical_height_for_label("lamp"), 0.80)
        self.assertAlmostEqual(typical_height_for_label(None), 0.80)


class TestExpectedSurface(unittest.TestCase):
    def test_wall_allowlist(self):
        for label in ("door", "window", "painting", "shelf", "bookshelf"):
            self.assertEqual(expected_surface_for_label(label), "wall")
            self.assertIn(normalize_dino_label(label), WALL_MOUNTED_LABELS)

    def test_floor_allowlist(self):
        self.assertEqual(expected_surface_for_label("rug"), "floor")

    def test_unknown_is_none(self):
        self.assertIsNone(expected_surface_for_label("chair"))
        self.assertIsNone(expected_surface_for_label("sofa"))


class TestSurfaceClassCheck(unittest.TestCase):
    def test_door_on_floor_fails(self):
        result = check_surface_class_consistency(
            {"mask": "mask_0.png", "surface": "floor", "label": "door"}
        )
        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["keep"])

    def test_door_on_wall_ok(self):
        result = check_surface_class_consistency(
            {"mask": "mask_0.png", "surface": "wall", "label": "Door(0.9)"}
        )
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["keep"])

    def test_chair_skipped(self):
        result = check_surface_class_consistency(
            {"mask": "mask_1.png", "surface": "floor", "label": "chair"}
        )
        self.assertEqual(result["status"], "skip")
        self.assertTrue(result["keep"])

    def test_rug_on_wall_fails(self):
        result = check_surface_class_consistency(
            {"mask": "mask_2.png", "surface": "wall", "label": "rug"}
        )
        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["keep"])


class TestApplySurfaceClassQuality(unittest.TestCase):
    def test_maps_labels_and_rejects(self):
        boxes = _boxes(("door", 0.9), ("chair", 0.8), ("painting", 0.7))
        placements = [
            {"mask": "mask_0.png", "surface": "floor", "translation": [1, 1.6, 0]},
            {"mask": "mask_1.png", "surface": "floor", "translation": [0, 1.6, 1]},
            {"mask": "mask_2.png", "surface": "wall", "translation": [2, 0.5, -2]},
        ]
        out = apply_surface_class_quality(placements, boxes=boxes)
        self.assertEqual(out[0]["label"], "door")
        self.assertFalse(out[0]["quality"]["keep"])
        self.assertIn("surface_class", out[0]["quality"]["fails"])
        self.assertEqual(out[0]["quality"]["message"], "Item on wrong surface")

        self.assertEqual(out[1]["label"], "chair")
        self.assertTrue(out[1]["quality"]["keep"])
        self.assertEqual(out[1]["quality"]["message"], "")

        self.assertEqual(out[2]["label"], "painting")
        self.assertTrue(out[2]["quality"]["keep"])

        kept = filter_kept_placements(out)
        self.assertEqual(len(kept), 2)
        self.assertEqual([p["mask"] for p in kept], ["mask_1.png", "mask_2.png"])

    def test_drop_rejected(self):
        boxes = _boxes(("window", 0.95))
        placements = [
            {"mask": "mask_0.png", "surface": "floor"},
        ]
        out = apply_surface_class_quality(
            placements, boxes=boxes, drop_rejected=True
        )
        self.assertEqual(out, [])

    def test_min_score_alignment(self):
        boxes = _boxes(("door", 0.2), ("shelf", 0.9))
        label_map = label_map_from_dino_boxes(boxes, min_score=0.5)
        self.assertEqual(set(label_map.keys()), {0})
        self.assertEqual(label_map[0]["label"], "shelf")
        self.assertEqual(label_map[0]["detection_id"], "det-1")

        placements = [{"mask": "mask_0.png", "surface": "floor"}]
        out = apply_surface_class_quality(
            placements, boxes=boxes, min_score=0.5
        )
        self.assertEqual(out[0]["label"], "shelf")
        self.assertFalse(out[0]["quality"]["keep"])


class TestPlanBGeometry(unittest.TestCase):
    def setUp(self):
        self.room = _square_room(half=2.0, camera_height=1.6)

    def test_inside_footprint_ok(self):
        placement = {
            "surface": "floor",
            "translation": [0.0, 1.6, 0.0],
            "scale": {"factor": 1.0, "object_height": 0.8, "target_height": 0.8},
        }
        result = check_inside_footprint(placement, self.room)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["keep"])

    def test_inside_footprint_rejects_outside(self):
        placement = {
            "surface": "floor",
            "translation": [5.0, 1.6, 0.0],
            "scale": {"factor": 1.0, "target_height": 0.8},
        }
        result = check_inside_footprint(placement, self.room)
        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["keep"])

    def test_inside_footprint_skips_wall(self):
        placement = {"surface": "wall", "translation": [0.0, 0.5, -1.9]}
        result = check_inside_footprint(placement, self.room)
        self.assertEqual(result["status"], "skip")

    def test_wall_penetration_ok_center(self):
        placement = {
            "surface": "floor",
            "translation": [0.0, 1.6, 0.0],
            "scale": {"factor": 1.0, "target_height": 0.8},
        }
        result = check_wall_penetration(placement, self.room)
        self.assertEqual(result["status"], "ok")
        self.assertLessEqual(result["max_penetration_m"], 0.08)

    def test_wall_penetration_light_nick_passes_default(self):
        # ~10 cm AABB overshoot near wall — proxy noise, not a clear break.
        placement = {
            "surface": "floor",
            "translation": [1.92, 1.6, 0.0],
            "scale": {"factor": 1.0, "target_height": 0.5},
        }
        result = check_wall_penetration(placement, self.room)
        self.assertLess(result["max_penetration_m"], 0.15)
        self.assertGreater(result["max_penetration_m"], 0.05)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["keep"])

    def test_wall_penetration_clear_break_still_fails(self):
        placement = {
            "surface": "floor",
            "translation": [1.85, 1.6, 0.0],
            "scale": {"factor": 1.0, "target_height": 1.5},
        }
        result = check_wall_penetration(placement, self.room)
        self.assertGreater(result["max_penetration_m"], 0.15)
        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["keep"])

    def test_wall_penetration_fails_near_outside(self):
        placement = {
            "surface": "floor",
            "translation": [1.85, 1.6, 0.0],
            "scale": {"factor": 1.0, "target_height": 1.5},
        }
        result = check_wall_penetration(
            placement, self.room, max_penetration_m=0.05
        )
        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["keep"])

    def test_wall_attachment_ok(self):
        placement = {"surface": "wall", "translation": [0.0, 0.5, -1.95]}
        result = check_wall_attachment(placement, self.room)
        self.assertEqual(result["status"], "ok")
        self.assertLessEqual(result["distance_m"], 0.30)

    def test_wall_attachment_fails_mid_room(self):
        placement = {"surface": "wall", "translation": [0.0, 0.5, 0.0]}
        result = check_wall_attachment(
            placement, self.room, max_distance_m=0.30
        )
        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["keep"])

    def test_height_band_floor_ok(self):
        placement = {
            "surface": "floor",
            "translation": [0.0, 1.6, 0.0],
            "scale": {"factor": 1.0, "target_height": 0.9},
        }
        result = check_height_band(placement, self.room)
        self.assertEqual(result["status"], "ok")

    def test_height_band_floor_floating_fails(self):
        placement = {
            "surface": "floor",
            "translation": [0.0, 0.1, 0.0],
            "scale": {"factor": 1.0, "target_height": 0.9},
        }
        result = check_height_band(placement, self.room)
        self.assertEqual(result["status"], "fail")

    def test_height_band_wall_out_of_band_fails(self):
        placement = {
            "surface": "wall",
            "translation": [0.0, 2.5, -1.9],
            "scale": {"factor": 1.0, "target_height": 0.5},
        }
        result = check_height_band(placement, self.room)
        self.assertEqual(result["status"], "fail")

    def test_apply_layout_geometry_quality(self):
        placements = [
            {
                "mask": "mask_0.png",
                "surface": "floor",
                "translation": [0.0, 1.6, 0.0],
                "scale": {"factor": 1.0, "target_height": 0.8},
            },
            {
                "mask": "mask_1.png",
                "surface": "floor",
                "translation": [6.0, 1.6, 0.0],
                "scale": {"factor": 1.0, "target_height": 0.8},
            },
            {
                "mask": "mask_2.png",
                "surface": "wall",
                "translation": [0.0, 0.4, 0.0],
                "scale": {"factor": 1.0, "target_height": 0.5},
            },
        ]
        out = apply_layout_geometry_quality(placements, self.room)
        self.assertTrue(out[0]["quality"]["keep"])
        self.assertFalse(out[1]["quality"]["keep"])
        self.assertIn("inside_footprint", out[1]["quality"]["fails"])
        self.assertEqual(out[1]["quality"]["message"], "Item is too big")
        self.assertFalse(out[2]["quality"]["keep"])
        self.assertIn("wall_attachment", out[2]["quality"]["fails"])
        self.assertEqual(out[2]["quality"]["message"], "Item is too big")

    def test_apply_placement_quality_combines_a_and_b(self):
        boxes = _boxes(("door", 0.9), ("chair", 0.8))
        placements = [
            {
                "mask": "mask_0.png",
                "surface": "floor",
                "translation": [0.0, 1.6, 0.0],
                "scale": {"factor": 1.0, "target_height": 0.8},
            },
            {
                "mask": "mask_1.png",
                "surface": "floor",
                "translation": [0.0, 1.6, 0.5],
                "scale": {"factor": 1.0, "target_height": 0.8},
            },
        ]
        out = apply_placement_quality(
            placements, boxes=boxes, layout=self.room
        )
        self.assertFalse(out[0]["quality"]["keep"])
        self.assertIn("surface_class", out[0]["quality"]["fails"])
        self.assertEqual(out[0]["quality"]["message"], "Item on wrong surface")
        self.assertTrue(out[1]["quality"]["keep"])
        self.assertEqual(out[1]["quality"]["message"], "")
        self.assertIn("inside_footprint", out[1]["quality"]["checks"])
        self.assertIn("wall_penetration", out[1]["quality"]["checks"])
        self.assertIn("height_band", out[1]["quality"]["checks"])
        self.assertIn("inter_object_clip", out[1]["quality"]["checks"])


class TestPlanCClipping(unittest.TestCase):
    def setUp(self):
        self.room = _square_room(half=2.0, camera_height=1.6)

    def test_overlap_helpers(self):
        min_a = np.array([0.0, 0.0, 0.0])
        max_a = np.array([1.0, 1.0, 1.0])
        min_b = np.array([0.5, 0.5, 0.5])
        max_b = np.array([1.5, 1.5, 1.5])
        self.assertAlmostEqual(aabb_overlap_volume(min_a, max_a, min_b, max_b), 0.125)
        self.assertAlmostEqual(aabb_overlap_ratio(min_a, max_a, min_b, max_b), 0.125)
        self.assertEqual(
            aabb_overlap_ratio(min_a, max_a, np.array([2, 2, 2]), np.array([3, 3, 3])),
            0.0,
        )

    def test_rejects_overlapping_pair(self):
        placements = [
            {
                "mask": "mask_0.png",
                "surface": "floor",
                "translation": [0.0, 1.6, 0.0],
                "scale": {"factor": 1.0, "target_height": 1.0},
                "quality": {"keep": True, "fails": [], "warns": [], "checks": {}},
            },
            {
                "mask": "mask_1.png",
                "surface": "floor",
                "translation": [0.15, 1.6, 0.0],
                "scale": {"factor": 1.0, "target_height": 1.0},
                "quality": {"keep": True, "fails": [], "warns": [], "checks": {}},
            },
        ]
        out = apply_inter_object_clipping_quality(
            placements, self.room, overlap_ratio_thresh=0.15
        )
        self.assertFalse(out[0]["quality"]["keep"])
        self.assertFalse(out[1]["quality"]["keep"])
        self.assertIn("inter_object_clip", out[0]["quality"]["fails"])
        self.assertIn("inter_object_clip", out[1]["quality"]["fails"])
        self.assertEqual(out[0]["quality"]["message"], "Item clipped into another object")
        self.assertEqual(out[0]["quality"]["checks"]["inter_object_clip"]["overlaps"][0]["mask"], "mask_1.png")

    def test_light_overlap_passes_default_thresh(self):
        # ~0.14 min-volume overlap — grazing AABB proxy, not a jammed pair.
        placements = [
            {
                "mask": "mask_0.png",
                "surface": "floor",
                "translation": [0.0, 1.6, 0.0],
                "scale": {"factor": 1.0, "target_height": 1.0},
                "quality": {"keep": True, "fails": [], "warns": [], "checks": {}},
            },
            {
                "mask": "mask_1.png",
                "surface": "floor",
                "translation": [0.60, 1.6, 0.0],
                "scale": {"factor": 1.0, "target_height": 1.0},
                "quality": {"keep": True, "fails": [], "warns": [], "checks": {}},
            },
        ]
        out = apply_inter_object_clipping_quality(placements, self.room)
        self.assertTrue(out[0]["quality"]["keep"])
        self.assertTrue(out[1]["quality"]["keep"])
        self.assertEqual(
            out[0]["quality"]["checks"]["inter_object_clip"]["status"], "ok"
        )

    def test_heavy_overlap_still_fails_default_thresh(self):
        placements = [
            {
                "mask": "mask_0.png",
                "surface": "floor",
                "translation": [0.0, 1.6, 0.0],
                "scale": {"factor": 1.0, "target_height": 1.0},
                "quality": {"keep": True, "fails": [], "warns": [], "checks": {}},
            },
            {
                "mask": "mask_1.png",
                "surface": "floor",
                "translation": [0.20, 1.6, 0.0],
                "scale": {"factor": 1.0, "target_height": 1.0},
                "quality": {"keep": True, "fails": [], "warns": [], "checks": {}},
            },
        ]
        out = apply_inter_object_clipping_quality(placements, self.room)
        self.assertFalse(out[0]["quality"]["keep"])
        self.assertFalse(out[1]["quality"]["keep"])
        self.assertIn("inter_object_clip", out[0]["quality"]["fails"])

    def test_separated_objects_ok(self):
        placements = [
            {
                "mask": "mask_0.png",
                "surface": "floor",
                "translation": [-1.0, 1.6, 0.0],
                "scale": {"factor": 1.0, "target_height": 0.6},
                "quality": {"keep": True, "fails": [], "warns": [], "checks": {}},
            },
            {
                "mask": "mask_1.png",
                "surface": "floor",
                "translation": [1.0, 1.6, 0.0],
                "scale": {"factor": 1.0, "target_height": 0.6},
                "quality": {"keep": True, "fails": [], "warns": [], "checks": {}},
            },
        ]
        out = apply_inter_object_clipping_quality(placements, self.room)
        self.assertTrue(out[0]["quality"]["keep"])
        self.assertTrue(out[1]["quality"]["keep"])
        self.assertEqual(
            out[0]["quality"]["checks"]["inter_object_clip"]["status"], "ok"
        )

    def test_already_rejected_skipped_from_pairs(self):
        placements = [
            {
                "mask": "mask_0.png",
                "surface": "floor",
                "translation": [0.0, 1.6, 0.0],
                "scale": {"factor": 1.0, "target_height": 1.0},
                "quality": {
                    "keep": False,
                    "fails": ["surface_class"],
                    "warns": [],
                    "checks": {},
                },
            },
            {
                "mask": "mask_1.png",
                "surface": "floor",
                "translation": [0.1, 1.6, 0.0],
                "scale": {"factor": 1.0, "target_height": 1.0},
                "quality": {"keep": True, "fails": [], "warns": [], "checks": {}},
            },
        ]
        out = apply_inter_object_clipping_quality(
            placements, self.room, only_kept=True
        )
        self.assertEqual(
            out[0]["quality"]["checks"]["inter_object_clip"]["status"], "skip"
        )
        self.assertTrue(out[1]["quality"]["keep"])
        self.assertEqual(
            out[1]["quality"]["checks"]["inter_object_clip"]["status"], "ok"
        )


class TestMarkQualityRejections(unittest.TestCase):
    def test_sets_error_and_counts_failure(self):
        placements = [
            {
                "mask": "mask_0.png",
                "translation": [0.0, 1.6, 0.0],
                "scale": {"factor": 1.0},
                "quality": {
                    "keep": False,
                    "fails": ["surface_class"],
                    "checks": {
                        "surface_class": {
                            "reason": "label 'door' expects surface 'wall', got 'floor'"
                        }
                    },
                },
            },
            {
                "mask": "mask_1.png",
                "translation": [0.5, 1.6, 0.0],
                "scale": {"factor": 1.0},
                "quality": {"keep": True, "fails": [], "checks": {}},
            },
            {
                "mask": "mask_2.png",
                "translation": None,
                "scale": None,
                "error": "unresolved: translation, scale",
            },
        ]
        out, success, failure = mark_quality_rejections(placements)
        self.assertEqual(success, 1)
        self.assertEqual(failure, 2)
        self.assertEqual(out[0]["error"], "Item on wrong surface")
        self.assertEqual(out[0]["quality"]["message"], "Item on wrong surface")
        self.assertEqual(out[1]["quality"]["message"], "")
        self.assertNotIn("error", out[1])
        self.assertEqual(out[2]["error"], "unresolved: translation, scale")


class TestQualityCheckMessage(unittest.TestCase):
    def test_pass_is_empty(self):
        self.assertEqual(quality_check_message([]), "")
        self.assertEqual(quality_check_message(None), "")

    def test_plan_priority(self):
        self.assertEqual(quality_check_message(["surface_class"]), "Item on wrong surface")
        self.assertEqual(quality_check_message(["inside_footprint"]), "Item is too big")
        self.assertEqual(quality_check_message(["wall_penetration"]), "Item is too big")
        self.assertEqual(quality_check_message(["height_band"]), "Item is too big")
        self.assertEqual(
            quality_check_message(["inter_object_clip"]),
            "Item clipped into another object",
        )
        self.assertEqual(
            quality_check_message(["inter_object_clip", "surface_class"]),
            "Item on wrong surface",
        )


if __name__ == "__main__":
    unittest.main()
