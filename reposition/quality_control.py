"""Placement quality checks — Plan A (label/surface) + Plan B (layout geometry)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .lgt_utils import (
    floor_polygon_from_layout,
    horizontal_wall_normal,
    inward_wall_normal,
    nearest_wall,
    point_in_floor_polygon,
)
from .sam_utils import mask_index_from_name

# Labels that must be wall-mounted. Keep this narrow: only classes that are
# almost never freestanding furniture in a room layout.
WALL_MOUNTED_LABELS: frozenset[str] = frozenset(
    {
        "door",
        "doors",
        "window",
        "windows",
        "painting",
        "paintings",
        "picture",
        "pictures",
        "artwork",
        "shelf",
        "shelves",
        "bookshelf",
        "bookshelves",
        "bookcase",
        "bookcases",
    }
)

# Optional floor-only allowlist (Plan A focus is wall; extend later if needed).
FLOOR_LABELS: frozenset[str] = frozenset(
    {
        "rug",
        "rugs",
        "carpet",
        "carpets",
        "mat",
        "mats",
    }
)

_LABEL_SCORE_RE = re.compile(r"^(?P<label>.*?)\s*\((?P<score>[0-9.]+)\)\s*$")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

# cheap reject thresholds for "obviously broken".
DEFAULT_FOOTPRINT_OUTSIDE_CORNERS = 3  # of 4 XZ corners
DEFAULT_WALL_PENETRATION_M = 0.12
DEFAULT_WALL_ATTACHMENT_M = 0.30
DEFAULT_FLOOR_Y_TOL_M = 0.40
DEFAULT_HEIGHT_CEILING_SLACK_M = 0.20
DEFAULT_WALL_Y_MARGIN_M = 0.05
DEFAULT_HALF_XZ_MIN_M = 0.12
DEFAULT_HALF_XZ_FRAC = 0.35
DEFAULT_OBJECT_HEIGHT_M = 0.80
# Plan C: overlap_vol / min(vol_a, vol_b) above this ⇒ reject both.
DEFAULT_OVERLAP_RATIO = 0.20


def normalize_dino_label(label: str | None) -> str | None:
    """
    Normalize a GroundingDINO phrase to a lowercase token for allowlist lookup.

    Strips trailing ``(score)``, punctuation, and collapses whitespace.
    ``"Painting(0.82)"`` → ``"painting"``, ``"book shelf"`` → ``"bookshelf"``.
    """
    if label is None:
        return None
    text = str(label).strip()
    if not text:
        return None
    match = _LABEL_SCORE_RE.match(text)
    if match:
        text = match.group("label").strip()
    text = text.lower().strip()
    # Join multi-word forms so "book shelf" matches "bookshelf".
    text = _NON_ALNUM_RE.sub("", text)
    return text or None


def expected_surface_for_label(label: str | None) -> str | None:
    """
    Return ``"wall"`` / ``"floor"`` when the label is on an allowlist,
    else ``None`` (no hard surface constraint).
    """
    token = normalize_dino_label(label)
    if token is None:
        return None
    if token in WALL_MOUNTED_LABELS:
        return "wall"
    if token in FLOOR_LABELS:
        return "floor"
    return None


def _load_boxes_payload(
    boxes: dict | list | Path | str | None,
) -> dict | list | None:
    if boxes is None:
        return None
    if isinstance(boxes, (dict, list)):
        return boxes
    path = Path(boxes)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def detections_aligned_to_masks(
    boxes: dict | list | Path | str | None,
    min_score: float | None = None,
) -> list[dict]:
    """
    Return detection records in the same order as SAM ``mask_i`` indices.

    Mirrors ``get_pano_masks.load_box_prompts_from_json`` filtering:
      - If ``min_score`` is set, keep detections with ``score >= min_score``.
      - If ``min_score`` is None and ``box_prompts`` exists, use full
        ``detections`` in order (same length as prompts sent to SAM).
    """
    payload = _load_boxes_payload(boxes)
    if payload is None:
        return []

    if isinstance(payload, list):
        detections = [d for d in payload if isinstance(d, dict)]
    else:
        detections = [
            d for d in (payload.get("detections") or []) if isinstance(d, dict)
        ]

    if min_score is not None:
        detections = [
            d
            for d in detections
            if d.get("score") is not None and float(d["score"]) >= float(min_score)
        ]

    aligned: list[dict] = []
    for det in detections:
        box = det.get("box")
        if not isinstance(box, dict):
            continue
        try:
            if int(box["xMin"]) >= int(box["xMax"]) or int(box["yMin"]) >= int(
                box["yMax"]
            ):
                continue
        except (KeyError, TypeError, ValueError):
            continue
        aligned.append(det)
    return aligned


def label_map_from_dino_boxes(
    boxes: dict | list | Path | str | None,
    min_score: float | None = None,
) -> dict[int, dict[str, Any]]:
    """
    Map mask index → ``{label, score, raw_label}`` from DINO boxes.

    ``mask_i.png`` corresponds to index ``i`` in the aligned detection list.
    """
    out: dict[int, dict[str, Any]] = {}
    for i, det in enumerate(detections_aligned_to_masks(boxes, min_score=min_score)):
        raw = det.get("label")
        out[i] = {
            "label": normalize_dino_label(raw),
            "raw_label": raw,
            "score": det.get("score"),
        }
    return out


def check_surface_class_consistency(
    placement: dict,
    label: str | None = None,
) -> dict:
    """
    Plan A: reject wall-required labels that were placed on the floor (and
    symmetrically for floor-only labels on a wall).
    """
    raw_label = label if label is not None else placement.get("label")
    if raw_label is None and isinstance(placement.get("dino"), dict):
        raw_label = placement["dino"].get("label") or placement["dino"].get("raw_label")

    token = normalize_dino_label(raw_label)
    expected = expected_surface_for_label(token)
    actual = placement.get("surface")

    result = {
        "check": "surface_class",
        "status": "ok",
        "keep": True,
        "label": token,
        "raw_label": raw_label,
        "expected_surface": expected,
        "actual_surface": actual,
        "reason": None,
    }

    if expected is None:
        result["status"] = "skip"
        result["reason"] = "label_not_in_allowlist"
        return result

    if actual is None:
        result["status"] = "skip"
        result["reason"] = "missing_surface"
        return result

    if actual != expected:
        result["status"] = "fail"
        result["keep"] = False
        result["reason"] = (
            f"label '{token}' expects surface '{expected}', got '{actual}'"
        )
        return result

    return result


# ---------------------------------------------------------------------------
# Plan B — cheap layout geometry checks (JSON / xyz2json frame)
# ---------------------------------------------------------------------------


def _layout_y_bounds(data: dict) -> tuple[float, float, float]:
    """Return ``(floor_y, ceiling_y, layout_height)`` in the JSON frame."""
    floor_y = float(data["cameraHeight"])
    ceiling = float(data.get("cameraCeilingHeight", data.get("layoutHeight", 2.6) - floor_y))
    ceiling_y = -float(ceiling)
    layout_height = float(data.get("layoutHeight", floor_y + ceiling))
    return floor_y, ceiling_y, layout_height


def placement_half_extents(
    placement: dict,
    default_height: float = DEFAULT_OBJECT_HEIGHT_M,
    half_xz_frac: float = DEFAULT_HALF_XZ_FRAC,
    half_xz_min: float = DEFAULT_HALF_XZ_MIN_M,
) -> tuple[float, float, float]:
    """
    Cheap axis-aligned half-size proxy ``(hx, height, hz)`` without loading a mesh.

    Height prefers ``scale.target_height``, else ``object_height * factor``,
    else ``default_height``. Horizontal half-extents scale with height.
    """
    scale = placement.get("scale") if isinstance(placement.get("scale"), dict) else {}
    height = None
    if scale.get("target_height") is not None:
        height = float(scale["target_height"])
    elif scale.get("object_height") is not None:
        factor = float(scale.get("factor") or 1.0)
        height = float(scale["object_height"]) * factor
    if height is None or not np.isfinite(height) or height <= 1e-4:
        height = float(default_height)
    half_xz = max(float(half_xz_min), float(half_xz_frac) * height)
    return float(half_xz), float(height), float(half_xz)


def _translation_xyz(placement: dict) -> np.ndarray | None:
    t = placement.get("translation")
    if t is None:
        return None
    arr = np.asarray(t, dtype=np.float64).reshape(-1)
    if arr.shape[0] != 3 or not np.all(np.isfinite(arr)):
        return None
    return arr


def footprint_corners_xz(
    placement: dict,
    half_xz: float | None = None,
) -> list[np.ndarray] | None:
    """Four horizontal AABB corners around the placement translation."""
    t = _translation_xyz(placement)
    if t is None:
        return None
    if half_xz is None:
        half_xz, _, _ = placement_half_extents(placement)
    hx = float(half_xz)
    x, z = float(t[0]), float(t[2])
    return [
        np.array([x - hx, z - hx], dtype=np.float64),
        np.array([x + hx, z - hx], dtype=np.float64),
        np.array([x - hx, z + hx], dtype=np.float64),
        np.array([x + hx, z + hx], dtype=np.float64),
    ]


def aabb_corners_json(placement: dict, data: dict) -> np.ndarray | None:
    """
    Approximate 8 AABB corners in the JSON frame.

    Floor objects: contact on the floor plane, body extends toward the ceiling
    (``-Y``). Wall objects: box centered on the translation.
    """
    t = _translation_xyz(placement)
    if t is None:
        return None
    hx, height, hz = placement_half_extents(placement)
    floor_y, _, _ = _layout_y_bounds(data)
    surface = placement.get("surface")

    if surface == "floor":
        y0 = float(t[1]) if abs(float(t[1]) - floor_y) <= DEFAULT_FLOOR_Y_TOL_M else floor_y
        ymin = y0 - height
        ymax = y0
        cx, cz = float(t[0]), float(t[2])
    else:
        cy = float(t[1])
        ymin = cy - 0.5 * height
        ymax = cy + 0.5 * height
        cx, cz = float(t[0]), float(t[2])

    corners = []
    for dx in (-hx, hx):
        for dy in (ymin, ymax):
            for dz in (-hz, hz):
                corners.append([cx + dx, dy, cz + dz])
    return np.asarray(corners, dtype=np.float64)


def aabb_minmax_json(
    placement: dict,
    data: dict | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(min_xyz, max_xyz)`` for the cheap placement AABB, or None."""
    layout = data if isinstance(data, dict) and "cameraHeight" in data else {
        "cameraHeight": 1.6,
        "cameraCeilingHeight": 1.0,
        "layoutHeight": 2.6,
    }
    corners = aabb_corners_json(placement, layout)
    if corners is None or len(corners) == 0:
        return None
    return corners.min(axis=0), corners.max(axis=0)


def aabb_volume(min_b: np.ndarray, max_b: np.ndarray) -> float:
    """Axis-aligned box volume (0 if degenerate)."""
    extents = np.maximum(0.0, np.asarray(max_b, dtype=np.float64) - np.asarray(min_b, dtype=np.float64))
    return float(np.prod(extents))


def aabb_overlap_volume(
    min_a: np.ndarray,
    max_a: np.ndarray,
    min_b: np.ndarray,
    max_b: np.ndarray,
) -> float:
    """Intersection volume of two AABBs."""
    overlap = np.maximum(
        0.0,
        np.minimum(np.asarray(max_a, dtype=np.float64), np.asarray(max_b, dtype=np.float64))
        - np.maximum(np.asarray(min_a, dtype=np.float64), np.asarray(min_b, dtype=np.float64)),
    )
    return float(np.prod(overlap))


def aabb_overlap_ratio(
    min_a: np.ndarray,
    max_a: np.ndarray,
    min_b: np.ndarray,
    max_b: np.ndarray,
) -> float:
    """
    Overlap volume divided by the smaller AABB volume.

    0 ⇒ no overlap; 1 ⇒ the smaller box is fully inside the other.
    """
    overlap = aabb_overlap_volume(min_a, max_a, min_b, max_b)
    if overlap <= 0.0:
        return 0.0
    vol_a = aabb_volume(min_a, max_a)
    vol_b = aabb_volume(min_b, max_b)
    denom = min(vol_a, vol_b)
    if denom <= 1e-12:
        return 0.0
    return float(overlap / denom)


def _room_centroid_json(data: dict) -> np.ndarray:
    """Horizontal centroid of the layout footprint (JSON frame)."""
    try:
        poly = floor_polygon_from_layout(data)
        c = np.asarray(poly, dtype=np.float64).mean(axis=0)
        return np.array([float(c[0]), 0.0, float(c[1])], dtype=np.float64)
    except (KeyError, ValueError):
        return np.zeros(3, dtype=np.float64)


def _signed_inside_wall_distance(
    point: np.ndarray,
    wall: dict,
    data: dict | None = None,
    room_center: np.ndarray | None = None,
) -> float:
    """
    Positive ⇒ point is on the room (inward) side of the wall plane.
    Negative ⇒ past the wall (outside / penetrating).
    """
    plane = np.asarray(wall["planeEquation"], dtype=np.float64).reshape(4)
    n = plane[:3]
    n_len = float(np.linalg.norm(n))
    if n_len < 1e-12:
        return 0.0
    n_unit = n / n_len
    signed = float(np.dot(n_unit, point) + plane[3] / n_len)

    if room_center is None:
        room_center = (
            _room_centroid_json(data)
            if data is not None
            else np.zeros(3, dtype=np.float64)
        )

    # Determine inward using the wall segment midpoint (not the query point /
    # not the centroid alone — those make the to_center vector degenerate).
    ref_point = room_center
    if data is not None:
        try:
            points = data["layoutPoints"]["points"]
            i0, i1 = wall["pointsIdx"]
            a = np.asarray(points[i0]["xyz"], dtype=np.float64)
            b = np.asarray(points[i1]["xyz"], dtype=np.float64)
            ref_point = 0.5 * (a + b)
        except (KeyError, IndexError, TypeError, ValueError):
            ref_point = room_center

    try:
        n_in = inward_wall_normal(n, ref_point, room_center=room_center)
    except ValueError:
        try:
            n_in = horizontal_wall_normal(n)
        except ValueError:
            return signed
    if float(np.dot(n_unit, n_in)) < 0.0:
        signed = -signed
    return signed


def check_inside_footprint(
    placement: dict,
    data: dict,
    outside_corner_fail: int = DEFAULT_FOOTPRINT_OUTSIDE_CORNERS,
) -> dict:
    """
    Plan B: floor objects must sit inside the layout footprint.

    Uses the 4 XZ AABB corners. Rejects when the center is outside or when
    ``outside_corner_fail`` or more corners are outside (default 3/4).
    """
    result: dict[str, Any] = {
        "check": "inside_footprint",
        "status": "ok",
        "keep": True,
        "reason": None,
        "outside_corners": 0,
        "center_inside": None,
    }
    if placement.get("surface") != "floor":
        result["status"] = "skip"
        result["reason"] = "not_floor"
        return result
    if _translation_xyz(placement) is None:
        result["status"] = "skip"
        result["reason"] = "missing_translation"
        return result

    try:
        polygon = floor_polygon_from_layout(data)
    except (KeyError, ValueError) as exc:
        result["status"] = "skip"
        result["reason"] = f"no_footprint: {exc}"
        return result

    t = _translation_xyz(placement)
    center_inside = point_in_floor_polygon(t, polygon)
    result["center_inside"] = bool(center_inside)

    corners = footprint_corners_xz(placement)
    outside = 0
    if corners is not None:
        for c in corners:
            if not point_in_floor_polygon(c, polygon):
                outside += 1
    result["outside_corners"] = int(outside)

    if not center_inside or outside >= int(outside_corner_fail):
        result["status"] = "fail"
        result["keep"] = False
        result["reason"] = (
            f"footprint outside (center_inside={center_inside}, "
            f"outside_corners={outside}/4)"
        )
        return result

    if outside > 0:
        result["status"] = "warn"
        result["reason"] = f"{outside}/4 footprint corners outside"
    return result


def check_wall_penetration(
    placement: dict,
    data: dict,
    max_penetration_m: float = DEFAULT_WALL_PENETRATION_M,
) -> dict:
    """
    Plan B: freestanding AABB must not stick clearly outside the room.

    Uses the layout footprint (not infinite wall planes) so L-shaped / notched
    rooms do not false-positive on walls whose planes cut through free space.
    Penetration = max distance of AABB XZ corners outside the polygon.
    """
    result: dict[str, Any] = {
        "check": "wall_penetration",
        "status": "ok",
        "keep": True,
        "reason": None,
        "max_penetration_m": 0.0,
    }
    if placement.get("surface") != "floor":
        result["status"] = "skip"
        result["reason"] = "not_floor"
        return result

    if _translation_xyz(placement) is None:
        result["status"] = "skip"
        result["reason"] = "missing_translation"
        return result

    try:
        polygon = floor_polygon_from_layout(data)
    except (KeyError, ValueError) as exc:
        result["status"] = "skip"
        result["reason"] = f"no_footprint: {exc}"
        return result

    corners = footprint_corners_xz(placement)
    if not corners:
        result["status"] = "skip"
        result["reason"] = "missing_corners"
        return result

    poly = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    max_pen = 0.0
    for c in corners:
        # measureDist=True: negative outside, 0 on edge, positive inside.
        dist = float(
            cv2.pointPolygonTest(poly, (float(c[0]), float(c[1])), True)
        )
        if dist < 0.0:
            max_pen = max(max_pen, -dist)

    result["max_penetration_m"] = float(max_pen)
    if max_pen > float(max_penetration_m):
        result["status"] = "fail"
        result["keep"] = False
        result["reason"] = (
            f"AABB extends {max_pen:.3f}m outside footprint "
            f"(limit {max_penetration_m:.3f}m)"
        )
    return result


def check_wall_attachment(
    placement: dict,
    data: dict,
    max_distance_m: float = DEFAULT_WALL_ATTACHMENT_M,
) -> dict:
    """
    Plan B: wall-mounted objects must lie close to the nearest layout wall.
    """
    result: dict[str, Any] = {
        "check": "wall_attachment",
        "status": "ok",
        "keep": True,
        "reason": None,
        "distance_m": None,
    }
    if placement.get("surface") != "wall":
        result["status"] = "skip"
        result["reason"] = "not_wall"
        return result

    t = _translation_xyz(placement)
    if t is None:
        result["status"] = "skip"
        result["reason"] = "missing_translation"
        return result

    wall, dist, _normal = nearest_wall(t, data)
    if wall is None:
        result["status"] = "skip"
        result["reason"] = "no_walls"
        return result

    distance = abs(float(dist))
    result["distance_m"] = distance
    if distance > float(max_distance_m):
        result["status"] = "fail"
        result["keep"] = False
        result["reason"] = (
            f"wall object {distance:.3f}m from nearest wall "
            f"(limit {max_distance_m:.3f}m)"
        )
    return result


def check_height_band(
    placement: dict,
    data: dict,
    floor_y_tol_m: float = DEFAULT_FLOOR_Y_TOL_M,
    ceiling_slack_m: float = DEFAULT_HEIGHT_CEILING_SLACK_M,
    wall_y_margin_m: float = DEFAULT_WALL_Y_MARGIN_M,
) -> dict:
    """
    Plan B: height sanity in the JSON frame.

    Floor objects: contact near the floor plane; top must not clearly pierce
    the ceiling. Wall objects: center height between floor and ceiling.
    """
    result: dict[str, Any] = {
        "check": "height_band",
        "status": "ok",
        "keep": True,
        "reason": None,
        "floor_y": None,
        "ceiling_y": None,
        "translation_y": None,
        "top_y": None,
    }
    t = _translation_xyz(placement)
    if t is None:
        result["status"] = "skip"
        result["reason"] = "missing_translation"
        return result

    floor_y, ceiling_y, _layout_h = _layout_y_bounds(data)
    result["floor_y"] = floor_y
    result["ceiling_y"] = ceiling_y
    result["translation_y"] = float(t[1])
    surface = placement.get("surface")
    _hx, height, _hz = placement_half_extents(placement)

    if surface == "floor":
        dy = abs(float(t[1]) - floor_y)
        top_y = float(t[1]) - height  # toward ceiling (−Y)
        result["top_y"] = top_y
        if dy > float(floor_y_tol_m):
            result["status"] = "fail"
            result["keep"] = False
            result["reason"] = (
                f"floor contact y={float(t[1]):.3f} far from floor "
                f"{floor_y:.3f} (Δ={dy:.3f}m)"
            )
            return result
        # Ceiling is at ceiling_y (more negative). Top pierces if top_y < ceiling_y - slack.
        if top_y < ceiling_y - float(ceiling_slack_m):
            result["status"] = "fail"
            result["keep"] = False
            result["reason"] = (
                f"object top y={top_y:.3f} pierces ceiling {ceiling_y:.3f}"
            )
            return result
        return result

    if surface == "wall":
        y = float(t[1])
        lo = ceiling_y + float(wall_y_margin_m)
        hi = floor_y - float(wall_y_margin_m)
        result["top_y"] = y + 0.5 * height
        if y < lo or y > hi:
            result["status"] = "fail"
            result["keep"] = False
            result["reason"] = (
                f"wall object y={y:.3f} outside band [{lo:.3f}, {hi:.3f}]"
            )
        return result

    result["status"] = "skip"
    result["reason"] = "unknown_surface"
    return result


def _merge_check(quality: dict, name: str, check: dict) -> dict:
    """Fold one check into a placement ``quality`` dict (hard fail + warns)."""
    quality = dict(quality or {})
    checks = dict(quality.get("checks") or {})
    checks[name] = check

    fails = [f for f in (quality.get("fails") or []) if f != name]
    warns = [w for w in (quality.get("warns") or []) if w != name]
    if check.get("status") == "fail":
        fails.append(name)
    elif check.get("status") == "warn":
        warns.append(name)

    prev_keep = quality.get("keep")
    keep = True if prev_keep is None else bool(prev_keep)
    if check.get("status") == "fail":
        keep = False

    quality.update({"keep": keep, "fails": fails, "warns": warns, "checks": checks})
    return quality


def apply_surface_class_quality(
    placements: list[dict] | None,
    boxes: dict | list | Path | str | None = None,
    min_score: float | None = None,
    label_map: dict[int, dict[str, Any]] | None = None,
    drop_rejected: bool = False,
) -> list[dict]:
    """
    Annotate placements with DINO labels and Plan A surface-class quality.

    Each placement gains::

        label: normalized DINO label (or None)
        dino: {label, raw_label, score} when available
        quality: {
          keep: bool,
          fails: [...],
          warns: [...],
          checks: {surface_class: {...}},
        }
    """
    if not placements:
        return []

    if label_map is None:
        label_map = label_map_from_dino_boxes(boxes, min_score=min_score)

    annotated: list[dict] = []
    for placement in placements:
        record = dict(placement)
        idx = mask_index_from_name(record.get("mask") or "")
        info = label_map.get(idx) if idx is not None else None

        if info is not None:
            record["label"] = info.get("label")
            record["dino"] = {
                "label": info.get("label"),
                "raw_label": info.get("raw_label"),
                "score": info.get("score"),
            }
        else:
            record.setdefault("label", record.get("label"))

        check = check_surface_class_consistency(record, label=record.get("label"))
        record["quality"] = _merge_check(record.get("quality") or {}, "surface_class", check)

        if drop_rejected and not record["quality"].get("keep", True):
            continue
        annotated.append(record)

    return annotated


def apply_layout_geometry_quality(
    placements: list[dict] | None,
    layout: dict | None,
    *,
    wall_penetration_m: float = DEFAULT_WALL_PENETRATION_M,
    wall_attachment_m: float = DEFAULT_WALL_ATTACHMENT_M,
    floor_y_tol_m: float = DEFAULT_FLOOR_Y_TOL_M,
    outside_corner_fail: int = DEFAULT_FOOTPRINT_OUTSIDE_CORNERS,
    drop_rejected: bool = False,
) -> list[dict]:
    """
    Plan B: annotate placements with footprint / wall / height geometry checks.

    Requires LGT ``coordinates`` (layout JSON). When ``layout`` is missing,
    Plan B checks are skipped and placements are returned unchanged.
    """
    if not placements:
        return []
    if not layout or not isinstance(layout, dict):
        return [dict(p) for p in placements]
    if "layoutPoints" not in layout or "layoutWalls" not in layout:
        return [dict(p) for p in placements]

    annotated: list[dict] = []
    for placement in placements:
        record = dict(placement)
        quality = dict(record.get("quality") or {})

        checks = [
            (
                "inside_footprint",
                check_inside_footprint(
                    record, layout, outside_corner_fail=outside_corner_fail
                ),
            ),
            (
                "wall_penetration",
                check_wall_penetration(
                    record, layout, max_penetration_m=wall_penetration_m
                ),
            ),
            (
                "wall_attachment",
                check_wall_attachment(
                    record, layout, max_distance_m=wall_attachment_m
                ),
            ),
            (
                "height_band",
                check_height_band(
                    record, layout, floor_y_tol_m=floor_y_tol_m
                ),
            ),
        ]
        for name, check in checks:
            quality = _merge_check(quality, name, check)

        record["quality"] = quality
        if drop_rejected and not quality.get("keep", True):
            continue
        annotated.append(record)

    return annotated


def apply_inter_object_clipping_quality(
    placements: list[dict] | None,
    layout: dict | None = None,
    *,
    overlap_ratio_thresh: float = DEFAULT_OVERLAP_RATIO,
    drop_rejected: bool = False,
    only_kept: bool = True,
) -> list[dict]:
    """
    Plan C: reject placements whose cheap AABBs overlap another object.

    Pairwise overlap volume / min(volume) above ``overlap_ratio_thresh``
    (default 0.15) marks **both** members of the pair as failed
    (``inter_object_clip``). Grazing contacts below the threshold are ignored.

    When ``only_kept`` is True (default), objects already rejected by Plan A/B
    are excluded from the pair search (but still receive a skip check).
    """
    if not placements:
        return []

    records = [dict(p) for p in placements]
    layout_data = layout if isinstance(layout, dict) else None

    # Build AABBs once.
    boxes: list[tuple[np.ndarray, np.ndarray] | None] = []
    eligible: list[bool] = []
    for record in records:
        bounds = aabb_minmax_json(record, layout_data)
        boxes.append(bounds)
        keep = (record.get("quality") or {}).get("keep", True)
        eligible.append(
            bounds is not None
            and record.get("translation") is not None
            and (keep if only_kept else True)
        )

    # Pairwise overlaps → list of partner infos per index.
    partners: list[list[dict]] = [[] for _ in records]
    n = len(records)
    for i in range(n):
        if not eligible[i] or boxes[i] is None:
            continue
        min_i, max_i = boxes[i]
        for j in range(i + 1, n):
            if not eligible[j] or boxes[j] is None:
                continue
            min_j, max_j = boxes[j]
            ratio = aabb_overlap_ratio(min_i, max_i, min_j, max_j)
            if ratio < float(overlap_ratio_thresh):
                continue
            vol = aabb_overlap_volume(min_i, max_i, min_j, max_j)
            info_i = {
                "mask": records[j].get("mask"),
                "index": j,
                "ratio": float(ratio),
                "volume": float(vol),
            }
            info_j = {
                "mask": records[i].get("mask"),
                "index": i,
                "ratio": float(ratio),
                "volume": float(vol),
            }
            partners[i].append(info_i)
            partners[j].append(info_j)

    annotated: list[dict] = []
    for idx, record in enumerate(records):
        quality = dict(record.get("quality") or {})
        if boxes[idx] is None:
            check = {
                "check": "inter_object_clip",
                "status": "skip",
                "keep": True,
                "reason": "missing_aabb",
                "overlaps": [],
                "overlap_ratio_thresh": float(overlap_ratio_thresh),
            }
        elif only_kept and not (quality.get("keep", True)):
            check = {
                "check": "inter_object_clip",
                "status": "skip",
                "keep": True,
                "reason": "already_rejected",
                "overlaps": [],
                "overlap_ratio_thresh": float(overlap_ratio_thresh),
            }
        elif partners[idx]:
            worst = max(partners[idx], key=lambda o: o["ratio"])
            check = {
                "check": "inter_object_clip",
                "status": "fail",
                "keep": False,
                "reason": (
                    f"AABB overlaps {worst.get('mask')} "
                    f"(ratio={worst['ratio']:.3f} ≥ {overlap_ratio_thresh:.3f})"
                ),
                "overlaps": partners[idx],
                "overlap_ratio_thresh": float(overlap_ratio_thresh),
            }
        else:
            check = {
                "check": "inter_object_clip",
                "status": "ok",
                "keep": True,
                "reason": None,
                "overlaps": [],
                "overlap_ratio_thresh": float(overlap_ratio_thresh),
            }

        quality = _merge_check(quality, "inter_object_clip", check)
        record["quality"] = quality
        if drop_rejected and not quality.get("keep", True):
            continue
        annotated.append(record)

    return annotated


def apply_placement_quality(
    placements: list[dict] | None,
    *,
    boxes: dict | list | Path | str | None = None,
    layout: dict | None = None,
    min_score: float | None = None,
    label_map: dict[int, dict[str, Any]] | None = None,
    drop_rejected: bool = False,
    wall_penetration_m: float = DEFAULT_WALL_PENETRATION_M,
    wall_attachment_m: float = DEFAULT_WALL_ATTACHMENT_M,
    floor_y_tol_m: float = DEFAULT_FLOOR_Y_TOL_M,
    overlap_ratio_thresh: float = DEFAULT_OVERLAP_RATIO,
) -> list[dict]:
    """Run Plan A, Plan B, then Plan C and return annotated placements."""
    annotated = apply_surface_class_quality(
        placements,
        boxes=boxes,
        min_score=min_score,
        label_map=label_map,
        drop_rejected=False,
    )
    annotated = apply_layout_geometry_quality(
        annotated,
        layout,
        wall_penetration_m=wall_penetration_m,
        wall_attachment_m=wall_attachment_m,
        floor_y_tol_m=floor_y_tol_m,
        drop_rejected=False,
    )
    annotated = apply_inter_object_clipping_quality(
        annotated,
        layout,
        overlap_ratio_thresh=overlap_ratio_thresh,
        drop_rejected=drop_rejected,
        only_kept=True,
    )
    return annotated


def filter_kept_placements(placements: list[dict] | None) -> list[dict]:
    """Return placements that pass quality (missing quality ⇒ keep)."""
    if not placements:
        return []
    return [
        p
        for p in placements
        if (p.get("quality") or {}).get("keep", True)
    ]


def quality_rejection_error(placement: dict) -> str:
    """
    Build a compact ``record['error']`` string from quality-control failures.

    Example: ``quality: surface_class: label 'door' expects surface 'wall', got 'floor'``.
    """
    quality = placement.get("quality") or {}
    fails = list(quality.get("fails") or [])
    checks = quality.get("checks") or {}
    if not fails:
        return "quality: rejected"
    parts: list[str] = []
    for name in fails:
        check = checks.get(name) if isinstance(checks, dict) else None
        reason = None
        if isinstance(check, dict):
            reason = check.get("reason")
        if reason:
            parts.append(f"{name}: {reason}")
        else:
            parts.append(str(name))
    return f"quality: {'; '.join(parts)}"


def mark_quality_rejections(
    placements: list[dict] | None,
) -> tuple[list[dict], int, int]:
    """
    Set ``error`` on QC-rejected placements and recount success / failure.

    Success = pose-resolved (translation + scale.factor) AND ``quality.keep``
    (default True) AND no ``error``. Everything else counts as failure,
    including unresolved translation/scale and quality-control skips.
    """
    if not placements:
        return [], 0, 0

    out: list[dict] = []
    success = 0
    failure = 0
    for placement in placements:
        record = dict(placement)
        quality = record.get("quality") or {}
        if quality.get("keep") is False:
            record["error"] = quality_rejection_error(record)

        scale = record.get("scale")
        has_pose = (
            record.get("translation") is not None
            and isinstance(scale, dict)
            and scale.get("factor") is not None
        )
        kept = quality.get("keep", True)
        if has_pose and kept and not record.get("error"):
            success += 1
        else:
            failure += 1
        out.append(record)

    return out, success, failure
