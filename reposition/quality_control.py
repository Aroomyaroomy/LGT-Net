"""Placement quality checks — Plan A (label/surface) + Plan B (layout) + Plan C (overlap)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .lgt_utils import (
    JSON_TO_OBJ3D,
    floor_polygon_from_layout,
    layout_y_bounds,
    nearest_wall,
    point_in_floor_polygon,
)
from .sam_utils import mask_index_from_name
from .size_priors import size_prior

WALL_MOUNTED_LABELS = frozenset(
    "door doors window windows painting paintings picture pictures artwork "
    "shelf shelves bookshelf bookshelves bookcase bookcases".split()
)
FLOOR_LABELS = frozenset("rug rugs carpet carpets mat mats".split())
_LABEL_SCORE_RE = re.compile(r"^(?P<label>.*?)\s*\((?P<score>[0-9.]+)\)\s*$")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

DEFAULT_FOOTPRINT_OUTSIDE_CORNERS = 3
DEFAULT_WALL_PENETRATION_M = 0.14
DEFAULT_WALL_ATTACHMENT_M = 0.30
DEFAULT_FLOOR_Y_TOL_M = 0.40
DEFAULT_HEIGHT_CEILING_SLACK_M = 0.20
DEFAULT_WALL_Y_MARGIN_M = 0.05
DEFAULT_HALF_XZ_MIN_M = 0.12
DEFAULT_HALF_XZ_FRAC = 0.35
DEFAULT_OBJECT_HEIGHT_M = 0.80
DEFAULT_OVERLAP_RATIO = 0.22
TYPICAL_OBJECT_HEIGHTS_M = {
    "sofachairbed": 0.85, "cabinetshelf": 0.90, "bookshelf": 1.80, "bookcase": 1.80,
    "cabinet": 0.90, "table": 0.75, "desk": 0.75, "chair": 0.85, "shelf": 1.50,
    "sofa": 0.85, "couch": 0.85, "bed": 0.55,
}
_DUMMY_LAYOUT = {"cameraHeight": 1.6, "cameraCeilingHeight": 1.0, "layoutHeight": 2.6}
_MSG_SURFACE, _MSG_SIZE, _MSG_CLIP = (
    "Item on wrong surface",
    "Item is too big",
    "Item clipped into another object",
)
_PLAN_B_FAILS = frozenset(
    {"inside_footprint", "wall_penetration", "wall_attachment", "height_band"}
)


def quality_check_message(fails) -> str:
    """Empty if passed; otherwise one of the three Plan A/B/C messages."""
    fails = set(fails or [])
    if "surface_class" in fails:
        return _MSG_SURFACE
    if fails & _PLAN_B_FAILS:
        return _MSG_SIZE
    if "inter_object_clip" in fails:
        return _MSG_CLIP
    return ""


def normalize_dino_label(label: str | None) -> str | None:
    """Lowercase DINO phrase for allowlist lookup: ``"Painting(0.82)"`` → ``"painting"``."""
    text = str(label or "").strip()
    if not text:
        return None
    match = _LABEL_SCORE_RE.match(text)
    if match:
        text = match.group("label").strip()
    text = _NON_ALNUM_RE.sub("", text.lower().strip())
    return text or None


def typical_height_for_label(
    label: str | None, default: float | None = DEFAULT_OBJECT_HEIGHT_M
) -> float | None:
    """Class-typical standing height in metres, or ``default`` when unknown."""
    prior = size_prior(label)
    if prior and 2 in prior["axes"] and np.isfinite(prior["median"][2]):
        return float(prior["median"][2])
    token = normalize_dino_label(label)
    if token:
        for key in sorted(TYPICAL_OBJECT_HEIGHTS_M, key=len, reverse=True):
            if key in token:
                return float(TYPICAL_OBJECT_HEIGHTS_M[key])
    try:
        value = float(default) if default is not None else None
    except (TypeError, ValueError):
        return None
    return value if value is not None and np.isfinite(value) and value > 1e-4 else None


def expected_surface_for_label(label: str | None) -> str | None:
    """``"wall"`` / ``"floor"`` when the label is on an allowlist, else ``None``."""
    token = normalize_dino_label(label)
    if token in WALL_MOUNTED_LABELS:
        return "wall"
    if token in FLOOR_LABELS:
        return "floor"
    return None


def detections_aligned_to_masks(
    boxes: dict | list | Path | str | None, min_score: float | None = None
) -> list[dict]:
    """Detection records in SAM ``mask_i`` order (same filtering as box prompts)."""
    if boxes is None:
        return []
    if not isinstance(boxes, (dict, list)):
        with open(Path(boxes), encoding="utf-8") as f:
            boxes = json.load(f)
    detections = boxes if isinstance(boxes, list) else (boxes.get("detections") or [])
    out = []
    for det in detections:
        if not isinstance(det, dict):
            continue
        if min_score is not None and (
            det.get("score") is None or float(det["score"]) < float(min_score)
        ):
            continue
        box = det.get("box")
        try:
            if int(box["xMin"]) >= int(box["xMax"]) or int(box["yMin"]) >= int(box["yMax"]):
                continue
        except (TypeError, KeyError, ValueError):
            continue
        out.append(det)
    return out


def label_map_from_dino_boxes(
    boxes: dict | list | Path | str | None, min_score: float | None = None
) -> dict[int, dict[str, Any]]:
    """Map mask index → ``{label, score, raw_label}`` from DINO boxes."""
    out = {}
    for i, det in enumerate(detections_aligned_to_masks(boxes, min_score=min_score)):
        raw = det.get("label")
        out[i] = {
            "detection_id": det.get("detection_id", f"detection_{i:04d}"),
            "mask_index": i,
            "label": normalize_dino_label(raw),
            "raw_label": raw,
            "score": det.get("score"),
        }
    return out


def _result(name: str, status: str = "ok", keep: bool = True, reason=None, **extra) -> dict:
    return {"check": name, "status": status, "keep": keep, "reason": reason, **extra}


def _fail(name: str, reason: str, **extra) -> dict:
    return _result(name, status="fail", keep=False, reason=reason, **extra)


def _skip(name: str, reason: str, **extra) -> dict:
    return _result(name, status="skip", reason=reason, **extra)


def check_surface_class_consistency(placement: dict, label: str | None = None) -> dict:
    """Plan A: wall-required labels must not sit on the floor (and vice versa)."""
    raw = label if label is not None else placement.get("label")
    if raw is None and isinstance(placement.get("dino"), dict):
        raw = placement["dino"].get("label") or placement["dino"].get("raw_label")
    token, expected, actual = (
        normalize_dino_label(raw),
        expected_surface_for_label(raw),
        placement.get("surface"),
    )
    base = dict(label=token, raw_label=raw, expected_surface=expected, actual_surface=actual)
    if expected is None:
        return _skip("surface_class", "label_not_in_allowlist", **base)
    if actual is None:
        return _skip("surface_class", "missing_surface", **base)
    if actual != expected:
        return _fail(
            "surface_class",
            f"label '{token}' expects surface '{expected}', got '{actual}'",
            **base,
        )
    return _result("surface_class", **base)


def placement_half_extents(
    placement: dict,
    default_height: float = DEFAULT_OBJECT_HEIGHT_M,
    half_xz_frac: float = DEFAULT_HALF_XZ_FRAC,
    half_xz_min: float = DEFAULT_HALF_XZ_MIN_M,
) -> tuple[float, float, float]:
    """Cheap AABB half-size ``(hx, height, hz)`` from scale metadata (no mesh)."""
    scale = placement.get("scale") if isinstance(placement.get("scale"), dict) else {}
    dimensions = scale.get("target_dimensions")
    if dimensions is not None:
        width, depth, height = map(float, dimensions)
        offsets = _dimension_offsets(placement, width, depth)
        hx, hz = np.max(np.abs(offsets[:, [0, 2]]), axis=0)
        return float(hx), height, float(hz)
    height = None
    if scale.get("target_height") is not None:
        height = float(scale["target_height"])
    elif scale.get("object_height") is not None:
        height = float(scale["object_height"]) * float(scale.get("factor") or 1.0)
    if height is None or not np.isfinite(height) or height <= 1e-4:
        height = float(default_height)
    try:
        half_xz = 0.5 * float(scale.get("factor") or 1.0) * float(scale["native_xz"])
    except (TypeError, ValueError, KeyError):
        half_xz = 0.0
    if not np.isfinite(half_xz) or half_xz <= 1e-4:
        half_xz = float(half_xz_frac) * height
    return max(float(half_xz_min), float(half_xz)), height, max(float(half_xz_min), float(half_xz))


def _translation_xyz(placement: dict) -> np.ndarray | None:
    t = placement.get("translation")
    if t is None:
        return None
    arr = np.asarray(t, dtype=np.float64).reshape(-1)
    return arr if arr.shape[0] == 3 and np.all(np.isfinite(arr)) else None


def _dimension_offsets(placement: dict, width: float, depth: float) -> np.ndarray:
    local = np.array([
        [-width / 2, 0, -depth / 2], [-width / 2, 0, depth / 2],
        [width / 2, 0, -depth / 2], [width / 2, 0, depth / 2],
    ])
    rotation = placement.get("rotation") or {}
    matrix = rotation.get("matrix") if isinstance(rotation, dict) else None
    offsets = local @ (np.asarray(matrix).reshape(3, 3) if matrix is not None else np.eye(3)).T
    return offsets @ JSON_TO_OBJ3D if rotation.get("frame") == "obj3d" else offsets


def footprint_corners_xz(placement: dict, half_xz: float | None = None) -> list[np.ndarray] | None:
    """Four horizontal corners around the placement translation."""
    t = _translation_xyz(placement)
    if t is None:
        return None
    scale = placement.get("scale") or {}
    dimensions = scale.get("target_dimensions")
    if half_xz is None and dimensions is not None:
        width, depth = map(float, dimensions[:2])
        return [c[[0, 2]] for c in _dimension_offsets(placement, width, depth) + t]
    hx, hz = (float(half_xz), float(half_xz)) if half_xz is not None else placement_half_extents(placement)[::2]
    x, z = float(t[0]), float(t[2])
    return [
        np.array([x - hx, z - hz]), np.array([x + hx, z - hz]),
        np.array([x - hx, z + hz]), np.array([x + hx, z + hz]),
    ]


def aabb_minmax_json(placement: dict, data: dict | None = None):
    """``(min_xyz, max_xyz)`` for the cheap placement AABB, or None."""
    t = _translation_xyz(placement)
    if t is None:
        return None
    hx, height, hz = placement_half_extents(placement)
    layout = data if isinstance(data, dict) and "cameraHeight" in data else _DUMMY_LAYOUT
    floor_y, _ = layout_y_bounds(layout)
    cx, cy, cz = float(t[0]), float(t[1]), float(t[2])
    if placement.get("surface") == "floor":
        y0 = cy if abs(cy - floor_y) <= DEFAULT_FLOOR_Y_TOL_M else floor_y
        ymin, ymax = y0 - height, y0
    else:
        ymin, ymax = cy - 0.5 * height, cy + 0.5 * height
    return (
        np.array([cx - hx, ymin, cz - hz], dtype=np.float64),
        np.array([cx + hx, ymax, cz + hz], dtype=np.float64),
    )


def aabb_overlap_volume(min_a, max_a, min_b, max_b) -> float:
    overlap = np.maximum(0.0, np.minimum(max_a, max_b) - np.maximum(min_a, min_b))
    return float(np.prod(np.asarray(overlap, dtype=np.float64)))


def aabb_overlap_ratio(min_a, max_a, min_b, max_b) -> float:
    """Overlap volume / min(vol_a, vol_b). 0 ⇒ none; 1 ⇒ smaller box fully inside."""
    overlap = aabb_overlap_volume(min_a, max_a, min_b, max_b)
    if overlap <= 0.0:
        return 0.0
    vols = [
        float(np.prod(np.maximum(0.0, np.asarray(mx) - np.asarray(mn))))
        for mn, mx in ((min_a, max_a), (min_b, max_b))
    ]
    denom = min(vols)
    return 0.0 if denom <= 1e-12 else float(overlap / denom)


def _floor_footprint_geometry(placement: dict, data: dict):
    """Shared Plan B floor stats, or ``(skip_reason, None, 0, 0.0)``."""
    if placement.get("surface") != "floor":
        return "not_floor", None, 0, 0.0
    t = _translation_xyz(placement)
    if t is None:
        return "missing_translation", None, 0, 0.0
    try:
        polygon = floor_polygon_from_layout(data)
    except (KeyError, ValueError) as exc:
        return f"no_footprint: {exc}", None, 0, 0.0
    poly = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    outside, max_pen = 0, 0.0
    for c in footprint_corners_xz(placement) or []:
        dist = float(cv2.pointPolygonTest(poly, (float(c[0]), float(c[1])), True))
        if dist < 0.0:
            outside += 1
            max_pen = max(max_pen, -dist)
    return None, point_in_floor_polygon(t, polygon), outside, max_pen


def check_inside_footprint(
    placement: dict, data: dict, outside_corner_fail: int = DEFAULT_FOOTPRINT_OUTSIDE_CORNERS
) -> dict:
    """Plan B: floor objects must sit inside the layout footprint."""
    skip, center_inside, outside, _ = _floor_footprint_geometry(placement, data)
    extra = dict(outside_corners=int(outside), center_inside=None if center_inside is None else bool(center_inside))
    if skip:
        return _skip("inside_footprint", skip, **extra)
    if not center_inside or outside >= int(outside_corner_fail):
        return _fail(
            "inside_footprint",
            f"footprint outside (center_inside={center_inside}, outside_corners={outside}/4)",
            **extra,
        )
    if outside:
        return _result("inside_footprint", status="warn", reason=f"{outside}/4 footprint corners outside", **extra)
    return _result("inside_footprint", **extra)


def check_wall_penetration(
    placement: dict, data: dict, max_penetration_m: float = DEFAULT_WALL_PENETRATION_M
) -> dict:
    """Plan B: floor AABB must not stick clearly outside the room footprint."""
    skip, _c, _o, max_pen = _floor_footprint_geometry(placement, data)
    extra = dict(max_penetration_m=float(max_pen))
    if skip:
        return _skip("wall_penetration", skip, **extra)
    if max_pen > float(max_penetration_m):
        return _fail(
            "wall_penetration",
            f"AABB extends {max_pen:.3f}m outside footprint (limit {max_penetration_m:.3f}m)",
            **extra,
        )
    return _result("wall_penetration", **extra)


def check_wall_attachment(
    placement: dict, data: dict, max_distance_m: float = DEFAULT_WALL_ATTACHMENT_M
) -> dict:
    """Plan B: wall-mounted objects must lie close to the nearest layout wall."""
    extra: dict[str, Any] = dict(distance_m=None)
    if placement.get("surface") != "wall":
        return _skip("wall_attachment", "not_wall", **extra)
    t = _translation_xyz(placement)
    if t is None:
        return _skip("wall_attachment", "missing_translation", **extra)
    wall, dist, _ = nearest_wall(t, data)
    if wall is None:
        return _skip("wall_attachment", "no_walls", **extra)
    extra["distance_m"] = abs(float(dist))
    if extra["distance_m"] > float(max_distance_m):
        return _fail(
            "wall_attachment",
            f"wall object {extra['distance_m']:.3f}m from nearest wall (limit {max_distance_m:.3f}m)",
            **extra,
        )
    return _result("wall_attachment", **extra)


def check_height_band(
    placement: dict,
    data: dict,
    floor_y_tol_m: float = DEFAULT_FLOOR_Y_TOL_M,
    ceiling_slack_m: float = DEFAULT_HEIGHT_CEILING_SLACK_M,
    wall_y_margin_m: float = DEFAULT_WALL_Y_MARGIN_M,
) -> dict:
    """Plan B: floor contact near the floor plane; wall center between floor and ceiling."""
    t = _translation_xyz(placement)
    extra: dict[str, Any] = dict(floor_y=None, ceiling_y=None, translation_y=None, top_y=None)
    if t is None:
        return _skip("height_band", "missing_translation", **extra)
    floor_y, ceiling_y = layout_y_bounds(data)
    extra.update(floor_y=floor_y, ceiling_y=ceiling_y, translation_y=float(t[1]))
    _, height, _ = placement_half_extents(placement)
    surface, y = placement.get("surface"), float(t[1])
    if surface == "floor":
        extra["top_y"] = y - height
        if abs(y - floor_y) > float(floor_y_tol_m):
            return _fail(
                "height_band",
                f"floor contact y={y:.3f} far from floor {floor_y:.3f} (Δ={abs(y - floor_y):.3f}m)",
                **extra,
            )
        if extra["top_y"] < ceiling_y - float(ceiling_slack_m):
            return _fail(
                "height_band",
                f"object top y={extra['top_y']:.3f} pierces ceiling {ceiling_y:.3f}",
                **extra,
            )
        return _result("height_band", **extra)
    if surface == "wall":
        lo, hi = ceiling_y + float(wall_y_margin_m), floor_y - float(wall_y_margin_m)
        extra["top_y"] = y + 0.5 * height
        if y < lo or y > hi:
            return _fail("height_band", f"wall object y={y:.3f} outside band [{lo:.3f}, {hi:.3f}]", **extra)
        return _result("height_band", **extra)
    return _skip("height_band", "unknown_surface", **extra)


def _merge_check(quality: dict, name: str, check: dict) -> dict:
    quality = quality or {}
    checks = dict(quality.get("checks") or {})
    checks[name] = check
    fails = [f for f in (quality.get("fails") or []) if f != name]
    warns = [w for w in (quality.get("warns") or []) if w != name]
    if check.get("status") == "fail":
        fails.append(name)
    elif check.get("status") == "warn":
        warns.append(name)
    return {
        "keep": not fails, "fails": fails, "warns": warns, "checks": checks,
        "message": quality_check_message(fails),
    }


def _keep(placement: dict) -> bool:
    return (placement.get("quality") or {}).get("keep", True)


def _finish(record: dict, quality: dict, drop_rejected: bool) -> dict | None:
    record["quality"] = quality
    return None if drop_rejected and not quality.get("keep", True) else record


def _append(out: list, record: dict, quality: dict, drop_rejected: bool) -> None:
    done = _finish(record, quality, drop_rejected)
    if done is not None:
        out.append(done)


def apply_surface_class_quality(
    placements: list[dict] | None,
    boxes: dict | list | Path | str | None = None,
    min_score: float | None = None,
    label_map: dict[int, dict[str, Any]] | None = None,
    drop_rejected: bool = False,
) -> list[dict]:
    """Annotate placements with DINO labels and Plan A surface-class quality."""
    if not placements:
        return []
    if label_map is None:
        label_map = label_map_from_dino_boxes(boxes, min_score=min_score)
    out = []
    for placement in placements:
        record = dict(placement)
        idx = mask_index_from_name(record.get("mask") or "")
        info = label_map.get(idx) if idx is not None else None
        if info is not None:
            record["label"] = info.get("label")
            record["dino"] = {
                "label": info.get("label"), "raw_label": info.get("raw_label"), "score": info.get("score")
            }
        _append(
            out, record,
            _merge_check(record.get("quality") or {}, "surface_class",
                         check_surface_class_consistency(record, label=record.get("label"))),
            drop_rejected,
        )
    return out


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
    """Plan B: annotate placements with footprint / wall / height geometry checks."""
    if not placements:
        return []
    if not isinstance(layout, dict) or "layoutPoints" not in layout or "layoutWalls" not in layout:
        return [dict(p) for p in placements]
    out = []
    for placement in placements:
        record = dict(placement)
        quality = dict(record.get("quality") or {})
        for name, check in (
            ("inside_footprint", check_inside_footprint(record, layout, outside_corner_fail=outside_corner_fail)),
            ("wall_penetration", check_wall_penetration(record, layout, max_penetration_m=wall_penetration_m)),
            ("wall_attachment", check_wall_attachment(record, layout, max_distance_m=wall_attachment_m)),
            ("height_band", check_height_band(record, layout, floor_y_tol_m=floor_y_tol_m)),
        ):
            quality = _merge_check(quality, name, check)
        _append(out, record, quality, drop_rejected)
    return out


def apply_inter_object_clipping_quality(
    placements: list[dict] | None,
    layout: dict | None = None,
    *,
    overlap_ratio_thresh: float = DEFAULT_OVERLAP_RATIO,
    drop_rejected: bool = False,
    only_kept: bool = True,
) -> list[dict]:
    """Plan C: reject placements whose cheap AABBs overlap another object."""
    if not placements:
        return []
    records = [dict(p) for p in placements]
    thresh = float(overlap_ratio_thresh)
    boxes, eligible, partners = [], [], [[] for _ in records]
    for record in records:
        bounds = aabb_minmax_json(record, layout if isinstance(layout, dict) else None)
        keep = _keep(record)
        boxes.append(bounds)
        eligible.append(bounds is not None and record.get("translation") is not None and (keep or not only_kept))
    n = len(records)
    for i in range(n):
        if not eligible[i]:
            continue
        min_i, max_i = boxes[i]
        for j in range(i + 1, n):
            if not eligible[j]:
                continue
            ratio = aabb_overlap_ratio(min_i, max_i, *boxes[j])
            if ratio < thresh:
                continue
            vol = aabb_overlap_volume(min_i, max_i, *boxes[j])
            hit = {"ratio": float(ratio), "volume": float(vol)}
            partners[i].append({**hit, "mask": records[j].get("mask"), "index": j})
            partners[j].append({**hit, "mask": records[i].get("mask"), "index": i})
    out = []
    for idx, record in enumerate(records):
        quality = dict(record.get("quality") or {})
        extra = dict(overlaps=partners[idx], overlap_ratio_thresh=thresh)
        if boxes[idx] is None:
            check = _skip("inter_object_clip", "missing_aabb", overlaps=[], overlap_ratio_thresh=thresh)
        elif only_kept and not _keep(record):
            check = _skip("inter_object_clip", "already_rejected", overlaps=[], overlap_ratio_thresh=thresh)
        elif partners[idx]:
            worst = max(partners[idx], key=lambda o: o["ratio"])
            check = _fail(
                "inter_object_clip",
                f"AABB overlaps {worst.get('mask')} (ratio={worst['ratio']:.3f} ≥ {thresh:.3f})",
                **extra,
            )
        else:
            check = _result("inter_object_clip", **extra)
        _append(out, record, _merge_check(quality, "inter_object_clip", check), drop_rejected)
    return out


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
        placements, boxes=boxes, min_score=min_score, label_map=label_map, drop_rejected=False
    )
    annotated = apply_layout_geometry_quality(
        annotated, layout, wall_penetration_m=wall_penetration_m,
        wall_attachment_m=wall_attachment_m, floor_y_tol_m=floor_y_tol_m, drop_rejected=False,
    )
    return apply_inter_object_clipping_quality(
        annotated, layout, overlap_ratio_thresh=overlap_ratio_thresh,
        drop_rejected=drop_rejected, only_kept=True,
    )


def filter_kept_placements(placements: list[dict] | None) -> list[dict]:
    """Return placements that pass quality (missing quality ⇒ keep)."""
    return [p for p in (placements or []) if _keep(p)]


def quality_rejection_error(placement: dict) -> str:
    """Compact ``record['error']`` from QC failures."""
    quality = placement.get("quality") or {}
    fails, checks = list(quality.get("fails") or []), quality.get("checks") or {}
    if not fails:
        return "quality: rejected"
    parts = []
    for name in fails:
        check = checks.get(name) if isinstance(checks, dict) else None
        reason = check.get("reason") if isinstance(check, dict) else None
        parts.append(f"{name}: {reason}" if reason else str(name))
    return f"quality: {'; '.join(parts)}"


def mark_quality_rejections(placements: list[dict] | None) -> tuple[list[dict], int, int]:
    """Set ``error`` on QC-rejected placements and recount success / failure."""
    if not placements:
        return [], 0, 0
    out, success, failure = [], 0, 0
    for placement in placements:
        record = dict(placement)
        quality = dict(record.get("quality") or {})
        quality["message"] = quality_check_message(quality.get("fails"))
        record["quality"] = quality
        if quality.get("keep") is False:
            record["error"] = quality["message"] or quality_rejection_error(record)
        scale = record.get("scale")
        posed = record.get("translation") is not None and isinstance(scale, dict) and scale.get("factor") is not None
        if posed and quality.get("keep", True) and not record.get("error"):
            success += 1
        else:
            failure += 1
        out.append(record)
    return out, success, failure
