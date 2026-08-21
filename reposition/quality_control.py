"""Placement quality checks — Plan A (label/surface) + Plan B (layout) + Plan C (overlap)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .lgt_utils import (
    floor_polygon_from_layout,
    layout_y_bounds,
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

FLOOR_LABELS: frozenset[str] = frozenset(
    {"rug", "rugs", "carpet", "carpets", "mat", "mats"}
)

_LABEL_SCORE_RE = re.compile(r"^(?P<label>.*?)\s*\((?P<score>[0-9.]+)\)\s*$")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

DEFAULT_FOOTPRINT_OUTSIDE_CORNERS = 3  # of 4 XZ corners
DEFAULT_WALL_PENETRATION_M = 0.14
DEFAULT_WALL_ATTACHMENT_M = 0.30
DEFAULT_FLOOR_Y_TOL_M = 0.40
DEFAULT_HEIGHT_CEILING_SLACK_M = 0.20
DEFAULT_WALL_Y_MARGIN_M = 0.05
DEFAULT_HALF_XZ_MIN_M = 0.12
DEFAULT_HALF_XZ_FRAC = 0.35
DEFAULT_OBJECT_HEIGHT_M = 0.80
DEFAULT_OVERLAP_RATIO = 0.22

# Typical standing heights used as the mesh-scale estimator (metres).
# Longer keys win so concatenated DINO tokens like ``sofachairbed`` resolve.
TYPICAL_OBJECT_HEIGHTS_M: dict[str, float] = {
    "sofachairbed": 0.85,
    "cabinetshelf": 0.90,
    "bookshelf": 1.80,
    "bookcase": 1.80,
    "cabinet": 0.90,
    "table": 0.75,
    "desk": 0.75,
    "chair": 0.85,
    "shelf": 1.50,
    "sofa": 0.85,
    "couch": 0.85,
    "bed": 0.55,
}

_DUMMY_LAYOUT = {
    "cameraHeight": 1.6,
    "cameraCeilingHeight": 1.0,
    "layoutHeight": 2.6,
}


def normalize_dino_label(label: str | None) -> str | None:
    """Lowercase DINO phrase for allowlist lookup: ``"Painting(0.82)"`` → ``"painting"``."""
    if not label:
        return None
    text = str(label).strip()
    if not text:
        return None
    match = _LABEL_SCORE_RE.match(text)
    if match:
        text = match.group("label").strip()
    text = _NON_ALNUM_RE.sub("", text.lower().strip())
    return text or None


def typical_height_for_label(
    label: str | None,
    default: float | None = DEFAULT_OBJECT_HEIGHT_M,
) -> float | None:
    """Class-typical standing height in metres, or ``default`` when unknown."""
    token = normalize_dino_label(label)
    if token:
        for key in sorted(TYPICAL_OBJECT_HEIGHTS_M, key=len, reverse=True):
            if key in token:
                return float(TYPICAL_OBJECT_HEIGHTS_M[key])
    if default is None:
        return None
    try:
        value = float(default)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) and value > 1e-4 else None


def expected_surface_for_label(label: str | None) -> str | None:
    """``"wall"`` / ``"floor"`` when the label is on an allowlist, else ``None``."""
    token = normalize_dino_label(label)
    if token in WALL_MOUNTED_LABELS:
        return "wall"
    if token in FLOOR_LABELS:
        return "floor"
    return None


def detections_aligned_to_masks(
    boxes: dict | list | Path | str | None,
    min_score: float | None = None,
) -> list[dict]:
    """Detection records in SAM ``mask_i`` order (same filtering as box prompts)."""
    if boxes is None:
        return []
    if isinstance(boxes, (dict, list)):
        payload = boxes
    else:
        with open(Path(boxes), encoding="utf-8") as f:
            payload = json.load(f)

    detections = payload if isinstance(payload, list) else (payload.get("detections") or [])
    detections = [d for d in detections if isinstance(d, dict)]
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
            if int(box["xMin"]) >= int(box["xMax"]) or int(box["yMin"]) >= int(box["yMax"]):
                continue
        except (KeyError, TypeError, ValueError):
            continue
        aligned.append(det)
    return aligned


def label_map_from_dino_boxes(
    boxes: dict | list | Path | str | None,
    min_score: float | None = None,
) -> dict[int, dict[str, Any]]:
    """Map mask index → ``{label, score, raw_label}`` from DINO boxes."""
    out: dict[int, dict[str, Any]] = {}
    for i, det in enumerate(detections_aligned_to_masks(boxes, min_score=min_score)):
        raw = det.get("label")
        out[i] = {
            "label": normalize_dino_label(raw),
            "raw_label": raw,
            "score": det.get("score"),
        }
    return out


def _result(name: str, status: str = "ok", keep: bool = True, reason=None, **extra) -> dict:
    return {"check": name, "status": status, "keep": keep, "reason": reason, **extra}


def check_surface_class_consistency(placement: dict, label: str | None = None) -> dict:
    """Plan A: wall-required labels must not sit on the floor (and vice versa)."""
    raw_label = label if label is not None else placement.get("label")
    if raw_label is None and isinstance(placement.get("dino"), dict):
        raw_label = placement["dino"].get("label") or placement["dino"].get("raw_label")

    token = normalize_dino_label(raw_label)
    expected = expected_surface_for_label(raw_label)
    actual = placement.get("surface")
    base = dict(label=token, raw_label=raw_label, expected_surface=expected, actual_surface=actual)

    if expected is None:
        return _result("surface_class", status="skip", reason="label_not_in_allowlist", **base)
    if actual is None:
        return _result("surface_class", status="skip", reason="missing_surface", **base)
    if actual != expected:
        return _result(
            "surface_class",
            status="fail",
            keep=False,
            reason=f"label '{token}' expects surface '{expected}', got '{actual}'",
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
    height = None
    if scale.get("target_height") is not None:
        height = float(scale["target_height"])
    elif scale.get("object_height") is not None:
        height = float(scale["object_height"]) * float(scale.get("factor") or 1.0)
    if height is None or not np.isfinite(height) or height <= 1e-4:
        height = float(default_height)
    native_xz = scale.get("native_xz")
    factor = float(scale.get("factor") or 1.0)
    if native_xz is not None:
        try:
            half_xz = 0.5 * factor * float(native_xz)
        except (TypeError, ValueError):
            half_xz = 0.0
        if not np.isfinite(half_xz) or half_xz <= 1e-4:
            half_xz = max(float(half_xz_min), float(half_xz_frac) * height)
        else:
            half_xz = max(float(half_xz_min), float(half_xz))
    else:
        half_xz = max(float(half_xz_min), float(half_xz_frac) * height)
    return half_xz, height, half_xz


def _translation_xyz(placement: dict) -> np.ndarray | None:
    t = placement.get("translation")
    if t is None:
        return None
    arr = np.asarray(t, dtype=np.float64).reshape(-1)
    if arr.shape[0] != 3 or not np.all(np.isfinite(arr)):
        return None
    return arr


def footprint_corners_xz(placement: dict, half_xz: float | None = None) -> list[np.ndarray] | None:
    """Four horizontal AABB corners around the placement translation."""
    t = _translation_xyz(placement)
    if t is None:
        return None
    if half_xz is None:
        half_xz, _, _ = placement_half_extents(placement)
    x, z, hx = float(t[0]), float(t[2]), float(half_xz)
    return [
        np.array([x - hx, z - hx], dtype=np.float64),
        np.array([x + hx, z - hx], dtype=np.float64),
        np.array([x - hx, z + hx], dtype=np.float64),
        np.array([x + hx, z + hx], dtype=np.float64),
    ]


def aabb_minmax_json(
    placement: dict,
    data: dict | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """``(min_xyz, max_xyz)`` for the cheap placement AABB, or None."""
    t = _translation_xyz(placement)
    if t is None:
        return None
    hx, height, hz = placement_half_extents(placement)
    layout = data if isinstance(data, dict) and "cameraHeight" in data else _DUMMY_LAYOUT
    floor_y, _ = layout_y_bounds(layout)
    cx, cz = float(t[0]), float(t[2])
    if placement.get("surface") == "floor":
        y0 = float(t[1]) if abs(float(t[1]) - floor_y) <= DEFAULT_FLOOR_Y_TOL_M else floor_y
        ymin, ymax = y0 - height, y0
    else:
        cy = float(t[1])
        ymin, ymax = cy - 0.5 * height, cy + 0.5 * height
    return (
        np.array([cx - hx, ymin, cz - hz], dtype=np.float64),
        np.array([cx + hx, ymax, cz + hz], dtype=np.float64),
    )


def aabb_overlap_volume(
    min_a: np.ndarray, max_a: np.ndarray, min_b: np.ndarray, max_b: np.ndarray
) -> float:
    overlap = np.maximum(
        0.0,
        np.minimum(np.asarray(max_a, dtype=np.float64), np.asarray(max_b, dtype=np.float64))
        - np.maximum(np.asarray(min_a, dtype=np.float64), np.asarray(min_b, dtype=np.float64)),
    )
    return float(np.prod(overlap))


def aabb_overlap_ratio(
    min_a: np.ndarray, max_a: np.ndarray, min_b: np.ndarray, max_b: np.ndarray
) -> float:
    """Overlap volume / min(vol_a, vol_b). 0 ⇒ none; 1 ⇒ smaller box fully inside."""
    overlap = aabb_overlap_volume(min_a, max_a, min_b, max_b)
    if overlap <= 0.0:
        return 0.0
    vol_a = float(np.prod(np.maximum(0.0, np.asarray(max_a) - np.asarray(min_a))))
    vol_b = float(np.prod(np.maximum(0.0, np.asarray(max_b) - np.asarray(min_b))))
    denom = min(vol_a, vol_b)
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
    outside = 0
    max_pen = 0.0
    for c in footprint_corners_xz(placement) or []:
        dist = float(cv2.pointPolygonTest(poly, (float(c[0]), float(c[1])), True))
        if dist < 0.0:
            outside += 1
            max_pen = max(max_pen, -dist)
    return None, point_in_floor_polygon(t, polygon), outside, max_pen


def check_inside_footprint(
    placement: dict,
    data: dict,
    outside_corner_fail: int = DEFAULT_FOOTPRINT_OUTSIDE_CORNERS,
) -> dict:
    """Plan B: floor objects must sit inside the layout footprint."""
    skip, center_inside, outside, _pen = _floor_footprint_geometry(placement, data)
    extra = dict(outside_corners=int(outside), center_inside=None if center_inside is None else bool(center_inside))
    if skip:
        return _result("inside_footprint", status="skip", reason=skip, **extra)
    if not center_inside or outside >= int(outside_corner_fail):
        return _result(
            "inside_footprint",
            status="fail",
            keep=False,
            reason=f"footprint outside (center_inside={center_inside}, outside_corners={outside}/4)",
            **extra,
        )
    if outside > 0:
        return _result("inside_footprint", status="warn", reason=f"{outside}/4 footprint corners outside", **extra)
    return _result("inside_footprint", **extra)


def check_wall_penetration(
    placement: dict,
    data: dict,
    max_penetration_m: float = DEFAULT_WALL_PENETRATION_M,
) -> dict:
    """Plan B: floor AABB must not stick clearly outside the room footprint."""
    skip, _center, _outside, max_pen = _floor_footprint_geometry(placement, data)
    extra = dict(max_penetration_m=float(max_pen))
    if skip:
        return _result("wall_penetration", status="skip", reason=skip, **extra)
    if max_pen > float(max_penetration_m):
        return _result(
            "wall_penetration",
            status="fail",
            keep=False,
            reason=f"AABB extends {max_pen:.3f}m outside footprint (limit {max_penetration_m:.3f}m)",
            **extra,
        )
    return _result("wall_penetration", **extra)


def check_wall_attachment(
    placement: dict,
    data: dict,
    max_distance_m: float = DEFAULT_WALL_ATTACHMENT_M,
) -> dict:
    """Plan B: wall-mounted objects must lie close to the nearest layout wall."""
    extra: dict[str, Any] = dict(distance_m=None)
    if placement.get("surface") != "wall":
        return _result("wall_attachment", status="skip", reason="not_wall", **extra)
    t = _translation_xyz(placement)
    if t is None:
        return _result("wall_attachment", status="skip", reason="missing_translation", **extra)
    wall, dist, _normal = nearest_wall(t, data)
    if wall is None:
        return _result("wall_attachment", status="skip", reason="no_walls", **extra)
    distance = abs(float(dist))
    extra["distance_m"] = distance
    if distance > float(max_distance_m):
        return _result(
            "wall_attachment",
            status="fail",
            keep=False,
            reason=f"wall object {distance:.3f}m from nearest wall (limit {max_distance_m:.3f}m)",
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
    extra: dict[str, Any] = dict(floor_y=None, ceiling_y=None, translation_y=None, top_y=None)
    t = _translation_xyz(placement)
    if t is None:
        return _result("height_band", status="skip", reason="missing_translation", **extra)

    floor_y, ceiling_y = layout_y_bounds(data)
    extra.update(floor_y=floor_y, ceiling_y=ceiling_y, translation_y=float(t[1]))
    _, height, _ = placement_half_extents(placement)
    surface = placement.get("surface")

    if surface == "floor":
        dy = abs(float(t[1]) - floor_y)
        top_y = float(t[1]) - height
        extra["top_y"] = top_y
        if dy > float(floor_y_tol_m):
            return _result(
                "height_band",
                status="fail",
                keep=False,
                reason=f"floor contact y={float(t[1]):.3f} far from floor {floor_y:.3f} (Δ={dy:.3f}m)",
                **extra,
            )
        if top_y < ceiling_y - float(ceiling_slack_m):
            return _result(
                "height_band",
                status="fail",
                keep=False,
                reason=f"object top y={top_y:.3f} pierces ceiling {ceiling_y:.3f}",
                **extra,
            )
        return _result("height_band", **extra)

    if surface == "wall":
        y = float(t[1])
        lo = ceiling_y + float(wall_y_margin_m)
        hi = floor_y - float(wall_y_margin_m)
        extra["top_y"] = y + 0.5 * height
        if y < lo or y > hi:
            return _result(
                "height_band",
                status="fail",
                keep=False,
                reason=f"wall object y={y:.3f} outside band [{lo:.3f}, {hi:.3f}]",
                **extra,
            )
        return _result("height_band", **extra)

    return _result("height_band", status="skip", reason="unknown_surface", **extra)


def _merge_check(quality: dict, name: str, check: dict) -> dict:
    quality = dict(quality or {})
    checks = dict(quality.get("checks") or {})
    checks[name] = check
    fails = [f for f in (quality.get("fails") or []) if f != name]
    warns = [w for w in (quality.get("warns") or []) if w != name]
    if check.get("status") == "fail":
        fails.append(name)
    elif check.get("status") == "warn":
        warns.append(name)
    keep = True if quality.get("keep") is None else bool(quality.get("keep"))
    if check.get("status") == "fail":
        keep = False
    quality.update({"keep": keep, "fails": fails, "warns": warns, "checks": checks})
    return quality


def _finish(record: dict, quality: dict, drop_rejected: bool) -> dict | None:
    record["quality"] = quality
    if drop_rejected and not quality.get("keep", True):
        return None
    return record


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
        check = check_surface_class_consistency(record, label=record.get("label"))
        done = _finish(record, _merge_check(record.get("quality") or {}, "surface_class", check), drop_rejected)
        if done is not None:
            annotated.append(done)
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
    """Plan B: annotate placements with footprint / wall / height geometry checks."""
    if not placements:
        return []
    if not isinstance(layout, dict) or "layoutPoints" not in layout or "layoutWalls" not in layout:
        return [dict(p) for p in placements]

    annotated: list[dict] = []
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
        done = _finish(record, quality, drop_rejected)
        if done is not None:
            annotated.append(done)
    return annotated


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
    layout_data = layout if isinstance(layout, dict) else None
    thresh = float(overlap_ratio_thresh)

    boxes: list[tuple[np.ndarray, np.ndarray] | None] = []
    eligible: list[bool] = []
    for record in records:
        bounds = aabb_minmax_json(record, layout_data)
        boxes.append(bounds)
        keep = (record.get("quality") or {}).get("keep", True)
        eligible.append(
            bounds is not None and record.get("translation") is not None and (keep if only_kept else True)
        )

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
            if ratio < thresh:
                continue
            vol = aabb_overlap_volume(min_i, max_i, min_j, max_j)
            partners[i].append({"mask": records[j].get("mask"), "index": j, "ratio": float(ratio), "volume": float(vol)})
            partners[j].append({"mask": records[i].get("mask"), "index": i, "ratio": float(ratio), "volume": float(vol)})

    annotated: list[dict] = []
    for idx, record in enumerate(records):
        quality = dict(record.get("quality") or {})
        if boxes[idx] is None:
            check = _result("inter_object_clip", status="skip", reason="missing_aabb", overlaps=[], overlap_ratio_thresh=thresh)
        elif only_kept and not quality.get("keep", True):
            check = _result("inter_object_clip", status="skip", reason="already_rejected", overlaps=[], overlap_ratio_thresh=thresh)
        elif partners[idx]:
            worst = max(partners[idx], key=lambda o: o["ratio"])
            check = _result(
                "inter_object_clip",
                status="fail",
                keep=False,
                reason=f"AABB overlaps {worst.get('mask')} (ratio={worst['ratio']:.3f} ≥ {thresh:.3f})",
                overlaps=partners[idx],
                overlap_ratio_thresh=thresh,
            )
        else:
            check = _result("inter_object_clip", overlaps=[], overlap_ratio_thresh=thresh)
        done = _finish(record, _merge_check(quality, "inter_object_clip", check), drop_rejected)
        if done is not None:
            annotated.append(done)
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
        placements, boxes=boxes, min_score=min_score, label_map=label_map, drop_rejected=False
    )
    annotated = apply_layout_geometry_quality(
        annotated,
        layout,
        wall_penetration_m=wall_penetration_m,
        wall_attachment_m=wall_attachment_m,
        floor_y_tol_m=floor_y_tol_m,
        drop_rejected=False,
    )
    return apply_inter_object_clipping_quality(
        annotated, layout, overlap_ratio_thresh=overlap_ratio_thresh, drop_rejected=drop_rejected, only_kept=True
    )


def filter_kept_placements(placements: list[dict] | None) -> list[dict]:
    """Return placements that pass quality (missing quality ⇒ keep)."""
    if not placements:
        return []
    return [p for p in placements if (p.get("quality") or {}).get("keep", True)]


def quality_rejection_error(placement: dict) -> str:
    """Compact ``record['error']`` from QC failures."""
    quality = placement.get("quality") or {}
    fails = list(quality.get("fails") or [])
    checks = quality.get("checks") or {}
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

    out: list[dict] = []
    success = failure = 0
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
        if has_pose and quality.get("keep", True) and not record.get("error"):
            success += 1
        else:
            failure += 1
        out.append(record)
    return out, success, failure
