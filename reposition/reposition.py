import os
from pathlib import Path

import numpy as np
from PIL import Image

from inference import preprocess

from .lgt_utils import (
    azimuth_unit_xz,
    check_floor_hit,
    closer_than_wall,
    contact_uvs_from_mask,
    height_from_angular_size,
    horizontal_range,
    inward_wall_normal,
    mask_angular_height,
    nearest_wall,
    range_from_angular_size,
    ray_intersect_floor,
    ray_intersect_wall,
    rotation_matrix_yaw,
    shrink_range_into_footprint,
    uv2equirectangular,
    wall_distance_from_depth,
    wall_fraction_from_elevation,
    yaw_from_wall_normal,
)
from .sam_utils import (
    load_sam3d_metadata,
    mask_index_from_name,
    object_height_from_sam3d,
)
from .quality_control import (
    apply_placement_quality,
    mark_quality_rejections,
)


def get_masks(mask_dir: Path) -> list[Path]:
    if not mask_dir.is_dir():
        raise NotADirectoryError(f"The {mask_dir} is not a directory")
    return sorted(
        (p for p in mask_dir.iterdir() if p.name.startswith("mask_") and p.name.endswith(".png")),
        key=lambda p: int(p.stem.split("_")[1]),
    )


def load_aligned_binary_mask(
    path: Path,
    do_manhattan: bool = True,
    vp_cache_path: str = None,
) -> np.ndarray:
    """
    Load a mask_*.png, resize to the LGT pano size, optionally apply the same
    VP rotation as the panorama (via vp_cache_path written by preprocess), and
    return a 2D boolean foreground mask.
    """
    img = np.array(Image.open(path).resize((1024, 512), Image.Resampling.NEAREST))
    if img.ndim == 2:
        rgb = np.stack([img, img, img], axis=-1)
    else:
        rgb = img[..., :3]

    if do_manhattan:
        if not vp_cache_path or not os.path.exists(vp_cache_path):
            raise ValueError(
                "do_manhattan requires an existing vp_cache_path from panorama preprocess"
            )
        rgb, _ = preprocess(rgb, vp_cache_path=vp_cache_path)

    gray = rgb.mean(axis=-1).astype(np.float32)
    if gray.max() > 1.0:
        gray = gray / 255.0
    return gray > 0.5

def placements_from_mask_dir(
    mask_dir: Path | str,
    data: dict,
    depth: np.ndarray = None,
    do_manhattan: bool = True,
    vp_cache_path: str = None,
    num_samples: int = 5,
    sam3d_metadata: dict[int, dict] | Path | str | None = None,
    furniture_dir: Path | str | None = None,
    dino_boxes: dict | list | Path | str | None = None,
    min_score: float | None = None,
) -> tuple[list[dict], int, int]:
    """
    For each mask_*.png under mask_dir, run resolve_standing_pose and
    return JSON-serializable placement records (translation, yaw rotation,
    and mesh scale).

    Optional SAM3D ``metadata`` / ``furniture_dir`` supply per-object heights
    for angular-size freestanding ranging and room-distance mesh scaling.
    Optional ``dino_boxes`` enables Plan A label/surface quality checks.

    Graceful failure policy (per component):
      - translation unresolved → skip object (never place at origin);
        ``error`` names the cause (``missing_layout``, ``no_contact_uvs``,
        ``outside_footprint``, ``no_wall_hit``, …)
      - scale unresolved → skip object (never invent a bare scale=1.0);
        ``error`` names the cause (``missing_object_height``, …)
      - rotation unresolved → keep object; default upright orientation is
        already applied inside ``resolve_standing_pose``
      - quality-control reject → skip object; reason written to ``error``

    ``failure`` counts skipped objects (unresolved pose **or** QC reject);
    ``success`` counts renderable ones that pass quality.
    """
    mask_dir = Path(mask_dir)
    metadata = None
    if sam3d_metadata is not None:
        metadata = (
            sam3d_metadata
            if isinstance(sam3d_metadata, dict)
            else load_sam3d_metadata(sam3d_metadata)
        )

    placements = []
    for path in get_masks(mask_dir):
        record = {
            'mask': path.name,
            'translation': None,
            'rotation': None,
            'contact_uv': None,
            'surface': None,
            'scale': None,
        }
        try:
            binary = load_aligned_binary_mask(
                path, do_manhattan=do_manhattan, vp_cache_path=vp_cache_path
            )
        except Exception as exc:
            record['error'] = str(exc)
            placements.append(record)
            continue

        if not binary.any():
            record['error'] = 'empty mask'
            placements.append(record)
            continue

        idx = mask_index_from_name(path.name)
        object_height = (
            object_height_from_sam3d(idx, metadata=metadata, furniture_dir=furniture_dir)
            if idx is not None
            else None
        )

        translation, contact_uv, surface, rotation, scale, error = resolve_standing_pose(
            binary,
            data,
            depth=depth,
            num_samples=num_samples,
            object_height=object_height,
        )
        # Skip whenever translation or scale is unresolved — those look broken
        # in the final scene. Rotation alone never causes a skip.
        if translation is None or scale is None:
            record['error'] = error or _unresolved_error(
                translation_detail="unknown" if translation is None else None,
                scale_detail="unknown" if scale is None else None,
            )
            placements.append(record)
            continue

        record['translation'] = np.asarray(translation, dtype=np.float64).reshape(3).tolist()
        if contact_uv is not None:
            record['contact_uv'] = np.asarray(contact_uv, dtype=np.float64).reshape(2).tolist()
        record['surface'] = surface
        record['rotation'] = rotation
        record['scale'] = scale
        placements.append(record)

    # Plan A/B/C: annotate quality, then fold rejects into error + failure count.
    placements = apply_placement_quality(
        placements,
        boxes=dino_boxes,
        layout=data,
        min_score=min_score,
        drop_rejected=False,
    )
    placements, success, failure = mark_quality_rejections(placements)
    return placements, success, failure


def _unresolved_error(
    *,
    translation_detail: str | None = None,
    scale_detail: str | None = None,
) -> str:
    parts = []
    if translation_detail:
        parts.append(f"translation: {translation_detail}")
    if scale_detail:
        parts.append(f"scale: {scale_detail}")
    return f"unresolved: {'; '.join(parts)}" if parts else "unresolved"


def _yaw_rotation_dict(
    yaw: float,
    normal: np.ndarray,
    *,
    source: str | None = None,
    range_method: str | None = None,
) -> dict:
    rec = {
        "yaw": float(yaw),
        "matrix": rotation_matrix_yaw(yaw).tolist(),
        "wall_normal": np.asarray(normal, dtype=np.float64).reshape(3).tolist(),
    }
    if source is not None:
        rec["source"] = source
    if range_method is not None:
        rec["range_method"] = range_method
    return rec


def default_standing_rotation(
    translation: np.ndarray,
    range_method: str | None = None,
) -> dict:
    """
    Upright fallback when wall-heuristic rotation cannot be resolved.

    Yaw so mesh local +Z faces the room center (JSON origin). Never returns
    None — rotation alone must not cause a placement skip.
    """
    t = np.asarray(translation, dtype=np.float64).reshape(3)
    to_center = np.array([-t[0], 0.0, -t[2]], dtype=np.float64)
    n = float(np.linalg.norm(to_center))
    normal = to_center / n if n >= 1e-8 else np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return _yaw_rotation_dict(
        yaw_from_wall_normal(normal),
        normal,
        source="default",
        range_method=range_method,
    )


def resolve_standing_rotation(
    translation: np.ndarray,
    data: dict,
    wall: dict = None,
) -> tuple[float | None, np.ndarray | None, np.ndarray | None]:
    """
    Build yaw-only rotation for a placed object at translation.

    Uses the given wall (wall-mounted hit) or the nearest layout wall
    (floor objects). Front faces flush into the room along the inward normal.

    Returns (yaw_radians, R_3x3, inward_normal) or (None, None, None).
    Callers should fall back to ``default_standing_rotation`` on failure —
    a missing rotation must not skip a otherwise-valid placement.
    """
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    if wall is not None:
        try:
            normal = inward_wall_normal(wall['normal'], translation)
        except ValueError:
            return None, None, None
    else:
        _, _, normal = nearest_wall(translation, data)
        if normal is None:
            return None, None, None

    yaw = yaw_from_wall_normal(normal)
    return yaw, rotation_matrix_yaw(yaw), normal


def _pose_preflight(data: dict) -> str | None:
    """Typical scene-level causes that make every mask fail translation."""
    if not isinstance(data, dict):
        return "missing_layout"
    if data.get("cameraHeight") is None:
        return "missing_camera_height"
    points = data.get("layoutPoints")
    walls = data.get("layoutWalls")
    n_pts = (points or {}).get("points") if isinstance(points, dict) else None
    n_walls = (walls or {}).get("walls") if isinstance(walls, dict) else None
    if not n_pts or not n_walls:
        return "missing_layout"
    return None


def freestanding_floor_translation(
    origin: np.ndarray,
    direction: np.ndarray,
    uv: np.ndarray,
    binary: np.ndarray,
    data: dict,
    depth: np.ndarray,
    object_height: float | None = None,
    min_reliable_floor_range: float = 0.4,
    nadir_dy: float = 0.92,
    wall_margin: float = 0.08,
) -> tuple[np.ndarray | None, str | None, str | None]:
    """
    Place a freestanding object on the floor using metric range along azimuth.

      1) If the classical floor hit is far enough from the camera, keep it
         (clamped inside the wall distance).
      2) Else if ``object_height`` is known, use angular-size → range.
      3) Else use a fraction of the layout wall distance along this azimuth.

    Returns ``(translation, method, error)``. ``error`` is None on success.
    """
    if depth is None:
        return None, None, "no_depth"

    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    camera_height = float(data["cameraHeight"])
    r_wall = wall_distance_from_depth(depth, uv, camera_height)
    if r_wall <= 1e-3:
        return None, None, "depth_wall_range_zero"

    az = azimuth_unit_xz(direction, uv)
    if az is None:
        return None, None, "no_azimuth"

    r_max = max(r_wall - wall_margin, min_reliable_floor_range * 0.5)
    dy = abs(float(direction[1]))

    r_floor = None
    try:
        p_floor = ray_intersect_floor(origin, direction, data)
        if check_floor_hit(p_floor, data) and closer_than_wall(p_floor, uv, depth, data):
            r_floor = horizontal_range(p_floor)
    except ValueError:
        pass

    if (
        r_floor is not None
        and r_floor >= min_reliable_floor_range
        and dy < nadir_dy
    ):
        range_m = min(r_floor, r_max)
        method = "floor"
    else:
        r_angular = None
        if object_height is not None and object_height > 1e-3:
            alpha = mask_angular_height(binary)
            if alpha > 1e-3:
                r_angular = range_from_angular_size(alpha, object_height)
        if r_angular is not None and np.isfinite(r_angular):
            range_m = float(np.clip(r_angular, min_reliable_floor_range * 0.5, r_max))
            method = "angular"
        else:
            frac = wall_fraction_from_elevation(direction)
            range_m = float(np.clip(r_wall * frac, min_reliable_floor_range * 0.5, r_max))
            method = "wall_fraction"

    p = shrink_range_into_footprint(origin, az, range_m, data)
    if p is None:
        return None, None, "outside_footprint"
    return p, method, None


def _try_translation(
    origin: np.ndarray,
    direction: np.ndarray,
    uv: np.ndarray,
    binary: np.ndarray,
    data: dict,
    depth: np.ndarray | None,
    object_height: float | None,
) -> tuple[np.ndarray | None, str | None, str | None, dict | None, str | None]:
    """
    Chronological translation attempts for one contact UV:

      1) depth-based floor range (if depth is present)
      2) legacy floor-plane ray
      3) nearest layout wall ray

    Returns ``(translation, surface, range_method, hit_wall, error)``.
    """
    fails: list[str] = []

    if depth is not None:
        t, method, reason = freestanding_floor_translation(
            origin, direction, uv, binary, data, depth, object_height=object_height
        )
        if t is not None:
            return t, "floor", method, None, None
        fails.append(reason)

    try:
        p_floor = ray_intersect_floor(origin, direction, data)
    except (ValueError, KeyError):
        p_floor = None
        fails.append("floor_ray_miss")
    else:
        if p_floor is not None and check_floor_hit(p_floor, data):
            return p_floor, "floor", "floor", None, None
        fails.append("outside_footprint" if p_floor is not None else "floor_ray_miss")

    try:
        p_wall, hit_wall = ray_intersect_wall(
            origin, direction, data, return_wall=True
        )
    except (KeyError, TypeError, ValueError):
        p_wall, hit_wall = None, None
    if p_wall is not None:
        return p_wall, "wall", None, hit_wall, None
    fails.append("no_wall_hit")
    return None, None, None, None, "; ".join(dict.fromkeys(fails))


def resolve_mesh_scale(
    binary: np.ndarray,
    translation: np.ndarray,
    surface: str | None,
    range_method: str | None,
    object_height: float | None = None,
    min_scale: float = 0.2,
    max_scale: float = 3.0,
) -> dict | None:
    """Choose a uniform mesh scale, or None when scale cannot be derived."""
    scale, _reason = _mesh_scale_or_reason(
        binary, translation, surface, range_method, object_height, min_scale, max_scale
    )
    return scale


def _mesh_scale_or_reason(
    binary: np.ndarray,
    translation: np.ndarray,
    surface: str | None,
    range_method: str | None,
    object_height: float | None = None,
    min_scale: float = 0.2,
    max_scale: float = 3.0,
) -> tuple[dict | None, str | None]:
    """Same as ``resolve_mesh_scale``, plus a failure token when scale is None."""
    if translation is None:
        return None, "missing_translation"

    base = {
        "object_height": float(object_height) if object_height is not None else None,
        "target_height": None,
        "range_m": None,
        "angular_height": None,
        "range_method": range_method,
    }

    if surface == "wall":
        return {**base, "factor": 1.0, "method": "sam3d"}, None
    if surface != "floor":
        return None, "unknown_surface"

    if object_height is None or object_height <= 1e-3:
        return None, "missing_object_height"

    range_m = horizontal_range(translation)
    alpha = mask_angular_height(binary)
    base["range_m"] = float(range_m)
    base["angular_height"] = float(alpha)

    if range_method in ("floor", "wall_fraction"):
        if alpha <= 1e-3:
            return None, "missing_angular_height"
        if range_m <= 1e-3:
            return None, "range_too_small"
        h_target = height_from_angular_size(alpha, range_m)
        factor = float(h_target / object_height)
        if not np.isfinite(factor) or factor <= 0.0:
            return None, "non_finite_factor"
        return {
            **base,
            "factor": float(np.clip(factor, min_scale, max_scale)),
            "method": "room_distance",
            "target_height": float(h_target),
        }, None

    if range_method == "angular":
        scale = {
            **base,
            "factor": 1.0,
            "method": "angular",
            "target_height": float(object_height),
        }
        if alpha > 1e-3:
            scale["angular_range_m"] = float(
                range_from_angular_size(alpha, object_height)
            )
        return scale, None

    return None, "no_scale_path"


def resolve_standing_pose(
    binary: np.ndarray,
    data: dict,
    origin: np.ndarray = None,
    num_samples: int = 5,
    depth: np.ndarray = None,
    object_height: float | None = None,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    str | None,
    dict | None,
    dict | None,
    str | None,
]:
    """
    Resolve translation, yaw rotation, and mesh scale for one mask.

    Chronological per contact UV:
      1) depth floor range (if depth) → 2) floor-plane ray → 3) wall ray
      4) yaw rotation (default upright if the wall heuristic fails)
      5) mesh scale (floor needs SAM3D height; wall uses native scale)

    Translation or scale failure skips the object (never a guessed origin/size).
    Rotation failure does not skip.

    Returns
      (translation, contact_uv, surface, rotation, scale, error)
    with ``error`` None on success, else ``unresolved: translation: …`` and/or
    ``unresolved: scale: …``. Pose fields are all None when error is set.
    """
    preflight = _pose_preflight(data)
    if preflight:
        return None, None, None, None, None, _unresolved_error(
            translation_detail=preflight
        )

    if origin is None:
        origin = np.zeros(3, dtype=np.float64)
    else:
        origin = np.asarray(origin, dtype=np.float64).reshape(3)

    if depth is not None:
        depth = np.asarray(depth, dtype=np.float64).reshape(-1)

    uvs = contact_uvs_from_mask(binary, num_samples=num_samples)
    if not uvs:
        return None, None, None, None, None, _unresolved_error(
            translation_detail="no_contact_uvs"
        )

    translation_detail = None
    scale_detail = None
    saw_translation = False

    for uv in uvs:
        translation, surface, range_method, hit_wall, t_fail = _try_translation(
            origin, direction=uv2equirectangular(uv), uv=uv, binary=binary,
            data=data, depth=depth, object_height=object_height,
        )
        if translation is None:
            translation_detail = t_fail
            continue

        saw_translation = True
        yaw, matrix, normal = resolve_standing_rotation(
            translation, data, wall=hit_wall
        )
        if yaw is not None and matrix is not None and normal is not None:
            rotation = _yaw_rotation_dict(yaw, normal, range_method=range_method)
        else:
            rotation = default_standing_rotation(translation, range_method)

        scale, s_fail = _mesh_scale_or_reason(
            binary, translation, surface, range_method, object_height=object_height
        )
        if scale is None:
            scale_detail = s_fail
            continue

        return translation, uv, surface, rotation, scale, None

    if saw_translation:
        error = _unresolved_error(scale_detail=scale_detail or "unknown")
    else:
        error = _unresolved_error(translation_detail=translation_detail or "no_in_room_hit")
    return None, None, None, None, None, error
