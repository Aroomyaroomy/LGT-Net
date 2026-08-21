import json
import re
from pathlib import Path

import cv2
import numpy as np

from .lgt_utils import (
    _normalize,
    floor_polygon_from_layout,
    height_from_angular_size,
    horizontal_range,
    mask_angular_height,
    point_in_floor_polygon,
)


# SAM3D scene / create_3d_obj mesh share a Y-up convention (floor at negative Y).
SAM3D_UP = np.array([0.0, 1.0, 0.0], dtype=np.float64)
OBJ3D_UP = np.array([0.0, 1.0, 0.0], dtype=np.float64)


def uprightness(
    rotation: np.ndarray | dict | None,
    *,
    local_up: np.ndarray | None = None,
    world_up: np.ndarray | None = None,
) -> float:
    """
    Evaluation-only uprightness score in ``[-1, 1]``.

    Pure geometric metric — not used by placement / rendering:

      ``cos(tilt) = (R @ local_up) · world_up``

    with both vectors unit-normalized. Defaults use create_3d_obj / Open3D
    ``+Y`` as both mesh-local up and world up.

      1.0  → perfectly upright
      0.0  → tipped onto its side
     -1.0  → upside down

    ``rotation`` may be a 3×3 matrix, a placement ``rotation`` dict with a
    ``matrix`` field, or ``None`` (treated as identity).
    """
    r = _rotation_matrix_from_eval_input(rotation)
    local = _normalize(OBJ3D_UP if local_up is None else local_up)
    world = _normalize(OBJ3D_UP if world_up is None else world_up)
    tipped = _normalize(r @ local)
    return float(np.clip(np.dot(tipped, world), -1.0, 1.0))


def uprightness_metrics(
    rotation: np.ndarray | dict | None,
    *,
    local_up: np.ndarray | None = None,
    world_up: np.ndarray | None = None,
) -> dict:
    """
    Evaluation-only uprightness breakdown (does not affect placement).

    Returns
      cos:          ``uprightness`` in ``[-1, 1]``
      tilt_deg:     angle between object up and world up in ``[0, 180]``
      upright_01:   ``(cos + 1) / 2`` mapped to ``[0, 1]`` for averages
    """
    cos = uprightness(rotation, local_up=local_up, world_up=world_up)
    tilt_deg = float(np.degrees(np.arccos(cos)))
    return {
        "cos": cos,
        "tilt_deg": tilt_deg,
        "upright_01": float(0.5 * (cos + 1.0)),
    }


def _rotation_matrix_from_eval_input(
    rotation: np.ndarray | dict | None,
) -> np.ndarray:
    if rotation is None:
        return np.eye(3, dtype=np.float64)
    if isinstance(rotation, dict):
        matrix = rotation.get("matrix")
        if matrix is None:
            return np.eye(3, dtype=np.float64)
        rotation = matrix
    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    return r


def quat_xyzw_to_rotation_matrix(quat) -> np.ndarray:
    """Unit quaternion ``[x, y, z, w]`` → 3×3 rotation (local → world)."""
    q = np.asarray(quat, dtype=np.float64).reshape(-1)
    if q.shape[0] != 4:
        raise ValueError(f"Expected quaternion of length 4, got shape {q.shape}")
    x, y, z, w = q
    n = float(np.dot(q, q))
    if n < 1e-12:
        raise ValueError("Quaternion has near-zero norm")
    x, y, z, w = q * (1.0 / np.sqrt(n))
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def signed_angle_about_axis(a: np.ndarray, b: np.ndarray, axis: np.ndarray) -> float:
    """Signed angle from ``a`` to ``b`` about ``axis`` (radians)."""
    axis = _normalize(axis)
    a_h = a - np.dot(a, axis) * axis
    b_h = b - np.dot(b, axis) * axis
    a_h = _normalize(a_h)
    b_h = _normalize(b_h)
    return float(np.arctan2(np.dot(axis, np.cross(a_h, b_h)), np.dot(a_h, b_h)))


def rotation_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    """Right-handed rotation by ``angle`` radians about ``axis``."""
    axis = _normalize(axis)
    return cv2.Rodrigues(axis * float(angle))[0].astype(np.float64)


def _horizontal_dir(vec: np.ndarray, up: np.ndarray) -> np.ndarray | None:
    """Project ``vec`` onto the plane orthogonal to ``up``; None if degenerate."""
    up = _normalize(up)
    h = np.asarray(vec, dtype=np.float64).reshape(3) - np.dot(vec, up) * up
    n = float(np.linalg.norm(h))
    if n < 1e-8:
        return None
    return h / n


def sam3d_orientation_to_obj3d(quat) -> tuple[np.ndarray, dict]:
    """
    Build object rotation in create_3d_obj / Open3D coordinates from a SAM3D
    metadata quaternion (local → SAM3D world, xyzw).

    Individual ``mesh_N.glb`` exports are already gravity-aligned in local
    space (local +Y ≈ up). Applying the full SAM3D matrix — or an automatic
    upside-down flip derived from that matrix — tips/inverts them because the
    quaternion's local-up axis often disagrees with the exported GLB frame.

    Hybrid used here:
      - Keep mesh local +Y as up (reconstructor baked uprightness into the GLB)
      - Take heading (yaw) only from the horizontally projected SAM3D forward
      - LGT still owns translation (floor/wall hit)

    Returns ``(R_3x3, info)``.
    """
    r_sam = quat_xyzw_to_rotation_matrix(quat)

    fwd_local = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    sam_fwd_h = _horizontal_dir(r_sam @ fwd_local, SAM3D_UP)
    if sam_fwd_h is None:
        sam_fwd_h = _horizontal_dir(
            r_sam @ np.array([1.0, 0.0, 0.0], dtype=np.float64), SAM3D_UP
        )

    # Unrotated mesh forward is local +Z on the horizontal plane.
    after_fwd_h = _horizontal_dir(fwd_local, OBJ3D_UP)
    if sam_fwd_h is None or after_fwd_h is None:
        yaw = 0.0
        r = np.eye(3, dtype=np.float64)
    else:
        yaw = signed_angle_about_axis(after_fwd_h, sam_fwd_h, OBJ3D_UP)
        r = rotation_about_axis(OBJ3D_UP, yaw)

    info = {
        "yaw": float(yaw),
        "source": "sam3d",
    }
    return r, info


def load_sam3d_metadata(metadata_path: Path | str) -> dict[int, dict]:
    """
    Load ``metadata.json`` from SAM3D into ``{object_index: record}``.

    Each record keeps ``rotation`` as a length-4 xyzw quaternion list when present.
    """
    path = Path(metadata_path)
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"SAM3D metadata must be a list, got {type(payload).__name__}")

    out: dict[int, dict] = {}
    for i, item in enumerate(payload):
        if not isinstance(item, dict):
            continue
        idx = int(item.get("object_index", i))
        rot = item.get("rotation")
        quat = None
        if isinstance(rot, (list, tuple)) and rot:
            q = rot[0] if isinstance(rot[0], (list, tuple)) else rot
            quat = [float(c) for c in q]
        out[idx] = {**item, "rotation_xyzw": quat}
    return out


def mask_index_from_name(mask_name: str) -> int | None:
    """Parse ``mask_12.png`` → ``12``."""
    match = re.fullmatch(r"mask_(\d+)\.png", Path(mask_name).name, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def object_aabb_extents_from_sam3d(
    object_index: int,
    metadata: dict[int, dict] | None = None,
    furniture_dir: Path | str | None = None,
) -> np.ndarray | None:
    """Native GLB AABB extents ``(sx, sy, sz)``, else SAM3D scale as a unit cube."""
    if furniture_dir is not None:
        mesh_path = Path(furniture_dir) / f"mesh_{object_index}.glb"
        if mesh_path.is_file():
            try:
                import open3d as o3d

                mesh = o3d.io.read_triangle_mesh(str(mesh_path))
                if not mesh.is_empty():
                    extents = np.asarray(
                        mesh.get_axis_aligned_bounding_box().get_extent(),
                        dtype=np.float64,
                    ).reshape(3)
                    if float(extents[1]) > 1e-4:
                        return extents
            except Exception:
                pass

    if metadata is not None and object_index in metadata:
        scale = metadata[object_index].get("scale")
        if scale is not None:
            s = scale[0] if isinstance(scale, (list, tuple)) and scale else scale
            if isinstance(s, (list, tuple)):
                s = s[0]
            s = float(s)
            if s > 1e-4:
                return np.array([s, s, s], dtype=np.float64)
    return None


def object_height_from_sam3d(
    object_index: int,
    metadata: dict[int, dict] | None = None,
    furniture_dir: Path | str | None = None,
) -> float | None:
    """
    Metric object height for angular-size ranging.

    Prefers the Y extent of ``mesh_{i}.glb`` when present (export is already
    scaled). Falls back to SAM3D ``scale`` as a unit-height proxy.
    """
    extents = object_aabb_extents_from_sam3d(
        object_index, metadata=metadata, furniture_dir=furniture_dir
    )
    if extents is None:
        return None
    h = float(extents[1])
    return h if h > 1e-4 else None


def merge_sam3d_orientations_into_placements(
    placements: list[dict],
    metadata: dict[int, dict] | Path | str | None,
) -> list[dict]:
    """
    Replace wall-heuristic rotations with SAM3D upright+yaw orientations.

    Translations are left unchanged (LGT floor/wall hits). When metadata is a
    path it is loaded via ``load_sam3d_metadata``. Objects without a quaternion
    keep their existing rotation.
    """
    if metadata is None:
        return placements
    if not isinstance(metadata, dict):
        metadata = load_sam3d_metadata(metadata)

    merged = []
    for placement in placements:
        record = dict(placement)
        idx = mask_index_from_name(record.get("mask") or "")
        meta = metadata.get(idx) if idx is not None else None
        quat = None if meta is None else meta.get("rotation_xyzw")
        if quat is None:
            merged.append(record)
            continue
        try:
            r_obj, info = sam3d_orientation_to_obj3d(quat)
        except ValueError:
            merged.append(record)
            continue
        record["rotation"] = {
            "yaw": info["yaw"],
            "matrix": r_obj.tolist(),
            "source": "sam3d",
            "frame": "obj3d",
        }
        merged.append(record)
    return merged


_DEFAULT_SCALE_CLIP = (0.2, 3.0)


def _metadata_from_arg(
    metadata: dict[int, dict] | Path | str | None,
) -> dict[int, dict] | None:
    if metadata is None:
        return None
    if isinstance(metadata, dict):
        return metadata
    return load_sam3d_metadata(metadata)


def _angular_height_from_mask_file(
    mask_dir: Path | str | None, mask_name: str | None
) -> float | None:
    if not mask_dir or not mask_name:
        return None
    path = Path(mask_dir) / Path(mask_name).name
    if not path.is_file():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    alpha = float(mask_angular_height(img > 0))
    return alpha if alpha > 1e-3 else None


def _placement_photometric_height(placement: dict, mask_dir: Path | str | None):
    """``(h_target, alpha, range_m)`` from stored scale, else mask α and translation."""
    scale = placement.get("scale") if isinstance(placement.get("scale"), dict) else {}
    translation = placement.get("translation")

    range_m = scale.get("range_m")
    if range_m is None and translation is not None:
        try:
            range_m = horizontal_range(translation)
        except Exception:
            range_m = None
    try:
        range_m = float(range_m) if range_m is not None else None
    except (TypeError, ValueError):
        range_m = None
    if range_m is not None and range_m <= 1e-3:
        range_m = None

    alpha = scale.get("angular_height")
    try:
        alpha = float(alpha) if alpha is not None else None
    except (TypeError, ValueError):
        alpha = None
    if alpha is None or alpha <= 1e-3:
        alpha = _angular_height_from_mask_file(mask_dir, placement.get("mask"))

    h_target = scale.get("target_height")
    try:
        h_target = float(h_target) if h_target is not None else None
    except (TypeError, ValueError):
        h_target = None
    if (h_target is None or h_target <= 1e-4) and alpha is not None and range_m is not None:
        h_target = float(height_from_angular_size(alpha, range_m))
    if h_target is not None and h_target <= 1e-4:
        h_target = None

    return h_target, alpha, range_m


def _xz_half_extent(native_xz: float, factor: float) -> float:
    return 0.5 * float(factor) * float(native_xz)


def _aabb_inside_footprint(translation, half_xz: float, polygon: np.ndarray) -> bool:
    cx, cz = float(translation[0]), float(translation[2])
    poly = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    h = float(half_xz)
    for dx, dz in ((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)):
        dist = float(cv2.pointPolygonTest(poly, (cx + dx * h, cz + dz * h), True))
        if dist < 0.0:
            return False
    return True


def clip_scale_factor_to_footprint(
    translation,
    factor: float,
    native_xz: float,
    layout: dict | None,
    *,
    min_scale: float = _DEFAULT_SCALE_CLIP[0],
    surface: str | None = None,
) -> tuple[float, bool]:
    """Shrink ``factor`` until the XZ AABB lies in the layout floor polygon.

    Does not move the translation. Wall-mounted items and placements whose
    centre is already outside the footprint are left unchanged.
    """
    factor = float(factor)
    if layout is None or surface == "wall":
        return factor, False
    if native_xz is None or float(native_xz) <= 1e-4 or not np.isfinite(factor):
        return factor, False
    try:
        polygon = floor_polygon_from_layout(layout)
    except (KeyError, TypeError, ValueError):
        return factor, False
    if not point_in_floor_polygon(translation, polygon):
        return factor, False

    lo = float(min_scale)
    if _aabb_inside_footprint(translation, _xz_half_extent(native_xz, factor), polygon):
        return factor, False
    if not _aabb_inside_footprint(translation, _xz_half_extent(native_xz, lo), polygon):
        return lo, True

    hi = factor
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        if _aabb_inside_footprint(translation, _xz_half_extent(native_xz, mid), polygon):
            lo = mid
        else:
            hi = mid
    return float(lo), True


def merge_sam3d_scale_into_placements(
    placements: list[dict],
    furniture_dir: Path | str | None = None,
    metadata: dict[int, dict] | Path | str | None = None,
    mask_dir: Path | str | None = None,
    min_scale: float = _DEFAULT_SCALE_CLIP[0],
    max_scale: float = _DEFAULT_SCALE_CLIP[1],
    layout: dict | None = None,
    boxes=None,
    label_map: dict | None = None,
) -> list[dict]:
    """
    Scale each mesh from a class-typical height, then clip to the floor polygon.

    ``H_target`` is the typical standing height for the DINO label (chair 0.85 m,
    table 0.75 m, bed 0.55 m, sofa 0.85 m, else 0.80 m). ``factor`` is
    ``clip(H_target / H_mesh)`` where ``H_mesh`` is the exported GLB Y-extent,
    else SAM3D metadata scale as a unit-height proxy.

    When ``layout`` is provided, floor objects are then shrunk uniformly until
    their XZ AABB lies inside the footprint. Photometric α / R are stored only
    as diagnostics and do not drive size.
    """
    if not placements:
        return placements

    from .quality_control import (
        DEFAULT_HALF_XZ_FRAC,
        DEFAULT_OBJECT_HEIGHT_M,
        label_map_from_dino_boxes,
        typical_height_for_label,
    )

    meta = _metadata_from_arg(metadata)
    furn = Path(furniture_dir) if furniture_dir is not None else None
    lo, hi = float(min_scale), float(max_scale)
    if label_map is None and boxes is not None:
        label_map = label_map_from_dino_boxes(boxes)

    merged = []
    for placement in placements:
        record = dict(placement)
        if record.get("translation") is None:
            merged.append(record)
            continue

        idx = mask_index_from_name(record.get("mask") or "")
        if label_map and idx is not None and record.get("label") is None:
            info = label_map.get(idx)
            if info:
                record["label"] = info.get("label")
                if record.get("dino") is None:
                    record["dino"] = {
                        "label": info.get("label"),
                        "raw_label": info.get("raw_label"),
                        "score": info.get("score"),
                    }

        extents = (
            object_aabb_extents_from_sam3d(idx, metadata=meta, furniture_dir=furn)
            if idx is not None
            else None
        )
        h_mesh = float(extents[1]) if extents is not None else None
        native_xz = (
            float(np.hypot(float(extents[0]), float(extents[2])))
            if extents is not None
            else None
        )
        _, alpha, range_m = _placement_photometric_height(record, mask_dir)

        typical = typical_height_for_label(record.get("label"))
        if typical is None:
            typical = float(DEFAULT_OBJECT_HEIGHT_M)
        h_target = float(typical)

        prev = record.get("scale") if isinstance(record.get("scale"), dict) else {}
        scale = dict(prev)
        if alpha is not None:
            scale["angular_height"] = float(alpha)
        if range_m is not None:
            scale["range_m"] = float(range_m)
        scale["typical_height"] = float(typical)

        if h_mesh is not None and h_target > 1e-4:
            factor = float(np.clip(h_target / h_mesh, lo, hi))
            if native_xz is None or native_xz <= 1e-4:
                native_xz = 2.0 * float(DEFAULT_HALF_XZ_FRAC) * h_mesh
            clipped_factor, clipped = clip_scale_factor_to_footprint(
                record["translation"],
                factor,
                native_xz,
                layout,
                min_scale=lo,
                surface=record.get("surface"),
            )
            scale["object_height"] = float(h_mesh)
            scale["native_xz"] = float(native_xz)
            scale["factor"] = float(clipped_factor)
            scale["target_height"] = float(clipped_factor * h_mesh)
            scale["method"] = (
                "class_typical_footprint" if clipped else "class_typical"
            )
            record["scale"] = scale
        elif scale:
            scale["target_height"] = float(h_target)
            record["scale"] = scale

        merged.append(record)
    return merged
