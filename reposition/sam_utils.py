import json
import re
from pathlib import Path

import cv2
import numpy as np

from .lgt_utils import _normalize


# SAM3D scene / create_3d_obj mesh share a Y-up convention (floor at negative Y).
SAM3D_UP = np.array([0.0, 1.0, 0.0], dtype=np.float64)
OBJ3D_UP = np.array([0.0, 1.0, 0.0], dtype=np.float64)


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


def rotation_aligning_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation ``R`` such that ``R @ a`` is parallel to ``b``."""
    a = _normalize(a)
    b = _normalize(b)
    cross = np.cross(a, b)
    dot = float(np.dot(a, b))
    if dot < -1.0 + 1e-9:
        # 180°: pick any axis orthogonal to a
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        axis = _normalize(axis)
        return cv2.Rodrigues(axis * np.pi)[0].astype(np.float64)
    if np.linalg.norm(cross) < 1e-12:
        return np.eye(3, dtype=np.float64)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + skew + skew @ skew * ((1.0 - dot) / (np.dot(cross, cross)))


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
    if furniture_dir is not None:
        mesh_path = Path(furniture_dir) / f"mesh_{object_index}.glb"
        if mesh_path.is_file():
            try:
                import open3d as o3d

                mesh = o3d.io.read_triangle_mesh(str(mesh_path))
                if not mesh.is_empty():
                    extents = mesh.get_axis_aligned_bounding_box().get_extent()
                    h = float(extents[1])
                    if h > 1e-4:
                        return h
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
                return s
    return None


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
