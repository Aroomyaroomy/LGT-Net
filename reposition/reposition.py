import os
from pathlib import Path

import numpy as np
from PIL import Image

from inference import preprocess
from utils.conversion import pixel2uv

from .lgt_utils import (
    azimuth_unit_xz,
    check_floor_hit,
    closer_than_wall,
    contact_uvs_from_mask,
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


def get_masks(mask_dir: Path) -> list[Path]:
    if not mask_dir.is_dir():
        raise NotADirectoryError(f"The {mask_dir} is not a directory")

    mask_paths = []
    for path in mask_dir.iterdir():
        if path.name.endswith('.png') and path.name.startswith('mask_'):
            mask_paths.append(path)

    return sorted(mask_paths, key=lambda x: int(x.stem.split('_')[1]))


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


def preprocess_masks(
    masks_paths: list[Path],
    do_manhattan: bool = True,
    vp_cache_path: str = None,
) -> list[np.ndarray]:
    """Return centroid contact UVs for each mask (legacy helper)."""
    contact_uvs = []
    for path in masks_paths:
        binary = load_aligned_binary_mask(
            path, do_manhattan=do_manhattan, vp_cache_path=vp_cache_path
        )
        if not binary.any():
            continue
        ys, xs = np.where(binary)
        u_centroid = float(xs.mean())
        v_centroid = float(ys.mean())
        contact_uv = pixel2uv(
            np.array([u_centroid, v_centroid], dtype=np.float64),
            w=binary.shape[1],
            h=binary.shape[0],
        )
        contact_uvs.append(contact_uv)
    return contact_uvs


def placements_from_mask_dir(
    mask_dir: Path | str,
    data: dict,
    depth: np.ndarray = None,
    do_manhattan: bool = True,
    vp_cache_path: str = None,
    num_samples: int = 5,
    sam3d_metadata: dict[int, dict] | Path | str | None = None,
    furniture_dir: Path | str | None = None,
) -> list[dict]:
    """
    For each mask_*.png under mask_dir, run resolve_standing_pose and
    return JSON-serializable placement records (translation + yaw rotation).

    Optional SAM3D ``metadata`` / ``furniture_dir`` supply per-object heights
    for angular-size freestanding ranging.
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
        object_height = None
        if idx is not None:
            object_height = object_height_from_sam3d(
                idx, metadata=metadata, furniture_dir=furniture_dir
            )

        translation, contact_uv, surface, rotation = resolve_standing_pose(
            binary,
            data,
            depth=depth,
            num_samples=num_samples,
            object_height=object_height,
        )
        if translation is not None:
            record['translation'] = np.asarray(translation, dtype=np.float64).reshape(3).tolist()
        if contact_uv is not None:
            record['contact_uv'] = np.asarray(contact_uv, dtype=np.float64).reshape(2).tolist()
        record['surface'] = surface
        record['rotation'] = rotation
        placements.append(record)

    return placements


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
) -> tuple[np.ndarray | None, str | None]:
    """
    Place a freestanding object on the floor using metric range along azimuth.

    Avoids the nadir singularity of pure ray–floor intersection:

      1) If the classical floor hit is far enough from the camera, keep it
         (clamped inside the wall distance).
      2) Else if ``object_height`` is known, use angular-size → range.
      3) Else use a fraction of the layout wall distance along this azimuth,
         with the fraction from ray elevation (near nadir → closer).

    Returns ``(translation, method)`` with method in
    ``{'floor', 'angular', 'wall_fraction'}`` or ``(None, None)``.
    """
    if depth is None:
        return None, None

    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    camera_height = float(data["cameraHeight"])
    r_wall = wall_distance_from_depth(depth, uv, camera_height)
    if r_wall <= 1e-3:
        return None, None

    az = azimuth_unit_xz(direction, uv)
    if az is None:
        return None, None

    r_max = max(r_wall - wall_margin, min_reliable_floor_range * 0.5)
    dy = abs(float(direction[1]))

    r_floor = None
    try:
        p_floor = ray_intersect_floor(origin, direction, data)
        if check_floor_hit(p_floor, data) and closer_than_wall(
            p_floor, uv, depth, data
        ):
            r_floor = horizontal_range(p_floor)
    except ValueError:
        p_floor = None

    method = None
    range_m = None

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
        return None, None
    return p, method


def resolve_standing_translation(
    binary: np.ndarray,
    data: dict,
    origin: np.ndarray = None,
    num_samples: int = 5,
    depth: np.ndarray = None,
    object_height: float | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    """
    Raycast from mask-centroid contact UVs onto the layout.

    For each contact UV:
      1) Freestanding floor placement (wall-fraction / angular / reliable floor)
      2) Else wall-mounted fallback: nearest layoutWalls segment hit along the same ray

    Returns (translation, contact_uv, surface) where surface is 'floor' or 'wall',
    or (None, None, None) if no valid hit is found.
    """
    translation, contact_uv, surface, _ = resolve_standing_pose(
        binary,
        data,
        origin=origin,
        num_samples=num_samples,
        depth=depth,
        object_height=object_height,
    )
    return translation, contact_uv, surface


def resolve_standing_pose(
    binary: np.ndarray,
    data: dict,
    origin: np.ndarray = None,
    num_samples: int = 5,
    depth: np.ndarray = None,
    object_height: float | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None, dict | None]:
    """
    Resolve translation and yaw rotation for one mask.

    Freestanding objects use metric range along the contact azimuth (fraction
    of wall distance and/or angular-size range) instead of a raw nadir-sensitive
    floor-plane hit. Wall-mounted objects still use layout wall raycasts.

    Rotation convention: mesh local +Z faces flush into the room along the
    inward normal of the nearest (or hit) wall — back against the wall.

    Returns
      (translation, contact_uv, surface, rotation)
    where rotation is
      {'yaw': float, 'matrix': 3x3 list, 'wall_normal': [nx, ny, nz]}
    or (None, None, None, None) on failure.
    """
    if origin is None:
        origin = np.zeros(3, dtype=np.float64)
    else:
        origin = np.asarray(origin, dtype=np.float64).reshape(3)

    if depth is not None:
        depth = np.asarray(depth, dtype=np.float64).reshape(-1)

    for uv in contact_uvs_from_mask(binary, num_samples=num_samples):
        direction = uv2equirectangular(uv)
        translation = None
        surface = None
        hit_wall = None
        range_method = None

        if depth is not None:
            translation, range_method = freestanding_floor_translation(
                origin,
                direction,
                uv,
                binary,
                data,
                depth,
                object_height=object_height,
            )
            if translation is not None:
                surface = "floor"

        if translation is None:
            # Legacy floor hit when depth is unavailable.
            try:
                p_floor = ray_intersect_floor(origin, direction, data)
            except ValueError:
                p_floor = None
            if p_floor is not None and check_floor_hit(p_floor, data):
                translation, surface, range_method = p_floor, "floor", "floor"

        if translation is None:
            p_wall, hit_wall = ray_intersect_wall(
                origin, direction, data, return_wall=True
            )
            if p_wall is not None:
                translation, surface = p_wall, "wall"

        if translation is None:
            continue

        yaw, matrix, normal = resolve_standing_rotation(
            translation, data, wall=hit_wall
        )
        rotation = None
        if yaw is not None and matrix is not None and normal is not None:
            rotation = {
                "yaw": float(yaw),
                "matrix": matrix.tolist(),
                "wall_normal": normal.tolist(),
            }
            if range_method is not None:
                rotation["range_method"] = range_method
        return translation, uv, surface, rotation

    return None, None, None, None
