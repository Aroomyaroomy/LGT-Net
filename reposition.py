import os

import cv2
import numpy as np

from pathlib import Path
from PIL import Image
from utils.conversion import pixel2uv, uv2lonlat, lonlat2xyz
from inference import preprocess


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
) -> list[dict]:
    """
    For each mask_*.png under mask_dir, run resolve_standing_pose and
    return JSON-serializable placement records (translation + yaw rotation).
    """
    mask_dir = Path(mask_dir)
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

        translation, contact_uv, surface, rotation = resolve_standing_pose(
            binary,
            data,
            depth=depth,
            num_samples=num_samples,
        )
        if translation is not None:
            record['translation'] = np.asarray(translation, dtype=np.float64).reshape(3).tolist()
        if contact_uv is not None:
            record['contact_uv'] = np.asarray(contact_uv, dtype=np.float64).reshape(2).tolist()
        record['surface'] = surface
        record['rotation'] = rotation
        placements.append(record)

    return placements


def to_json_frame(xyz: np.ndarray) -> np.ndarray:
    """
    Map an internal LGT xyz vector into the xyz2json / layoutPoints frame.
    Matches utils.writer.xyz2json: 180 deg yaw then flip X. Y is unchanged, so
    the floor remains y = cameraHeight.
    """
    xyz = np.asarray(xyz, dtype=np.float64).reshape(3)
    r_180 = cv2.Rodrigues(np.array([0, -np.pi, 0], np.float32))[0]
    out = np.dot(r_180, xyz)
    out[0] *= -1
    return out


def uv2equirectangular(uv: np.ndarray) -> np.ndarray:
    """
    Map panorama UV to a unit ray direction in the xyz2json coordinate frame.
    Camera / ray origin is (0, 0, 0).
    """
    lonlat = uv2lonlat(np.asarray(uv, dtype=np.float64))
    direction = lonlat2xyz(lonlat)
    return to_json_frame(direction)


def solve_t(y: float, origin: np.ndarray, direction: np.ndarray) -> float:
    """
    Solve origin + t * direction for intersection with the horizontal plane at height y.
    Rejects rays parallel to the plane or with non-positive t (wrong direction).
    """
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    dy = direction[1]
    if abs(dy) < 1e-12:
        raise ValueError("Ray is parallel to the floor plane")
    t = (y - origin[1]) / dy
    if t <= 0:
        raise ValueError(f"Ray does not hit the floor in the forward direction (t={t})")
    return float(t)


def solve_t_plane(
    origin: np.ndarray,
    direction: np.ndarray,
    plane: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Solve origin + t * direction against plane n·x + d = 0.
    plane is [nx, ny, nz, d] as in layoutWalls[].planeEquation.
    """
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    plane = np.asarray(plane, dtype=np.float64).reshape(4)
    normal = plane[:3]
    d = plane[3]
    denom = float(np.dot(normal, direction))
    if abs(denom) < eps:
        raise ValueError("Ray is parallel to the plane")
    t = -float(np.dot(normal, origin) + d) / denom
    if t <= 0:
        raise ValueError(f"Ray does not hit the plane in the forward direction (t={t})")
    return float(t)


def ray_intersect_floor(origin: np.ndarray, direction: np.ndarray, data: dict) -> np.ndarray:
    """
    Intersect a ray with the floor plane in xyz2json coordinates.

    Floor plane: y = data['cameraHeight'] (default 1.6m).
    origin and direction must already be in that JSON layout frame
    (use uv2equirectangular for mask contact UVs).

    Returns a candidate translation point where we think the object is located.
    """
    camera_height = float(data['cameraHeight'])
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    t = solve_t(camera_height, origin, direction)
    return origin + t * direction


def _point_on_segment_xz(
    point: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    tol: float = 5e-2,
) -> bool:
    """True if point's XZ lies on segment a--b (JSON layoutPoints), within tol meters."""
    p = np.asarray(point, dtype=np.float64).reshape(3)[[0, 2]]
    a_xz = np.asarray(a, dtype=np.float64).reshape(3)[[0, 2]]
    b_xz = np.asarray(b, dtype=np.float64).reshape(3)[[0, 2]]
    ab = b_xz - a_xz
    ap = p - a_xz
    ab_len2 = float(np.dot(ab, ab))
    if ab_len2 < 1e-12:
        return float(np.linalg.norm(ap)) <= tol
    t = float(np.dot(ap, ab) / ab_len2)
    if t < -1e-3 or t > 1.0 + 1e-3:
        return False
    t = float(np.clip(t, 0.0, 1.0))
    return float(np.linalg.norm(p - (a_xz + t * ab))) <= tol


def _wall_y_bounds(data: dict) -> tuple[float, float]:
    """Valid wall height range in JSON frame (ceiling negative-y, floor +cameraHeight)."""
    camera_height = float(data['cameraHeight'])
    ceiling = float(data.get('cameraCeilingHeight', data['layoutHeight'] - camera_height))
    return -ceiling, camera_height


def ray_intersect_wall(
    origin: np.ndarray,
    direction: np.ndarray,
    data: dict,
    segment_tol: float = 5e-2,
    return_wall: bool = False,
) -> np.ndarray | None | tuple[np.ndarray, dict]:
    """
    Intersect a ray with layoutWalls in xyz2json coordinates.
    Returns the nearest forward hit that lies on a wall segment and within room height,
    or None if no valid wall hit exists.

    If return_wall=True, returns (hit, wall_dict) instead of just hit.
    """
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    points = data['layoutPoints']['points']
    walls = data['layoutWalls']['walls']
    y_lo, y_hi = _wall_y_bounds(data)

    best_t = None
    best_hit = None
    best_wall = None
    for wall in walls:
        try:
            t = solve_t_plane(origin, direction, wall['planeEquation'])
        except ValueError:
            continue
        hit = origin + t * direction
        if hit[1] < y_lo - segment_tol or hit[1] > y_hi + segment_tol:
            continue
        i0, i1 = wall['pointsIdx']
        a = points[i0]['xyz']
        b = points[i1]['xyz']
        if not _point_on_segment_xz(hit, a, b, tol=segment_tol):
            continue
        if best_t is None or t < best_t:
            best_t = t
            best_hit = hit
            best_wall = wall

    if return_wall:
        if best_hit is None:
            return None, None
        return best_hit, best_wall
    return best_hit


def patch_index_from_uv(uv: np.ndarray, patch_num: int = 256) -> int:
    """Map panorama u in [0, 1] to a 256-bin LGT depth patch index."""
    u = float(np.asarray(uv, dtype=np.float64).reshape(-1)[0])
    return int(np.round(u * patch_num - 0.5)) % patch_num


def horizontal_range(point: np.ndarray) -> float:
    """Camera-centered horizontal distance (XZ) in JSON / metric layout units."""
    p = np.asarray(point, dtype=np.float64).reshape(3)
    return float(np.linalg.norm(p[[0, 2]]))


def wall_distance_from_depth(
    depth: np.ndarray,
    uv: np.ndarray,
    camera_height: float = 1.6,
) -> float:
    """
    Metric distance to the floor-wall boundary along the contact azimuth.
    Model depth is in plan_y=1 units; scale by camera_height to match xyz2json.
    """
    depth = np.asarray(depth, dtype=np.float64).reshape(-1)
    idx = patch_index_from_uv(uv, patch_num=len(depth))
    return float(np.abs(depth[idx]) * camera_height)


def closer_than_wall(
    point: np.ndarray,
    uv: np.ndarray,
    depth: np.ndarray,
    data: dict,
    margin: float = 5e-2,
) -> bool:
    """
    Step 5: standing hit must be closer to the camera than the wall along that azimuth.
    """
    camera_height = float(data['cameraHeight'])
    return horizontal_range(point) <= wall_distance_from_depth(depth, uv, camera_height) + margin


def floor_polygon_from_layout(data: dict) -> np.ndarray:
    """
    Build the room footprint polygon in JSON coordinates from layoutPoints.
    Returns (N, 2) array of (x, z); y is height and is dropped.
    """
    points = data['layoutPoints']['points']
    if not points:
        raise ValueError("layoutPoints is empty")
    xyz = np.array([p['xyz'] for p in points], dtype=np.float64)
    return xyz[:, [0, 2]]


def point_in_floor_polygon(p_xz: np.ndarray, polygon: np.ndarray) -> bool:
    """
    True if the horizontal point lies inside (or on the boundary of) the floor polygon.

    p_xz: (2,) as (x, z), or (3,) xyz in which case y is dropped.
    polygon: (N, 2) footprint from floor_polygon_from_layout.
    """
    p = np.asarray(p_xz, dtype=np.float64).reshape(-1)
    if p.shape[0] == 3:
        p = p[[0, 2]]
    elif p.shape[0] != 2:
        raise ValueError(f"p_xz must have shape (2,) or (3,), got {p.shape}")

    poly = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    if len(poly) < 3:
        raise ValueError("floor polygon needs at least 3 vertices")

    # cv2.pointPolygonTest: +1 inside, 0 on edge, -1 outside
    return cv2.pointPolygonTest(poly.astype(np.float32), (float(p[0]), float(p[1])), False) >= 0


def check_floor_hit(p_floor: np.ndarray, data: dict) -> bool:
    """
    Step 4: keep the floor-ray hit only if it lies inside the room footprint.
    """
    polygon = floor_polygon_from_layout(data)
    return point_in_floor_polygon(p_floor, polygon)


def contact_uvs_from_mask(binary: np.ndarray, num_samples: int = 5) -> list[np.ndarray]:
    """
    Sample contact UVs near the mask centroid (not the bottom row).

    Bottom-row contacts often look nearly straight down on panos and pull
    floor hits under the camera. Centroid rays stay farther from nadir.
    Samples span the mask width at the centroid row; center first, then outward.
    """
    binary = np.asarray(binary)
    if binary.ndim != 2:
        raise ValueError("binary mask must be 2D")

    ys, xs = np.where(binary)
    if len(xs) == 0:
        return []

    h, w = binary.shape
    u_centroid = float(xs.mean())
    v_centroid = float(ys.mean())

    # Horizontal span of the mask on the centroid row (fallback: full mask width)
    row = int(np.clip(round(v_centroid), 0, h - 1))
    row_xs = xs[ys == row]
    if len(row_xs) == 0:
        u_min, u_max = float(xs.min()), float(xs.max())
    else:
        u_min, u_max = float(row_xs.min()), float(row_xs.max())

    if num_samples <= 1 or u_min == u_max:
        us = [u_centroid]
    else:
        us = list(np.linspace(u_min, u_max, num_samples))
        us.sort(key=lambda u: abs(u - u_centroid))

    return [
        pixel2uv(np.array([u, v_centroid], dtype=np.float64), w=w, h=h)
        for u in us
    ]


# Back-compat alias
bottom_contact_uvs_from_mask = contact_uvs_from_mask


def horizontal_wall_normal(normal: np.ndarray) -> np.ndarray:
    """Project a wall normal onto the XZ plane and renormalize."""
    n = np.asarray(normal, dtype=np.float64).reshape(3).copy()
    n[1] = 0.0
    norm = float(np.linalg.norm(n))
    if norm < 1e-12:
        raise ValueError("Wall normal has no horizontal component")
    return n / norm


def inward_wall_normal(
    normal: np.ndarray,
    point: np.ndarray,
    room_center: np.ndarray = None,
) -> np.ndarray:
    """
    Return the horizontal wall normal that points into the room.

    layoutWalls normals may face either inward or outward; we flip so the
    normal points toward room_center (default: camera / JSON origin).
    """
    n = horizontal_wall_normal(normal)
    if room_center is None:
        room_center = np.zeros(3, dtype=np.float64)
    else:
        room_center = np.asarray(room_center, dtype=np.float64).reshape(3)
    point = np.asarray(point, dtype=np.float64).reshape(3)
    to_center = room_center - point
    to_center[1] = 0.0
    if float(np.dot(n, to_center)) < 0.0:
        n = -n
    return n


def nearest_wall(
    point: np.ndarray,
    data: dict,
    segment_tol: float = 0.25,
) -> tuple[dict | None, float, np.ndarray | None]:
    """
    Find the layout wall closest to point (JSON frame).

    Prefers walls whose floor segment is near the point's XZ; falls back to
    absolute point-to-plane distance.

    Returns (wall, signed_plane_distance, inward_horizontal_normal).
    """
    point = np.asarray(point, dtype=np.float64).reshape(3)
    points = data['layoutPoints']['points']
    walls = data['layoutWalls']['walls']

    best = None
    best_score = None
    for wall in walls:
        plane = np.asarray(wall['planeEquation'], dtype=np.float64).reshape(4)
        normal = plane[:3]
        n_len = float(np.linalg.norm(normal))
        if n_len < 1e-12:
            continue
        dist = float(np.dot(normal, point) + plane[3]) / n_len
        i0, i1 = wall['pointsIdx']
        on_segment = _point_on_segment_xz(
            point, points[i0]['xyz'], points[i1]['xyz'], tol=segment_tol
        )
        # Prefer walls the point actually sits against; otherwise use |dist|
        score = (0 if on_segment else 1, abs(dist))
        if best_score is None or score < best_score:
            best_score = score
            try:
                inward = inward_wall_normal(normal, point)
            except ValueError:
                continue
            best = (wall, dist, inward)

    if best is None:
        return None, float('inf'), None
    return best


def yaw_from_wall_normal(normal: np.ndarray) -> float:
    """
    Yaw (radians about +Y) so mesh local +Z faces flush into the room.

    Convention:
      - mesh local +Z = front
      - normal = inward horizontal wall normal (into the room)
      - flush against wall => front aligns with inward normal (back to the wall)

    R_y(yaw) @ (0,0,1) = (sin(yaw), 0, cos(yaw)), so yaw = atan2(nx, nz).
    """
    n = horizontal_wall_normal(normal)
    return float(np.arctan2(n[0], n[2]))


def rotation_matrix_yaw(yaw: float) -> np.ndarray:
    """Right-handed yaw rotation about +Y (JSON / layout up axis)."""
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
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


def resolve_standing_translation(
    binary: np.ndarray,
    data: dict,
    origin: np.ndarray = None,
    num_samples: int = 5,
    depth: np.ndarray = None,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    """
    Raycast from mask-centroid contact UVs onto the layout.

    For each contact UV:
      1) Floor hit inside footprint (optional Step-5 closer-than-wall check if depth given)
      2) Else wall-mounted fallback: nearest layoutWalls segment hit along the same ray

    Returns (translation, contact_uv, surface) where surface is 'floor' or 'wall',
    or (None, None, None) if no valid hit is found.
    """
    translation, contact_uv, surface, _ = resolve_standing_pose(
        binary, data, origin=origin, num_samples=num_samples, depth=depth
    )
    return translation, contact_uv, surface


def resolve_standing_pose(
    binary: np.ndarray,
    data: dict,
    origin: np.ndarray = None,
    num_samples: int = 5,
    depth: np.ndarray = None,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None, dict | None]:
    """
    Resolve translation and yaw rotation for one mask.

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

        try:
            p_floor = ray_intersect_floor(origin, direction, data)
        except ValueError:
            p_floor = None

        if p_floor is not None and check_floor_hit(p_floor, data):
            if depth is None or closer_than_wall(p_floor, uv, depth, data):
                translation, surface = p_floor, 'floor'

        if translation is None:
            p_wall, hit_wall = ray_intersect_wall(
                origin, direction, data, return_wall=True
            )
            if p_wall is not None:
                translation, surface = p_wall, 'wall'

        if translation is None:
            continue

        yaw, matrix, normal = resolve_standing_rotation(
            translation, data, wall=hit_wall
        )
        rotation = None
        if yaw is not None and matrix is not None and normal is not None:
            rotation = {
                'yaw': float(yaw),
                'matrix': matrix.tolist(),
                'wall_normal': normal.tolist(),
            }
        return translation, uv, surface, rotation

    return None, None, None, None
