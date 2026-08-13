import cv2
import numpy as np

from utils.conversion import pixel2uv, uv2lonlat, lonlat2xyz


def to_json_frame(xyz: np.ndarray) -> np.ndarray:
    """
    Map an internal LGT xyz vector into the xyz2json / layoutPoints frame.

    Equivalent to utils.writer.xyz2json (180° yaw then flip X): ``(x, y, -z)``.
    Y is unchanged, so the floor remains y = cameraHeight.
    """
    xyz = np.asarray(xyz, dtype=np.float64).reshape(3)
    return np.array([xyz[0], xyz[1], -xyz[2]], dtype=np.float64)


# Linear map from xyz2json / layoutPoints coordinates into visualization.obj3d
# create_3d_obj mesh coordinates.
#
# LGT spherical (lonlat2xyz): p = (x, y, z) with +Y toward the floor.
# xyz2json frame:              j = (x, y, -z)
# create_3d_obj mesh:          m = (z, -y, x)  i.e. stack([.., -sin(lat), ..])
# Combined j -> m:             m = (-j_z, -j_y, j_x)
#
# Floor at j_y = +cameraHeight therefore becomes m_y = -cameraHeight, matching
# the textured layout mesh (floor below camera, ceiling above).
JSON_TO_OBJ3D = np.array(
    [
        [0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)


def json_frame_to_obj3d(xyz: np.ndarray) -> np.ndarray:
    """
    Convert a point from the xyz2json / placement frame into create_3d_obj
    mesh coordinates used by the exported room ``*_3d.obj``.

    ``m = (-z, -y, x)`` for json ``(x, y, z)``.
    """
    xyz = np.asarray(xyz, dtype=np.float64).reshape(3)
    return JSON_TO_OBJ3D @ xyz


def json_rotation_to_obj3d(rotation: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation expressed in the xyz2json frame into create_3d_obj
    mesh coordinates: ``R_mesh = T @ R_json @ T.T``.
    """
    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    return JSON_TO_OBJ3D @ r @ JSON_TO_OBJ3D.T


def placement_to_obj3d_frame(
    translation: np.ndarray,
    rotation: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Convert placement translation (and optional rotation matrix) from the
    xyz2json frame into the create_3d_obj frame for mesh visualization.
    """
    t_mesh = json_frame_to_obj3d(translation)
    r_mesh = json_rotation_to_obj3d(rotation) if rotation is not None else None
    return t_mesh, r_mesh


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n < eps:
        raise ValueError("Cannot normalize near-zero vector")
    return v / n


def uv2equirectangular(uv: np.ndarray) -> np.ndarray:
    """
    Map panorama UV to a unit ray direction in the xyz2json coordinate frame.
    Camera / ray origin is (0, 0, 0).
    """
    lonlat = uv2lonlat(np.asarray(uv, dtype=np.float64))
    direction = lonlat2xyz(lonlat)
    return to_json_frame(direction)


def solve_t(y: float, origin: np.ndarray, direction: np.ndarray) -> float:
    """Intersect a ray with the horizontal plane at height y (JSON +Y)."""
    return solve_t_plane(origin, direction, np.array([0.0, 1.0, 0.0, -float(y)]))


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
    """Intersect a ray with the floor plane y = cameraHeight in xyz2json coordinates."""
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    t = solve_t(float(data["cameraHeight"]), origin, direction)
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


def layout_y_bounds(data: dict) -> tuple[float, float]:
    """Return ``(floor_y, ceiling_y)`` in the JSON frame (+Y toward the floor)."""
    floor_y = float(data["cameraHeight"])
    ceiling = float(data.get("cameraCeilingHeight", data.get("layoutHeight", 2.6) - floor_y))
    return floor_y, -float(ceiling)


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
    floor_y, ceiling_y = layout_y_bounds(data)

    best_t = None
    best_hit = None
    best_wall = None
    for wall in walls:
        try:
            t = solve_t_plane(origin, direction, wall['planeEquation'])
        except ValueError:
            continue
        hit = origin + t * direction
        if hit[1] < ceiling_y - segment_tol or hit[1] > floor_y + segment_tol:
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
    if p.size >= 3:
        p = p[[0, 2]]
    poly = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    if len(poly) < 3:
        raise ValueError("floor polygon needs at least 3 vertices")
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


def mask_angular_height(binary: np.ndarray) -> float:
    """Vertical angular span of a mask in an equirectangular pano (radians)."""
    binary = np.asarray(binary)
    ys, _ = np.where(binary)
    if len(ys) == 0:
        return 0.0
    h = binary.shape[0]
    v0 = (float(ys.min()) + 0.5) / h
    v1 = (float(ys.max()) + 0.5) / h
    return abs(v1 - v0) * np.pi


def range_from_angular_size(
    angular_height: float,
    object_height: float,
    eps: float = 1e-3,
) -> float:
    """
    Pinhole-style range from apparent angular height:
    ``R ≈ H / (2 * tan(α / 2))``.
    """
    alpha = float(np.clip(angular_height, eps, np.pi - eps))
    height = float(max(object_height, eps))
    return float(height / (2.0 * np.tan(0.5 * alpha)))


def height_from_angular_size(
    angular_height: float,
    range_m: float,
    eps: float = 1e-3,
) -> float:
    """
    Inverse of ``range_from_angular_size``: metric height implied by apparent
    angular height at a known horizontal range.
    ``H ≈ 2 R tan(α / 2)``.
    """
    alpha = float(np.clip(angular_height, eps, np.pi - eps))
    r = float(max(range_m, eps))
    return float(2.0 * r * np.tan(0.5 * alpha))


def azimuth_unit_xz(direction: np.ndarray, uv: np.ndarray) -> np.ndarray | None:
    """
    Unit horizontal bearing in the JSON XZ plane.

    Prefers the ray's XZ component; near nadir falls back to longitude-only
    bearing from ``uv`` so freestanding range placement still has an azimuth.
    """
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    horiz = direction.copy()
    horiz[1] = 0.0
    n = float(np.linalg.norm(horiz))
    if n >= 1e-8:
        return horiz / n

    lon = float(uv2lonlat(np.asarray(uv, dtype=np.float64).reshape(2))[0])
    # Equator ray in LGT spherical coords, then into xyz2json.
    bearing = to_json_frame(
        np.array([np.sin(lon), 0.0, np.cos(lon)], dtype=np.float64)
    )
    bearing[1] = 0.0
    n = float(np.linalg.norm(bearing))
    if n < 1e-8:
        return None
    return bearing / n


def wall_fraction_from_elevation(
    direction: np.ndarray,
    fraction_near: float = 0.28,
    fraction_far: float = 0.82,
    dy_near: float = 0.97,
    dy_far: float = 0.35,
) -> float:
    """
    Map ray elevation to a freestanding wall-distance fraction.

    In the JSON frame +Y points at the floor, so ``|d_y|≈1`` is nadir (near
    camera) and smaller ``|d_y|`` is toward the horizon (farther into the room).
    """
    dy = abs(float(np.asarray(direction, dtype=np.float64).reshape(3)[1]))
    if dy_near <= dy_far:
        return float(fraction_near)
    t = (dy - dy_far) / (dy_near - dy_far)
    t = float(np.clip(t, 0.0, 1.0))
    return float(fraction_far * (1.0 - t) + fraction_near * t)


def place_on_floor_at_range(
    origin: np.ndarray,
    azimuth_xz: np.ndarray,
    range_m: float,
    data: dict,
) -> np.ndarray:
    """Point on the JSON floor plane at ``range_m`` along ``azimuth_xz``."""
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    azimuth_xz = _normalize(
        np.array([azimuth_xz[0], 0.0, azimuth_xz[2]], dtype=np.float64)
    )
    camera_height = float(data["cameraHeight"])
    r = float(max(range_m, 0.0))
    return np.array(
        [
            origin[0] + r * azimuth_xz[0],
            camera_height,
            origin[2] + r * azimuth_xz[2],
        ],
        dtype=np.float64,
    )


def shrink_range_into_footprint(
    origin: np.ndarray,
    azimuth_xz: np.ndarray,
    range_m: float,
    data: dict,
    min_range: float = 0.2,
    steps: int = 12,
) -> np.ndarray | None:
    """Reduce range until the floor point lies inside the layout footprint."""
    r = float(range_m)
    for _ in range(steps):
        p = place_on_floor_at_range(origin, azimuth_xz, r, data)
        if check_floor_hit(p, data):
            return p
        r *= 0.85
        if r < min_range:
            break
    return None
