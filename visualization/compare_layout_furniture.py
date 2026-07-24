"""
@author: LGT-Net Project
@time: 2026/07/24
@description:
    将 LGT-Net 生成的房间结构（.obj / .gltf）与 SAM3D 生成的家具模型
    （.glb / .ply）放在同一个 3D 场景中进行对比可视化。

    用法示例：
        # 命令行方式
        python visualization/compare_layout_furniture.py \
            --room_mesh src/output/room_3d.obj \
            --furniture_globs "src/output/furniture/*.glb" \
            --furniture_globs "src/output/furniture/*.ply"

        # 编程方式
        from visualization.compare_layout_furniture import compare_layout_and_furniture
        compare_layout_and_furniture(
            room_path="src/output/room_3d.obj",
            furniture_paths=["src/output/chair.glb", "src/output/table.ply"],
            show=True,
        )

    依赖: open3d, numpy (项目已有依赖，无需额外安装)
"""

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# 常量定义
# ---------------------------------------------------------------------------

# LGT-Net 默认相机高度（米），与 utils/writer.py 中保持一致
DEFAULT_CAMERA_HEIGHT = 1.6

# 房间 mesh 默认渲染颜色（浅灰色，半透明以看清内部家具）
ROOM_DEFAULT_COLOR = np.array([0.75, 0.75, 0.75])  # RGB, 范围 [0, 1]

# 家具模型默认渲染颜色列表（用于逐个区分不同家具）
FURNITURE_COLORS = [
    np.array([0.90, 0.40, 0.30]),  # 红棕色 — 家具 A
    np.array([0.30, 0.55, 0.85]),  # 蓝色   — 家具 B
    np.array([0.35, 0.70, 0.35]),  # 绿色   — 家具 C
    np.array([0.85, 0.75, 0.25]),  # 金黄色 — 家具 D
    np.array([0.65, 0.35, 0.75]),  # 紫色   — 家具 E
    np.array([0.40, 0.70, 0.75]),  # 青色   — 家具 F
    np.array([0.90, 0.55, 0.65]),  # 粉色   — 家具 G
    np.array([0.55, 0.55, 0.35]),  # 橄榄色 — 家具 H
]

# 支持的 mesh 文件扩展名
ROOM_MESH_EXTS = {'.obj', '.gltf', '.glb'}
FURNITURE_MESH_EXTS = {'.glb', '.ply', '.obj', '.stl', '.fbx'}


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _is_valid_mesh_file(file_path: str, allowed_exts: set) -> bool:
    """检查文件是否为支持的 mesh 格式。

    Args:
        file_path: 文件路径字符串。
        allowed_exts: 允许的扩展名集合。

    Returns:
        True 如果文件存在且扩展名在允许列表中。
    """
    path = Path(file_path)
    return path.is_file() and path.suffix.lower() in allowed_exts


def _resolve_glob_paths(glob_patterns: List[str]) -> List[str]:
    """将 glob 模式展开为具体的文件路径列表。

    Args:
        glob_patterns: glob 模式字符串列表，例如 ["/path/to/*.glb"]。

    Returns:
        排序后的文件路径列表。
    """
    all_paths = set()
    for pattern in glob_patterns:
        matched = glob.glob(pattern, recursive=True)
        all_paths.update(os.path.abspath(p) for p in matched)
    return sorted(all_paths)


def _classify_files(file_paths: List[str]) -> Tuple[List[str], List[str]]:
    """将文件列表分为"房间 mesh"和"家具 mesh"两类。

    根据扩展名判断：
      - .obj / .gltf 视为房间结构（LGT 输出）
      - .glb / .ply / .stl / .fbx 视为家具模型（SAM3D 输出）

    Args:
        file_paths: 文件路径列表。

    Returns:
        (room_paths, furniture_paths) 两个列表。
    """
    room_paths: List[str] = []
    furniture_paths: List[str] = []

    for fp in file_paths:
        ext = Path(fp).suffix.lower()
        if ext in ROOM_MESH_EXTS:
            room_paths.append(fp)
        elif ext in FURNITURE_MESH_EXTS:
            furniture_paths.append(fp)
        else:
            print(f"[警告] 跳过未知格式文件: {fp}")

    return room_paths, furniture_paths


# ---------------------------------------------------------------------------
# Mesh 加载与预处理
# ---------------------------------------------------------------------------

def load_mesh(file_path: str, verbose: bool = True) -> Optional["open3d.geometry.TriangleMesh"]:
    """使用 Open3D 加载单个 mesh 文件。

    支持格式：.obj, .gltf, .glb, .ply, .stl, .fbx 等。

    Args:
        file_path: mesh 文件路径。
        verbose: 是否打印加载信息。

    Returns:
        加载成功的 TriangleMesh 对象，失败则返回 None。
    """
    import open3d as o3d

    path = Path(file_path)
    if not path.is_file():
        print(f"[错误] 文件不存在: {file_path}")
        return None

    ext = path.suffix.lower()
    try:
        # 根据文件格式选择合适的读取方式
        if ext in {'.obj', '.gltf', '.glb', '.fbx'}:
            # 带纹理的 mesh —— 使用 read_triangle_mesh 保留材质信息
            mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=True)
        elif ext in {'.ply', '.stl'}:
            mesh = o3d.io.read_triangle_mesh(str(path))
        else:
            # fallback: 尝试通用读取
            mesh = o3d.io.read_triangle_mesh(str(path))

        if mesh.is_empty():
            print(f"[警告] 加载的 mesh 为空: {file_path}")
            return None

        # 计算法向量以保证光照渲染正确
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals(normalized=True)

        if verbose:
            verts = len(mesh.vertices)
            tris = len(mesh.triangles)
            has_tex = mesh.has_textures()
            print(f"  [OK] 已加载: {path.name}  (顶点: {verts}, 三角面: {tris}, 纹理: {has_tex})")

        return mesh

    except Exception as e:
        print(f"[错误] 加载 mesh 失败 ({file_path}): {e}")
        return None


def load_room_mesh(file_path: str, verbose: bool = True) -> Optional["open3d.geometry.TriangleMesh"]:
    """加载 LGT-Net 生成的房间结构 mesh。

    房间 mesh 通常是从全景深度图生成的纹理化 3D 结构（obj/gltf 格式）。
    加载后会对其设置统一的浅灰色以便和家具区分。

    Args:
        file_path: 房间 mesh 文件路径 (.obj / .gltf)。
        verbose: 是否打印加载信息。

    Returns:
        房间 TriangleMesh 对象，失败返回 None。
    """
    import open3d as o3d

    mesh = load_mesh(file_path, verbose=verbose)
    if mesh is None:
        return None

    if verbose:
        print(f"  [房间] {Path(file_path).name} — LGT-Net 房间结构")

    return mesh


def load_furniture_meshes(
    file_paths: List[str],
    verbose: bool = True,
) -> List["open3d.geometry.TriangleMesh"]:
    """批量加载 SAM3D 生成的家具模型。

    每个家具 mesh 会被自动赋予不同颜色以便逐个区分。
    如果文件路径不存在或加载失败，会打印警告并跳过。

    Args:
        file_paths: 家具 mesh 文件路径列表 (.glb / .ply)。
        verbose: 是否打印加载信息。

    Returns:
        加载成功的家具 TriangleMesh 列表。
    """
    import open3d as o3d

    furniture_meshes: List[o3d.geometry.TriangleMesh] = []

    for i, fp in enumerate(file_paths):
        if verbose:
            print(f"  [家具 {i + 1}] 正在加载: {Path(fp).name} ...")

        mesh = load_mesh(fp, verbose=verbose)
        if mesh is None:
            continue

        furniture_meshes.append(mesh)

    if verbose:
        print(f"  共加载 {len(furniture_meshes)} 个家具模型")

    return furniture_meshes


# ---------------------------------------------------------------------------
# Mesh 着色与变换
# ---------------------------------------------------------------------------

def apply_uniform_color(
    mesh: "open3d.geometry.TriangleMesh",
    color: np.ndarray,
    alpha: float = 1.0,
) -> None:
    """给 mesh 应用统一的单色（覆盖原有纹理/顶点颜色）。

    常用于给家具模型着色以便在场景中区分不同物体。

    Args:
        mesh: 目标 TriangleMesh。
        color: RGB 颜色，numpy array 形状 (3,)，每个通道范围 [0, 1]。
        alpha: 透明度（暂未在 Open3D 中实际生效，保留以备后续使用）。
    """
    import open3d as o3d

    num_verts = len(mesh.vertices)
    colors = np.tile(color, (num_verts, 1))
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)


def set_room_transparency(
    mesh: "open3d.geometry.TriangleMesh",
    alpha: float = 0.6,
) -> None:
    """将房间 mesh 设置为半透明，以便透过墙壁看到内部的家具。

    注意：Open3D 的默认可视化器对半透明支持有限，
    这里的 alpha 值通过降低颜色亮度来近似透明效果。

    Args:
        mesh: 房间 TriangleMesh。
        alpha: 透明度系数 [0, 1]，越小越透明（颜色越暗）。
    """
    import open3d as o3d

    if mesh.has_vertex_colors():
        colors = np.asarray(mesh.vertex_colors)
        # 通过混合白色来模拟半透明（Alpha blending to white background）
        white_bg = np.ones_like(colors)
        blended = colors * alpha + white_bg * (1 - alpha)
        mesh.vertex_colors = o3d.utility.Vector3dVector(blended)
    else:
        # 如果没有顶点颜色，设置一个带透明度的灰色
        num_verts = len(mesh.vertices)
        color = ROOM_DEFAULT_COLOR * alpha + np.ones(3) * (1 - alpha)
        colors = np.tile(color, (num_verts, 1))
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)


def transform_mesh(
    mesh: "open3d.geometry.TriangleMesh",
    translation: Optional[np.ndarray] = None,
    rotation: Optional[np.ndarray] = None,
    scale: float = 1.0,
) -> "open3d.geometry.TriangleMesh":
    """对 mesh 进行平移、旋转和缩放变换。

    Args:
        mesh: 要变换的 TriangleMesh。
        translation: 平移向量 (3,)，单位米。None 表示不平移。
        rotation: 3×3 旋转矩阵。None 表示不旋转。
        scale: 均匀缩放因子，默认 1.0（不变）。

    Returns:
        变换后的 mesh（原地修改并返回）。
    """
    if scale != 1.0:
        mesh.scale(scale, center=np.zeros(3))

    if rotation is not None:
        mesh.rotate(rotation, center=np.zeros(3))

    if translation is not None:
        mesh.translate(translation)

    return mesh


def align_furniture_to_room(
    furniture_mesh: "open3d.geometry.TriangleMesh",
    floor_y: float = 0.0,
) -> "open3d.geometry.TriangleMesh":
    """将家具模型对齐到房间地面。

    自动检测家具 mesh 的包围盒最低点，将其平移到房间地面高度 y = floor_y。

    Args:
        furniture_mesh: 家具 TriangleMesh。
        floor_y: 房间地面的 y 坐标（LGT 坐标系中地面高度）。

    Returns:
        对齐后的家具 mesh。
    """
    bbox = furniture_mesh.get_axis_aligned_bounding_box()
    min_bound = bbox.min_bound  # (x_min, y_min, z_min)
    offset_y = floor_y - min_bound[1]

    furniture_mesh.translate(np.array([0.0, offset_y, 0.0]))
    return furniture_mesh


# ---------------------------------------------------------------------------
# 坐标轴与参考网格
# ---------------------------------------------------------------------------

def create_ground_grid(
    size: float = 6.0,
    step: float = 0.5,
    origin: np.ndarray = np.array([0.0, 0.0, 0.0]),
    color: np.ndarray = np.array([0.5, 0.5, 0.5]),
) -> List["open3d.geometry.LineSet"]:
    """在 y=0 平面上创建地面参考网格，帮助判断家具摆放位置和尺度。

    Args:
        size: 网格总边长（米），网格从 -size/2 到 +size/2。
        step: 网格线间距（米）。
        origin: 网格中心点位置。
        color: 网格线 RGB 颜色。

    Returns:
        包含一个 LineSet 的列表（为与 Open3D 接口兼容，统一返回列表）。
    """
    import open3d as o3d

    lines = []
    half = size / 2

    # 沿 x 方向的网格线 (z 方向变化)
    z = -half
    while z <= half:
        start = origin + np.array([-half, 0.0, z])
        end = origin + np.array([half, 0.0, z])
        lines.append((start, end))
        z += step

    # 沿 z 方向的网格线 (x 方向变化)
    x = -half
    while x <= half:
        start = origin + np.array([x, 0.0, -half])
        end = origin + np.array([x, 0.0, half])
        lines.append((start, end))
        x += step

    if not lines:
        return []

    points = []
    line_indices = []
    for start, end in lines:
        idx_start = len(points)
        points.append(start)
        points.append(end)
        line_indices.append([idx_start, idx_start + 1])

    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(np.array(points))
    grid.lines = o3d.utility.Vector2iVector(np.array(line_indices))
    colors = np.tile(color, (len(line_indices), 1))
    grid.colors = o3d.utility.Vector3dVector(colors)

    return [grid]


def create_coordinate_frame(size: float = 1.0) -> "open3d.geometry.TriangleMesh":
    """创建 RGB 三色坐标轴（X=红, Y=绿, Z=蓝），用于判断场景朝向。

    Args:
        size: 坐标轴长度（米）。

    Returns:
        TriangleMesh 格式的坐标轴（通过 create_coordinate_frame 生成）。
    """
    import open3d as o3d

    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)


# ---------------------------------------------------------------------------
# 核心对比可视化
# ---------------------------------------------------------------------------

def compare_layout_and_furniture(
    room_path: Optional[str] = None,
    furniture_paths: Optional[List[str]] = None,
    room_color: np.ndarray = ROOM_DEFAULT_COLOR,
    furniture_colors: Optional[List[np.ndarray]] = None,
    show_ground_grid: bool = True,
    ground_grid_size: float = 6.0,
    ground_grid_step: float = 0.5,
    show_coordinate_frame: bool = True,
    coordinate_frame_size: float = 1.0,
    room_transparency: float = 0.4,
    auto_align_furniture: bool = True,
    room_mesh_show_back_face: bool = False,
    window_title: str = "LGT-Net 房间结构 × SAM3D 家具模型 — 对比视图",
    point_size: float = 1.0,
    show: bool = True,
    save_screenshot: Optional[str] = None,
    verbose: bool = True,
    return_geometries: bool = False,
) -> Optional[List]:
    """将 LGT-Net 房间结构与 SAM3D 家具模型放在同一 3D 场景中对比可视化。

    这是本模块的核心函数。它：
      1. 加载 LGT 生成的房间结构 mesh（.obj / .gltf）
      2. 加载 SAM3D 生成的家具模型（.glb / .ply），每个家具自动赋予不同颜色
      3. 自动对齐家具到房间地面
      4. 可选显示地面参考网格和坐标轴
      5. 启动交互式 3D 可视化窗口

    Args:
        room_path:
            LGT-Net 生成的房间 mesh 文件路径（.obj / .gltf）。
            设为 None 则只显示家具（无房间外壳）。
        furniture_paths:
            SAM3D 生成的家具 mesh 文件路径列表。
            支持 .glb / .ply / .obj / .stl 等格式。
            设为 None 或空列表则只显示房间。
        room_color:
            房间 mesh 的统一 RGB 颜色，形状 (3,)，默认浅灰色。
        furniture_colors:
            每个家具的颜色列表，长度应与 furniture_paths 一致。
            设为 None 则自动使用预定义的调色板。
        show_ground_grid:
            是否显示 y=0 平面的参考网格。
        ground_grid_size:
            地面网格总边长（米）。
        ground_grid_step:
            地面网格线间距（米）。
        show_coordinate_frame:
            是否显示 RGB 坐标轴。
        coordinate_frame_size:
            坐标轴长度（米）。
        room_transparency:
            房间墙壁透明度 [0, 1]，越小越透明。
            用于透过墙壁看到内部家具。
        auto_align_furniture:
            是否自动将家具平移到房间地面（y = 0 平面）。
        room_mesh_show_back_face:
            是否渲染房间 mesh 的背面（通常不需要）。
        window_title:
            可视化窗口标题。
        point_size:
            点云模式下点的大小（仅当 mesh 被转为点云时生效）。
        show:
            是否调用 Open3D 的 draw_geometries 弹出交互窗口。
            设为 False 则只返回 geometries 列表而不显示。
        save_screenshot:
            若提供路径，在关闭窗口后将当前视图保存为图片。
            （需要 Open3D >= 0.18 的截图支持）
        verbose:
            是否打印详细加载信息。
        return_geometries:
            是否返回 geometries 列表（用于进一步编程操作）。

    Returns:
        如果 return_geometries=True，返回 geometry 对象列表；
        否则返回 None。

    示例:
        >>> compare_layout_and_furniture(
        ...     room_path="src/output/room_3d.obj",
        ...     furniture_paths=["chair.glb", "table.ply", "sofa.glb"],
        ...     show=True,
        ... )

        >>> # 只查看房间结构
        >>> compare_layout_and_furniture(
        ...     room_path="src/output/room_3d.obj",
        ...     show=True,
        ... )

        >>> # 只查看家具模型
        >>> compare_layout_and_furniture(
        ...     furniture_paths=["chair.glb", "table.ply"],
        ...     show=True,
        ... )
    """
    import open3d as o3d

    geometries: List = []

    # ---- 1. 加载房间结构 mesh ----
    room_mesh = None
    if room_path is not None:
        if verbose:
            print(f"\n{'='*60}")
            print(f"  加载 LGT-Net 房间结构")
            print(f"{'='*60}")

        room_mesh = load_room_mesh(room_path, verbose=verbose)

        if room_mesh is not None:
            # 若房间 mesh 无纹理，应用统一颜色
            if not room_mesh.has_textures() and not room_mesh.has_vertex_colors():
                apply_uniform_color(room_mesh, room_color)

            # 设置半透明效果（透过墙壁看家具）
            if room_transparency < 1.0:
                set_room_transparency(room_mesh, alpha=room_transparency)

            geometries.append(room_mesh)
        else:
            print("[警告] 房间 mesh 加载失败，将继续只显示家具模型")

    # ---- 2. 加载家具模型 ----
    furniture_meshes: List = []
    if furniture_paths:
        if verbose:
            print(f"\n{'='*60}")
            print(f"  加载 SAM3D 家具模型")
            print(f"{'='*60}")

        furniture_meshes = load_furniture_meshes(furniture_paths, verbose=verbose)

        # 为每个家具赋予不同颜色
        color_palette = furniture_colors if furniture_colors else FURNITURE_COLORS
        for i, fm in enumerate(furniture_meshes):
            # 如果没有纹理，赋予颜色以便区分
            if not fm.has_textures():
                color = color_palette[i % len(color_palette)]
                apply_uniform_color(fm, color)
                if verbose:
                    c = (color * 255).astype(int)
                    print(f"    家具 {i + 1} 着色: RGB({c[0]}, {c[1]}, {c[2]})")

            # 自动对齐到地面 (y=0)
            if auto_align_furniture:
                align_furniture_to_room(fm, floor_y=0.0)

            geometries.append(fm)

    if not furniture_meshes and room_mesh is None:
        print("[错误] 没有成功加载任何 mesh，无法进行可视化")
        return None

    # ---- 3. 添加地面参考网格 ----
    if show_ground_grid:
        grid = create_ground_grid(
            size=ground_grid_size,
            step=ground_grid_step,
        )
        geometries.extend(grid)

    # ---- 4. 添加坐标轴 ----
    if show_coordinate_frame:
        axis = create_coordinate_frame(size=coordinate_frame_size)
        geometries.append(axis)

    # ---- 5. 启动交互式 3D 可视化 ----
    if show and geometries:
        if verbose:
            print(f"\n{'='*60}")
            print(f"  3D 对比视图")
            print(f"  房间: {'[OK]' if room_mesh else '[NO]'}")
            print(f"  家具数量: {len(furniture_meshes)}")
            print(f"  操作提示:")
            print(f"    鼠标左键拖拽  = 旋转视角")
            print(f"    鼠标右键拖拽  = 平移视角")
            print(f"    鼠标滚轮      = 缩放")
            print(f"    Ctrl + 左/右  = 翻滚")
            print(f"    R 键          = 重置视角")
            print(f"    W 键          = 线框/实体切换")
            print(f"{'='*60}\n")

        try:
            o3d.visualization.draw_geometries(
                geometries,
                window_name=window_title,
                mesh_show_back_face=room_mesh_show_back_face,
                point_show_normal=False,
                width=1280,
                height=720,
                left=50,
                top=50,
            )
        except Exception as e:
            print(f"[错误] Open3D 可视化失败: {e}")
            print("  请检查 mesh 数据是否有效，或尝试仅加载单个文件进行排查")

    # ---- 6. 可选截图 ----
    if save_screenshot and geometries:
        try:
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=window_title, width=1280, height=720)
            for g in geometries:
                vis.add_geometry(g)
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(save_screenshot)
            vis.destroy_window()
            if verbose:
                print(f"  截图已保存: {save_screenshot}")
        except Exception as e:
            print(f"[警告] 截图失败: {e}")

    if return_geometries:
        return geometries
    return None


# ---------------------------------------------------------------------------
# 快捷函数
# ---------------------------------------------------------------------------

def compare_from_directory(
    room_dir: str,
    furniture_dir: str,
    room_glob: str = "*.obj",
    furniture_glob: str = "*.glb",
    **kwargs,
) -> Optional[List]:
    """从两个目录中自动匹配并对比房间与家具。

    自动选取第一个匹配到的房间文件和所有匹配到的家具文件。

    Args:
        room_dir: 存放房间 mesh 的目录路径。
        furniture_dir: 存放家具 mesh 的目录路径。
        room_glob: 房间文件的 glob 模式，默认 "*.obj"。
        furniture_glob: 家具文件的 glob 模式，默认 "*.glb"。
        **kwargs: 传递给 compare_layout_and_furniture 的其他参数。

    Returns:
        见 compare_layout_and_furniture 的返回值。

    示例:
        >>> compare_from_directory(
        ...     room_dir="src/output/job_001/",
        ...     furniture_dir="src/output/furniture/",
        ...     show=True,
        ... )
    """
    room_pattern = os.path.join(room_dir, room_glob)
    furniture_pattern = os.path.join(furniture_dir, furniture_glob)

    room_files = sorted(glob.glob(room_pattern))
    furniture_files = sorted(glob.glob(furniture_pattern, recursive=True))

    if not room_files:
        print(f"[警告] 在 {room_dir} 中未找到匹配 {room_glob} 的房间文件")

    if not furniture_files:
        print(f"[警告] 在 {furniture_dir} 中未找到匹配 {furniture_glob} 的家具文件")

    room_path = room_files[0] if room_files else None

    return compare_layout_and_furniture(
        room_path=room_path,
        furniture_paths=furniture_files if furniture_files else None,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。

    Returns:
        配置好的 ArgumentParser 对象。
    """
    parser = argparse.ArgumentParser(
        description=(
            "LGT-Net 房间结构 × SAM3D 家具模型 — 3D 对比可视化工具\n\n"
            "将 LGT-Net 生成的房间结构 mesh (.obj/.gltf) 与 "
            "SAM3D 生成的家具模型 (.glb/.ply) 放在同一场景中交互式查看。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基础用法：指定房间和家具文件
  python visualization/compare_layout_furniture.py \\
      --room_mesh src/output/room_3d.obj \\
      --furniture chair.glb table.ply sofa.glb

  # 使用 glob 模式批量加载家具
  python visualization/compare_layout_furniture.py \\
      --room_mesh src/output/room_3d.obj \\
      --furniture_globs "furniture/*.glb" "furniture/*.ply"

  # 只用目录自动匹配
  python visualization/compare_layout_furniture.py \\
      --room_dir src/output/ \\
      --furniture_dir furniture/

  # 只查看房间结构（无家具）
  python visualization/compare_layout_furniture.py \\
      --room_mesh src/output/room_3d.obj

  # 只查看家具模型
  python visualization/compare_layout_furniture.py \\
      --furniture chair.glb table.ply
        """,
    )

    # ---- 输入源 ----
    input_group = parser.add_argument_group("输入源（三选一）")
    input_group.add_argument(
        "--room_mesh",
        type=str,
        default=None,
        help="LGT-Net 生成的房间结构 mesh 路径 (.obj / .gltf)",
    )
    input_group.add_argument(
        "--furniture",
        type=str,
        nargs="*",
        default=None,
        help="SAM3D 生成的家具模型路径列表 (.glb / .ply)，空格分隔",
    )
    input_group.add_argument(
        "--furniture_globs",
        type=str,
        nargs="*",
        default=None,
        help="家具文件的 glob 模式，例如 'output/*.glb' 'output/*.ply'",
    )
    input_group.add_argument(
        "--room_dir",
        type=str,
        default=None,
        help="房间 mesh 所在目录（自动选取第一个 .obj 文件）",
    )
    input_group.add_argument(
        "--furniture_dir",
        type=str,
        default=None,
        help="家具 mesh 所在目录（自动选取所有 .glb/.ply 文件）",
    )

    # ---- 显示选项 ----
    display_group = parser.add_argument_group("显示选项")
    display_group.add_argument(
        "--no_grid",
        action="store_true",
        help="隐藏地面参考网格",
    )
    display_group.add_argument(
        "--grid_size",
        type=float,
        default=6.0,
        help="地面网格总边长（米），默认 6.0",
    )
    display_group.add_argument(
        "--grid_step",
        type=float,
        default=0.5,
        help="地面网格线间距（米），默认 0.5",
    )
    display_group.add_argument(
        "--no_axis",
        action="store_true",
        help="隐藏 RGB 坐标轴",
    )
    display_group.add_argument(
        "--room_transparency",
        type=float,
        default=0.4,
        help="房间墙壁透明度 [0-1]，越小越透明。默认 0.4",
    )
    display_group.add_argument(
        "--show_back_face",
        action="store_true",
        help="渲染房间 mesh 的背面",
    )
    display_group.add_argument(
        "--no_align",
        action="store_true",
        help="禁止自动将家具对齐到地面",
    )
    display_group.add_argument(
        "--title",
        type=str,
        default="LGT-Net 房间结构 × SAM3D 家具模型 — 对比视图",
        help="窗口标题",
    )

    # ---- 输出选项 ----
    output_group = parser.add_argument_group("输出选项")
    output_group.add_argument(
        "--save_screenshot",
        type=str,
        default=None,
        help="截图保存路径（需要 Open3D >= 0.18）",
    )
    output_group.add_argument(
        "--no_show",
        action="store_true",
        help="不弹出交互窗口（与 --save_screenshot 配合使用）",
    )
    output_group.add_argument(
        "--quiet",
        action="store_true",
        help="静默模式，不打印详细信息",
    )

    return parser


def main():
    """命令行主入口。"""
    parser = build_arg_parser()
    args = parser.parse_args()

    verbose = not args.quiet

    # ---- 收集房间路径 ----
    room_path: Optional[str] = None
    if args.room_mesh:
        room_path = args.room_mesh
    elif args.room_dir:
        room_files = sorted(glob.glob(os.path.join(args.room_dir, "*.obj")))
        if not room_files:
            room_files = sorted(glob.glob(os.path.join(args.room_dir, "*.gltf")))
        if room_files:
            room_path = room_files[0]
            if verbose:
                print(f"自动选择房间文件: {room_path}")
        else:
            print(f"[警告] 在 {args.room_dir} 中未找到 .obj 或 .gltf 文件")

    # ---- 收集家具路径 ----
    furniture_paths: List[str] = []

    # 直接指定的家具文件
    if args.furniture:
        for fp in args.furniture:
            fp = os.path.abspath(fp)
            if _is_valid_mesh_file(fp, FURNITURE_MESH_EXTS | ROOM_MESH_EXTS):
                furniture_paths.append(fp)
            else:
                print(f"[警告] 跳过无效文件: {fp}")

    # glob 模式展开
    if args.furniture_globs:
        expanded = _resolve_glob_paths(args.furniture_globs)
        furniture_paths.extend(expanded)

    # 从目录自动匹配
    if args.furniture_dir:
        for ext in ['.glb', '.ply', '.obj', '.stl']:
            furniture_paths.extend(
                sorted(glob.glob(os.path.join(args.furniture_dir, f"*{ext}")))
            )

    # 去重
    furniture_paths = sorted(set(furniture_paths))

    # 去重时排除 room_path 自身
    if room_path and room_path in furniture_paths:
        furniture_paths.remove(room_path)

    # 分类：把 .obj/.gltf 放到房间，其余留作家具
    if not room_path and furniture_paths:
        room_candidates, furniture_paths = _classify_files(furniture_paths)
        if room_candidates and not room_path:
            room_path = room_candidates[0]
            if verbose:
                print(f"自动识别房间文件: {room_path}")

    # ---- 参数校验 ----
    if not room_path and not furniture_paths:
        parser.error(
            "请至少指定 --room_mesh/--room_dir 或 --furniture/--furniture_globs/--furniture_dir\n"
            "使用 --help 查看详细用法"
        )

    if verbose:
        print(f"\n  房间: {room_path or '(无)'}")
        print(f"  家具数量: {len(furniture_paths)}")
        for fp in furniture_paths:
            print(f"    - {Path(fp).name}")

    # ---- 执行对比可视化 ----
    compare_layout_and_furniture(
        room_path=room_path,
        furniture_paths=furniture_paths if furniture_paths else None,
        show_ground_grid=not args.no_grid,
        ground_grid_size=args.grid_size,
        ground_grid_step=args.grid_step,
        show_coordinate_frame=not args.no_axis,
        room_transparency=args.room_transparency,
        auto_align_furniture=not args.no_align,
        room_mesh_show_back_face=args.show_back_face,
        window_title=args.title,
        show=not args.no_show,
        save_screenshot=args.save_screenshot,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# 编程接口快捷入口
# ---------------------------------------------------------------------------

def quick_compare(
    room: str,
    *furniture: str,
    **kwargs,
) -> Optional[List]:
    """最简调用接口：给定房间文件和若干家具文件，直接弹出对比窗口。

    Args:
        room: 房间 mesh 路径 (.obj / .gltf)。
        *furniture: 可变数量的家具 mesh 路径 (.glb / .ply)。
        **kwargs: 传递给 compare_layout_and_furniture 的额外参数。

    Returns:
        见 compare_layout_and_furniture。

    示例:
        >>> from visualization.compare_layout_furniture import quick_compare
        >>> quick_compare("room.obj", "chair.glb", "table.glb", "sofa.ply")
    """
    return compare_layout_and_furniture(
        room_path=room,
        furniture_paths=list(furniture) if furniture else None,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 直接运行
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
