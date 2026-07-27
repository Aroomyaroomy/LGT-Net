"""
@author: Zhigang Jiang
@time: 2022/05/25
@description: reference: https://github.com/sunset1995/PanoPlane360/blob/main/vis_planes.py
"""
import os

import numpy as np
from utils.conversion import pixel2lonlat


def point_cloud_path_from_mesh_path(mesh_path):
    root, _ = os.path.splitext(mesh_path)
    return f'{root}.ply'


def create_3d_obj(
    img,
    depth,
    save_path=None,
    mesh=True,
    mesh_show_back_face=False,
    show=False,
    write_mesh=None,
    write_point_cloud=None,
):
    """Build and optionally persist a layout mesh and/or colored point cloud.

    When ``save_path`` is set and ``write_mesh`` / ``write_point_cloud`` are left
    as ``None``, both artifacts are written (legacy behavior). Pass explicit
    booleans to generate only what you need.
    """
    import open3d
    assert img.shape[0] == depth.shape[0], ""
    h = img.shape[0]
    w = img.shape[1]

    if write_mesh is None:
        write_mesh = save_path is not None
    if write_point_cloud is None:
        write_point_cloud = save_path is not None

    need_mesh = write_mesh or (show and mesh)
    need_point_cloud = write_point_cloud or (show and not mesh)

    # Project to 3d
    lon = pixel2lonlat(np.array(range(w)), w=w, axis=0)[None].repeat(h, axis=0)
    lat = pixel2lonlat(np.array(range(h)), h=h, axis=1)[..., None].repeat(w, axis=1)

    z = depth * np.sin(lat)
    x = depth * np.cos(lat) * np.cos(lon)
    y = depth * np.cos(lat) * np.sin(lon)
    pts_xyz = np.stack([x, -z, y], -1).reshape(-1, 3)
    pts_rgb = img.reshape(-1, 3)

    point_cloud = None
    if need_point_cloud:
        point_cloud = open3d.geometry.PointCloud()
        point_cloud.points = open3d.utility.Vector3dVector(pts_xyz)
        point_cloud.colors = open3d.utility.Vector3dVector(pts_rgb)

    triangle_mesh = None
    if need_mesh:
        pid = np.arange(len(pts_xyz)).reshape(h, w)
        faces = np.concatenate([
            np.stack([
                pid[:-1, :-1], pid[1:, :-1], np.roll(pid, -1, axis=1)[:-1, :-1],
            ], -1),
            np.stack([
                pid[1:, :-1], np.roll(pid, -1, axis=1)[1:, :-1], np.roll(pid, -1, axis=1)[:-1, :-1],
            ], -1)
        ]).reshape(-1, 3).tolist()
        triangle_mesh = open3d.geometry.TriangleMesh()
        triangle_mesh.vertices = open3d.utility.Vector3dVector(pts_xyz)
        triangle_mesh.vertex_colors = open3d.utility.Vector3dVector(pts_rgb)
        triangle_mesh.triangles = open3d.utility.Vector3iVector(faces)

    mesh_path = None
    point_cloud_path = None
    if save_path:
        if write_mesh:
            open3d.io.write_triangle_mesh(save_path, triangle_mesh, write_triangle_uvs=True)
            mesh_path = save_path
        if write_point_cloud:
            point_cloud_path = point_cloud_path_from_mesh_path(save_path)
            open3d.io.write_point_cloud(point_cloud_path, point_cloud)

    if show:
        scene = triangle_mesh if mesh else point_cloud
        open3d.visualization.draw_geometries([scene], mesh_show_back_face=mesh_show_back_face)

    return {
        'mesh_path': mesh_path,
        'point_cloud_path': point_cloud_path,
    }


if __name__ == '__main__':
    from dataset.mp3d_dataset import MP3DDataset
    from utils.boundary import depth2boundaries, layout2depth
    from visualization.boundary import draw_boundaries

    mp3d_dataset = MP3DDataset(root_dir='../src/dataset/mp3d', mode='train', for_test_index=10, patch_num=1024)
    gt = mp3d_dataset.__getitem__(3)

    boundary_list = depth2boundaries(gt['ratio'], gt['depth'], step=None)
    pano_img = draw_boundaries(gt['image'].transpose(1, 2, 0), boundary_list=boundary_list, show=True)
    layout_depth = layout2depth(boundary_list, show=False)
    create_3d_obj(gt['image'].transpose(1, 2, 0), layout_depth, save_path=f"../src/output/{gt['id']}_3d.gltf",
                  mesh=True)
