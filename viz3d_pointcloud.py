import argparse
from pathlib import Path

import open3d as o3d


def viz3d_pointcloud(point_cloud_path: Path):
    point_cloud = o3d.io.read_point_cloud(str(point_cloud_path))
    if point_cloud.is_empty():
        raise ValueError(f"No points found in {point_cloud_path}")
    o3d.visualization.draw_geometries([point_cloud])


def args_parser():
    parser = argparse.ArgumentParser(description="Visualize panorama layout point cloud")
    parser.add_argument(
        "--point_cloud_path",
        type=Path,
        required=True,
        help="Path to the panorama layout point cloud (.ply)",
    )

    args = parser.parse_args()

    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("-" * 50)
    return args


if __name__ == "__main__":
    args = args_parser()
    viz3d_pointcloud(args.point_cloud_path)
