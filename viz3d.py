import open3d as o3d
import argparse
from pathlib import Path

def viz3d(mesh_path: Path, mesh_show_back_face: bool = False):
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=mesh_show_back_face)


def args_parser():
    parser = argparse.ArgumentParser(description="Visualize panorama layout")
    parser.add_argument("--mesh_path", type=Path, required=True, help="Path to the 3d panorama layout object")
    parser.add_argument("--show_back_face", action="store_true", help="Show the back face of the mesh")

    args = parser.parse_args()

    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("-" * 50)
    return args

if __name__ == "__main__":
    args = args_parser()
    viz3d(args.mesh_path, args.show_back_face)