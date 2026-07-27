import os
import json
from io import BytesIO
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

    return sorted(mask_paths, key=lambda x: int(x.name.split('_')[1]))


def preprocess_masks(masks_paths: list[Path], do_manhattan: bool = True, vp_cache_path: str = None, cached_vp: np.ndarray = None) -> list[np.ndarray]:
    contact_uvs = []
    for path in masks_paths:
        # preprocess with same pipeline as the full panorama
        mask = np.array(Image.open(path).resize((1024, 512), Image.Resampling.NEAREST))[..., :3]
        if do_manhattan:
            # vp_cache_path is always initialized as job_dir/job_id_vp.txt
            mask, _ = preprocess(mask, vp_cache_path=vp_cache_path, cached_vp = cached_vp)

        mask = (mask / 255.0).astype(np.float32)
        
        # find u-v coordinates of object contact point, where u is the horizontal and v is the vertical
        grayscale = mask.mean(axis=-1) if mask.ndim == 3 else mask

        binary = (grayscale > 0.5).astype(np.bool_)

        ys, xs = [0.], [0.]
        ys, xs = np.where(binary)
        
        v_bottom = ys.max()
        u_bottom_edge = xs[ys == v_bottom]
        u_center = 0.5 * (u_bottom_edge.min() + u_bottom_edge.max())

        contact_uv = pixel2uv(np.array([u_center, v_bottom], dtype=np.float16), 
                                w=binary.shape[1], h=binary.shape[0], dtype=np.float16)

        contact_uvs.append(contact_uv)

    return contact_uvs


def uv2equirectangular(uv: np.array([float, float])):
    lonlat = uv2lonlat(uv)
    xyz = lonlat2xyz(lonlat)
    return xyz

