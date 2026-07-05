"""
@author: Zhening Hu
@time: 2026/06/30
@description: fastapi endpoint serving LGT-Net
"""

import os
import secrets
import uuid
from io import BytesIO

import shutil

import asyncio
import datetime
import cv2
import numpy as np
import torch
import uvicorn
from argparse import Namespace
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Literal, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, Header, status
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from config.defaults import get_config
from inference import preprocess, save_pred_json
from models.build import build_model
from postprocessing.post_process import post_process
from utils.boundary import corners2boundaries, layout2depth
from utils.conversion import depth2xyz
from utils.logger import get_logger
from utils.misc import tensor2np

logger = get_logger()


# def down_ckpt(model_cfg, ckpt_dir):
#     model_ids = [
#         ['src/config/mp3d.yaml', '1o97oAmd-yEP5bQrM0eAWFPLq27FjUDbh'],
#         ['src/config/zind.yaml', '1PzBj-dfDfH_vevgSkRe5kczW0GVl_43I'],
#         ['src/config/pano.yaml', '1JoeqcPbm_XBPOi6O9GjjWi3_rtyPZS8m'],
#         ['src/config/s2d3d.yaml', '1PfJzcxzUsbwwMal7yTkBClIFgn8IdEzI'],
#         ['src/config/ablation_study/full.yaml', '1U16TxUkvZlRwJNaJnq9nAUap-BhCVIha']
#     ]

#     for model_id in model_ids:
#         if model_id[0] != model_cfg:
#             continue
#         path = os.path.join(ckpt_dir, 'best.pkl')
#         if not os.path.exists(path):
#             logger.info(f"Downloading {model_id}")
#             os.makedirs(ckpt_dir, exist_ok=True)
#             gdown.download(f"https://drive.google.com/uc?id={model_id[1]}", path, False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix='LGT_',
        env_file='.env',
        validate_default=True,
        frozen=True,
    )

    device: Literal['cuda', 'cpu'] = 'cuda'
    config_dir: str = 'src/config/zind.yaml'
    mesh_format: Literal['.gltf', '.glb', '.obj'] = '.obj'
    mesh_resolution: int = 1024
    visualize_3d: bool = False
    output_dir: str = 'src/output'
    file_ttl: int = 600 # seconds

    secret_key: Optional[str] = Field(default=None, min_length=16)

    dev_mode: bool = False

    @model_validator(mode='after')
    def require_secret_key(self) -> 'Settings':
        if self.secret_key is not None:
            stripped = self.secret_key.strip()
            if not stripped:
                object.__setattr__(self, 'secret_key', None)
            elif stripped != self.secret_key:
                object.__setattr__(self, 'secret_key', stripped)

        if self.dev_mode:
            if not self.secret_key:
                object.__setattr__(self, 'secret_key', 'dev-only-insecure-key')
            return self

        if not self.secret_key:
            raise ValueError(
                'LGT_SECRET_KEY is required when LGT_DEV_MODE is not enabled. '
                'Set LGT_SECRET_KEY in the environment or .env file.'
            )
        return self

@lru_cache
def get_settings() -> Settings:
    return Settings()


def resolve_device(requested_device: str) -> str:
    if requested_device == 'cuda' and not torch.cuda.is_available():
        logger.info('CUDA not available, falling back to CPU')
        return 'cpu'
    return requested_device


def build_model_from_settings(cfg_path: str, settings: Settings) -> torch.nn.Module:
    device = resolve_device(settings.device)
    legacy_cfg = Namespace(cfg=cfg_path, device=device)
    config = get_config(legacy_cfg)
    if device == 'cpu' and 'cuda' in config.TRAIN.DEVICE:
        config.defrost()
        config.TRAIN.DEVICE = 'cpu'
        config.freeze()
    model, _, _, _ = build_model(config, logger)
    model.eval()
    return model


async def ttl_clean_up_loop(settings: BaseSettings = Depends(get_settings), interval: int = 600):
    while True:
        await asyncio.sleep(interval)
        try:
            log = await asyncio.to_thread(clean_up_local_files, settings)
            logger.info(f"TTL File Clean Up: Total Files {log['total_files']}, Deleted Files {log['deleted_files']}")
        except Exception as e:
            logger.exception(f"Error cleaning up local files: {e}")
            continue

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.device = resolve_device(settings.device)
    app.state.model = build_model_from_settings(settings.config_dir, settings)

    clean_up_task = asyncio.create_task(ttl_clean_up_loop(settings, interval=600))

    yield

    clean_up_task.cancel()

    try:
        await clean_up_task
    except asyncio.CancelledError:
        pass

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_model(request: Request) -> torch.nn.Module:
    return request.app.state.model


def get_device(request: Request) -> str:
    return request.app.state.device


def validate_api_key(
    x_api_key: str = Header(..., alias="X-API-KEY"),
    settings: Settings = Depends(get_settings),
) -> None:
    expected_key = settings.secret_key
    if expected_key is None or not secrets.compare_digest(x_api_key, expected_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


# -------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------


app = FastAPI(
    title='LGT-Net',
    description='A GUI-less endpoint for transforming 2D panoramas into 3D room layout',
    lifespan=lifespan,
)


@app.get('/health')
def health(request: Request):
    model_loaded = hasattr(request.app.state, 'model') and request.app.state.model is not None
    body = {'status': 'healthy' if model_loaded else 'unhealthy', 'model_loaded': model_loaded}
    if not model_loaded:
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=body)
    return body


@torch.no_grad()
@app.post('/predict', dependencies=[Depends(validate_api_key)])
def predict(
    image: UploadFile = File(...),
    post_processing: Literal['manhattan', 'atalanta', 'original'] = Form('manhattan'),
    pre_processing: bool = Form(True),
    output_3d: bool = Form(False),
    settings: Settings = Depends(get_settings),
    model: torch.nn.Module = Depends(get_model),
    device: str = Depends(get_device),
):
    if image.content_type and not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail='Invalid input file format - must be an image')

    raw_bytes = image.file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail='Empty image file')

    job_id = uuid.uuid4().hex
    job_dir = os.path.join(settings.output_dir, job_id)
    os.makedirs(job_dir, exist_ok=True)

    img_array = np.array(
        Image.open(BytesIO(raw_bytes)).resize((1024, 512), Image.Resampling.BICUBIC)
    )[..., :3]

    if pre_processing:
        vp_cache_path = os.path.join(job_dir, f'{job_id}_vp.txt')
        img_array, _ = preprocess(img_array, vp_cache_path=vp_cache_path)

    img_array = (img_array / 255.0).astype(np.float32)

    dt = model(torch.from_numpy(img_array.transpose(2, 0, 1)[None]).to(device))
    if post_processing != 'original':
        dt['processed_xyz'] = post_process(tensor2np(dt['depth']), type_name=post_processing)

    output_xyz = dt['processed_xyz'][0] if 'processed_xyz' in dt else depth2xyz(tensor2np(dt['depth'][0]))
    json_data = save_pred_json(output_xyz, tensor2np(dt['ratio'][0])[0])

    mesh_url = None
    if output_3d:
        from visualization.obj3d import create_3d_obj
        dt_boundaries = corners2boundaries(
            tensor2np(dt['ratio'][0])[0],
            corners_xyz=output_xyz,
            step=None,
            length=settings.mesh_resolution if 'processed_xyz' in dt else None,
            visible=bool('processed_xyz' in dt),
        )
        dt_layout_depth = layout2depth(dt_boundaries, show=False)
        mesh_path = os.path.join(job_dir, f'{job_id}_3d{settings.mesh_format}')
        create_3d_obj(
            cv2.resize(img_array, dt_layout_depth.shape[::-1]),
            dt_layout_depth,
            save_path=mesh_path,
            mesh=True,
            show=settings.visualize_3d,
        )
        mesh_url = f'/jobs/{job_id}/mesh'

    return {
        'job_id': job_id,
        'coordinates': json_data,
        'mesh_url': mesh_url,
    }


@app.get('/jobs/{job_id}/mesh', dependencies=[Depends(validate_api_key)], response_class=FileResponse)
def download_mesh(job_id: str, settings: Settings = Depends(get_settings)):
    if len(job_id) != 32 or any(c not in '0123456789abcdef' for c in job_id):
        raise HTTPException(status_code=400, detail='Invalid job_id')

    mesh_name = f'{job_id}_3d{settings.mesh_format}'
    mesh_path = os.path.join(settings.output_dir, job_id, mesh_name)
    if not os.path.isfile(mesh_path):
        raise HTTPException(status_code=404, detail='Mesh file not found')

    if settings.mesh_format == '.obj':
        media_type = 'model/obj'
    elif settings.mesh_format == '.gltf':
        media_type = 'model/gltf+json'
    else:
        media_type = 'model/gltf-binary'

    return FileResponse(mesh_path, media_type=media_type, filename=mesh_name)


def clean_up_local_files(settings: BaseSettings) -> dict:
    log = {"total_files": 0, "deleted_files": 0}
    for file in os.listdir(settings.output_dir):
        file_path = os.path.join(settings.output_dir, file)
        if not os.path.exists(file_path):
            continue

        file_age = datetime.datetime.now() - datetime.datetime.fromtimestamp(os.path.getctime(file_path))
        try:
            if file_age.total_seconds() > settings.file_ttl:
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                elif os.path.isfile(file_path):
                    os.remove(file_path)
                log["deleted_files"] += 1
        except PermissionError:
            logger.warning(f"PermissionError when deleting file at {file_path}")
        log["total_files"] += 1
    return log


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)