"""
@author: Zhening Hu
@time: 2026-07-30
@description: fastAPI inference endpoint for Inpaint-Anything (inpainting model only)
"""

import os
import secrets
import uuid
from io import BytesIO
import shutil
import glob
import asyncio
import datetime
import torch
import uvicorn
import logging

from contextlib import asynccontextmanager
from functools import lru_cache
from fastapi import FastAPI, File, UploadFile, HTTPException, status, Form, Depends, Header, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional, Literal

from lama_inpaint import inpaint_img_with_builded_lama, build_lama_model
from utils import load_img_to_array, dilate_mask, save_array_to_img
from utils.mask_processing import crop_for_filling_pre, crop_for_filling_post


LOGGER = logging.getLogger(__name__)


@torch.no_grad()
def run_inpaint(
    model: torch.nn.Module,
    image_bytes: bytes,
    mask_paths: List[str],
    save_path: str,
    crop_size: int,
    device: str,
) -> None:
    """CPU/GPU-heavy inpaint work. Must run off the asyncio event loop."""
    img = load_img_to_array(BytesIO(image_bytes))
    for mask_path in mask_paths:
        mask = dilate_mask(load_img_to_array(mask_path), 15)
        image_crop, mask_crop = crop_for_filling_pre(img, mask, crop_size=crop_size)
        image_crop_filled = inpaint_img_with_builded_lama(
            model, image_crop, mask_crop, device=device
        )
        img = crop_for_filling_post(img, mask, image_crop_filled, crop_size=crop_size)
    save_array_to_img(img, save_path)

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="INPAINT_",
        validate_default=True,
        frozen=True,
    )

    device: Literal["cuda", "cpu"] = "cuda"
    config_dir: str = "lama/configs/prediction/default.yaml"
    ckpt_dir: str = "pretrained_models/big-lama/"
    file_ttl: int = 600 # seconds
    output_dir: str = "data/"

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
                'DINO_SECRET_KEY is required when DINO_DEV_MODE is not enabled. '
                'Set DINO_SECRET_KEY in the environment or .env file.'
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == 'cuda' and not torch.cuda.is_available():
        LOGGER.info('CUDA is not available, using CPU instead')
        return 'cpu'
    return requested_device


def build_model_from_settings(config_dir: str, ckpt_dir: str, device: str):
    model = build_lama_model(config_dir, ckpt_dir, device)
    model.eval()
    LOGGER.info(f"Model loaded with {sum(p.numel() for p in model.parameters())} parameters")
    return model

async def ttl_clean_up_loop(settings: BaseSettings = Depends(get_settings), interval: int = 600):
    while True:
        await asyncio.sleep(interval)
        try:
            log = await asyncio.to_thread(clean_up_local_files, settings)
            LOGGER.info(f"TTL File Clean Up: Total Files {log['total_files']}, Deleted Files {log['deleted_files']}")
        except Exception as e:
            LOGGER.exception(f"Error cleaning up local files: {e}")
            continue


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    os.makedirs(settings.output_dir, exist_ok=True)
    app.state.device = resolve_device(settings.device)
    model = build_model_from_settings(settings.config_dir, settings.ckpt_dir, app.state.device)
    app.state.model = model
    app.state.settings = settings
    # Serialize GPU inference: one shared LaMa model is not safely concurrent.
    app.state.infer_lock = asyncio.Lock()

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


def get_infer_lock(request: Request) -> asyncio.Lock:
    return request.app.state.infer_lock


def validate_api_key(
    x_api_key: str = Header(..., alias="X-API-KEY"),
    settings: Settings = Depends(get_settings),
) -> None:
    expected_key = settings.secret_key
    if expected_key is None or len(expected_key) != len(x_api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    if not secrets.compare_digest(x_api_key, expected_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


def validate_job_id(job_id: str) -> None:
    if len(job_id) != 32 or any(c not in '0123456789abcdef' for c in job_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Invalid job_id')


app = FastAPI(
    title="Inpaint-Anything",
    description="Inference endpoint for Inpaint-Anything (inpainting model only)",
    lifespan=lifespan,
)


@app.get('/health')
def health(request: Request):
    model_loaded = hasattr(request.app.state, 'model') and request.app.state.model is not None
    body = {'status': 'healthy' if model_loaded else 'unhealthy', 'model_loaded': model_loaded}
    if not model_loaded:
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=body)
    return body



@app.post('/predict', dependencies=[Depends(validate_api_key)])
async def predict(
    image: UploadFile = File(...),
    mask_dir: str = Form(...),
    crop_size: int = Form(default=512),
    model: torch.nn.Module = Depends(get_model),
    device: str = Depends(get_device),
    settings: Settings = Depends(get_settings),
    infer_lock: asyncio.Lock = Depends(get_infer_lock),
):
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty image upload")

    mask_paths = sorted(glob.glob(os.path.join(mask_dir, 'mask_*.png')))
    if len(mask_paths) == 0:
        LOGGER.error(f"No valid mask files found in {mask_dir}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No valid mask files found")

    job_id = uuid.uuid4().hex
    job_dir = os.path.join(settings.output_dir, job_id)
    os.makedirs(job_dir, exist_ok=True)
    save_path = os.path.join(job_dir, 'inpainted.png')

    try:
        # Hold the lock only around inference so /health and downloads stay responsive.
        async with infer_lock:
            await asyncio.to_thread(
                run_inpaint,
                model,
                image_bytes,
                mask_paths,
                save_path,
                crop_size,
                device,
            )
    except Exception as e:
        LOGGER.exception(f"Prediction failed for job {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inpainting failed",
        ) from e

    return {
        'job_id': job_id,
        'image_url': f'/jobs/{job_id}/inpainted.png',
    }


@app.get('/jobs/{job_id}/inpainted.png', dependencies=[Depends(validate_api_key)], response_class=FileResponse)
def download_image(job_id: str, settings: Settings = Depends(get_settings)):
    validate_job_id(job_id)

    image_path = os.path.join(settings.output_dir, job_id, "inpainted.png")
    if not os.path.isfile(image_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Inpainted image not found')

    return FileResponse(image_path, media_type='image/png', filename='inpainted.png')    


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
            LOGGER.warning(f"PermissionError when deleting file at {file_path}")
        log["total_files"] += 1
    return log


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)