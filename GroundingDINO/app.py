"""
@author: Zhening Hu
@time: 2026-07-27
@description: Inference endpoint for GroundingDINO
"""

import os
import secrets
import uuid
from io import BytesIO
import shutil
import json

import asyncio
import datetime
import torch
import uvicorn
import logging
from contextlib import asynccontextmanager
from functools import lru_cache
from argparse import Namespace, ArgumentParser
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form, status, Depends, Header
from fastapi.responses import FileResponse, JSONResponse
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Literal, Optional
from PIL import UnidentifiedImageError

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from app_utils import load_image, get_grounding_output, plot_boxes_to_image, save_boxes_to_json


LOGGER = logging.getLogger(__name__)


def parse_token_spans(raw: str) -> List[List[List[int]]]:
    """Parse multipart form JSON into GroundingDINO token_spans.

    Expected shape: [[[start, end], ...], ...] — non-empty list of phrases,
    each a non-empty list of [start, end] character offsets into the caption.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Invalid token_spans - must be valid JSON',
        ) from exc

    if not isinstance(data, list) or not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Invalid token_spans - must be a non-empty JSON list of phrases',
        )

    for phrase in data:
        if not isinstance(phrase, list) or not phrase:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Invalid token_spans - each phrase must be a non-empty list of [start, end] spans',
            )
        for span in phrase:
            if (
                not isinstance(span, (list, tuple))
                or len(span) != 2
                or not all(isinstance(x, int) and not isinstance(x, bool) for x in span)
            ):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail='Invalid token_spans - each span must be [start, end] integers',
                )
            start, end = span
            if start < 0 or end < 0 or start > end:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail='Invalid token_spans - require 0 <= start <= end',
                )

    return data


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix='DINO_',
        env_file='.env',
        validate_default=True,
        frozen=True,
    )

    device: Literal['cpu', 'cuda'] = 'cuda'
    config_dir: str = 'groundingdino/config/GroundingDINO_SwinT_OGC.py'
    ckpt_dir: str = 'weights/groundingdino_swint_ogc.pth'
    # Docker Compose mounts ./data -> /data; override with DINO_OUTPUT_DIR.
    output_dir: str = 'data'
    file_ttl: int = 600  # seconds

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


def build_model_from_settings(cfg_path: str, ckpt_path: str, device: str):
    args = SLConfig.fromfile(cfg_path)
    args.device = resolve_device(device)
    model = build_model(args)
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    load_res = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    LOGGER.info(f'Model loaded with {load_res}')
    _ = model.eval()
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
    app.state.settings = settings
    app.state.device = resolve_device(settings.device)
    app.state.model = build_model_from_settings(settings.config_dir, settings.ckpt_dir, app.state.device)

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
    if expected_key is None or len(expected_key) != len(x_api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    if not secrets.compare_digest(x_api_key, expected_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


def validate_job_id(job_id: str) -> None:
    if len(job_id) != 32 or any(c not in '0123456789abcdef' for c in job_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Invalid job_id')


app = FastAPI(
    title='GroundingDINO',
    description='Inference endpoint for serving GroundingDINO predictions',
    lifespan=lifespan
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
    text_prompt: str = Form(...),
    box_threshold: float = Form(default=0.3),
    text_threshold: float = Form(default=0.25),
    token_spans: Optional[str] = Form(default=None),
    visualize: bool = Form(default=False),
    model: torch.nn.Module = Depends(get_model),
    device: str = Depends(get_device),
    settings: Settings = Depends(get_settings),
):
    if image.content_type and not image.content_type.startswith('image/'):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid image content type - must be an image")

    image_bytes = image.file.read()
    if not image_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty image upload")

    try:
        image_pil, image_tensor = load_image(image_bytes)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image file: {exc}",
        ) from exc

    parsed_tokens = None
    if token_spans is not None:
        parsed_tokens = parse_token_spans(token_spans)
        text_threshold = None
        LOGGER.info("Using token_spans. Set the text_threshold to None.")

    job_id = uuid.uuid4().hex
    job_dir = os.path.join(settings.output_dir, job_id)
    os.makedirs(job_dir, exist_ok=True)

    size = image_pil.size
    image_size = [size[1], size[0]]

    boxes_filt, pred_phrases = get_grounding_output(
        model,
        image_tensor,
        text_prompt,
        box_threshold,
        text_threshold,
        device=device,
        token_spans=parsed_tokens,
    )

    save_boxes_to_json(
        boxes_filt,
        pred_phrases,
        image_size,
        job_dir,
        job_id,
    )
    boxes_url = f'/jobs/{job_id}/boxes'

    visualization_url = None
    if visualize:
        pred_dict = {
            "boxes": boxes_filt,
            "size": image_size,
            "labels": pred_phrases,
        }
        image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
        image_with_box.save(os.path.join(job_dir, f'{job_id}_pred.jpg'))
        visualization_url = f'/jobs/{job_id}/visualization'

    return {
        'job_id': job_id,
        'boxes_url': boxes_url,
        'visualization_url': visualization_url,
        'num_detections': len(pred_phrases),
    }


@app.get('/jobs/{job_id}/boxes', dependencies=[Depends(validate_api_key)], response_class=FileResponse)
def download_boxes(job_id: str, settings: Settings = Depends(get_settings)):
    validate_job_id(job_id)

    boxes_name = f'{job_id}_boxes.json'
    boxes_path = os.path.join(settings.output_dir, job_id, boxes_name)
    if not os.path.isfile(boxes_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Boxes file not found')

    return FileResponse(boxes_path, media_type='application/json', filename=boxes_name)


@app.get('/jobs/{job_id}/visualization', dependencies=[Depends(validate_api_key)], response_class=FileResponse)
def download_visualization(job_id: str, settings: Settings = Depends(get_settings)):
    validate_job_id(job_id)

    image_name = f'{job_id}_pred.jpg'
    image_path = os.path.join(settings.output_dir, job_id, image_name)
    if not os.path.isfile(image_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Visualization file not found')

    return FileResponse(image_path, media_type='image/jpeg', filename=image_name)


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
    # hosted on port 8001 to avoid conflicts with LGT
    uvicorn.run(app, host="0.0.0.0", port=8001)