from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
import time
import numpy as np
import requests
from typing import Optional, List, Literal
from urllib.parse import urlparse

# Catches errors from Open3D caused by Chinese text
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from get_pano_masks import (
    DEFAULT_POLL_INTERVAL,
    DEFAULT_POLL_MAX_ATTEMPTS,
    wait_and_fetch_mask_job,
    submit_mask_job,
    load_box_prompts_from_json,
)
from gen_furniture_3d import wait_and_fetch_3d_job, submit_3d_job
from reposition import (
    apply_placement_quality,
    detections_aligned_to_masks,
    json_frame_to_obj3d,
    json_rotation_to_obj3d,
    load_sam3d_metadata,
    mark_quality_rejections,
    merge_sam3d_orientations_into_placements,
    merge_sam3d_scale_into_placements,
    placement_to_obj3d_frame,
    prompt_for_room,
    quality_check_message,
    scale_from_prior,
    size_prior,
    uprightness_metrics,
)
from visualization.compare_layout_furniture import (
    FURNITURE_COLORS,
    ROOM_DEFAULT_COLOR,
    align_furniture_to_room,
    apply_uniform_color,
    capture_fitted_screenshots,
    create_coordinate_frame,
    create_ground_grid,
    layout_json_to_hull,
    load_mesh,
    load_room_mesh,
    save_layout_floorplan,
    set_room_transparency,
    transform_mesh,
)


"""
Run the full pipeline:
    1. GroundingDINO bounding box detection
    2.a SAM object segmentation
    2.b SAM 3D object reconstruction
    3. Inpaint-Anything object inpainting
    4. LGT-Net room layout prediction

Assumptions:
    1. The image is a URL of a panorama accessed via GET method
    2. GroundingDINO, Inpaint-Anything, and LGT-Net are running as standalone services
    3. SAM + SAM3D is running as a FAL service
"""

# GLOBAL CONSTANTS
DEFAULT_POLL_INTERVAL = 5
DEFAULT_POLL_MAX_ATTEMPTS = 20
SAM_BASE_URL = "https://ai-test.aroomy.com/api/sam3d/masks"
SAM3D_BASE_URL = "https://ai-test.aroomy.com/api/sam3d"


"""
HELPER FUNCTIONS
"""

def download_binary(url: str, output_path: str, headers: Optional[dict] = None, max_retries: int = 5) -> None:
    """Download a binary file from a URL to a local path with retries."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, headers=headers or {}, timeout=(30, 120))
            response.raise_for_status()
            with open(output_path, "wb") as f:
                f.write(response.content)
            return
        except (requests.ConnectionError, requests.Timeout,
                requests.exceptions.SSLError) as e:
            last_error = e
            wait = 2 ** attempt
            print(f"  [download] retry {attempt}/{max_retries} in {wait}s: {e}")
            time.sleep(wait)
    raise last_error


def load_image_bytes(image_url_or_path: str) -> tuple[bytes, str, str]:
    """
    Load image bytes from an HTTP(S) URL or a local filesystem path.

    Returns:
        (image_bytes, filename, content_type)
    """
    if image_url_or_path.startswith(("http://", "https://")):
        response = requests.get(image_url_or_path, timeout=60)
        response.raise_for_status()
        filename = os.path.basename(urlparse(image_url_or_path).path) or "pano.jpg"
        content_type = response.headers.get("Content-Type", "image/jpeg")
        if not content_type.startswith("image/"):
            content_type = "image/jpeg"
        return response.content, filename, content_type

    path = os.path.abspath(image_url_or_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    with open(path, "rb") as f:
        image_bytes = f.read()
    filename = os.path.basename(path) or "pano.jpg"
    ext = os.path.splitext(filename)[1].lower()
    content_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(ext, "image/jpeg")
    return image_bytes, filename, content_type


def fetch_boxes_json(
    boxes_url: str,
    service_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict:
    """
    GET GroundingDINO boxes JSON and return the parsed dict (no local file write).

    Suitable for passing directly to ``load_box_prompts_from_json``.

    Args:
        boxes_url:     Absolute boxes URL, or a path like ``/jobs/{id}/boxes``.
        service_url: Required when ``boxes_url`` is a relative path.
        api_key:       Optional X-API-KEY for the DINO service.

    Returns:
        Parsed boxes.json dict (detections / box_prompts / image_size, etc.).
    """
    if boxes_url.startswith("http://") or boxes_url.startswith("https://"):
        url = boxes_url
    else:
        if not service_url:
            raise ValueError(
                "service_url is required when boxes_url is not an absolute URL"
            )
        url = f"{service_url.rstrip('/')}{boxes_url}"

    headers = {}
    if api_key is not None:
        headers["X-API-KEY"] = api_key

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


"""
MODEL RUNNERS
"""

def run_dino(
    image_url: str,
    root_dir: str,
    service_url: str,
    text_prompt: str,
    api_key: Optional[str] = None,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    visualize: bool = False,
    download_boxes: bool = False,
    token_spans: Optional[str] = None,
) -> dict:
    """
    Call GroundingDINO /predict and return the parsed boxes.json dict.

    Args:
        image_url:      Publicly accessible panorama URL (fetched via GET).
        root_dir:       Directory used when optionally saving boxes / visualization.
        dino_base_url:  GroundingDINO service URL (default localhost:8001)
        text_prompt:    Detection prompt, e.g. "sofa. chair. table."
        api_key:        X-API-KEY for the DINO service (optional if service allows).
        box_threshold:  Box confidence threshold.
        text_threshold: Text matching threshold.
        visualize:      If True, request and download the prediction visualization.
        download_boxes: If True, also write boxes.json under root_dir.
        token_spans:    Optional JSON token-spans string for phrase grounding.

    Returns:
        Parsed boxes.json dict formatted for ``load_box_prompts_from_json``.
    """
    service_url = service_url.rstrip("/")
    os.makedirs(root_dir, exist_ok=True)

    headers = {}
    if api_key is not None:
        headers["X-API-KEY"] = api_key

    image_response = requests.get(image_url, timeout=60)
    image_response.raise_for_status()
    image_bytes = image_response.content

    filename = os.path.basename(urlparse(image_url).path) or "pano.jpg"
    content_type = image_response.headers.get("Content-Type", "image/jpeg")
    if not content_type.startswith("image/"):
        content_type = "image/jpeg"

    form_data = {
        "text_prompt": text_prompt,
        "box_threshold": str(box_threshold),
        "text_threshold": str(text_threshold),
        "visualize": str(visualize).lower(),
    }
    if token_spans is not None:
        form_data["token_spans"] = token_spans

    predict_response = requests.post(
        f"{service_url}/predict",
        headers=headers,
        files={"image": (filename, image_bytes, content_type)},
        data=form_data,
        timeout=120,
    )
    predict_response.raise_for_status()
    predict_data = predict_response.json()

    job_id = predict_data.get("job_id")
    if not job_id:
        raise RuntimeError(f"GroundingDINO /predict returned no job_id: {predict_data}")

    boxes_url = predict_data.get("boxes_url") or f"/jobs/{job_id}/boxes"
    boxes = fetch_boxes_json(boxes_url, service_url, api_key)
    for i, detection in enumerate(boxes.get("detections") or []):
        detection["detection_id"] = f"{job_id}:{i:04d}"

    boxes_path = None
    if download_boxes:
        boxes_path = os.path.join(root_dir, f"{job_id}_boxes.json")
        with open(boxes_path, "w", encoding="utf-8") as f:
            json.dump(boxes, f, indent=2)
        print(f"  [DINO] boxes -> {os.path.abspath(boxes_path)}")

    visualization_path = None
    if visualize:
        viz_url = predict_data.get("visualization_url") or f"/jobs/{job_id}/visualization"
        visualization_path = os.path.join(root_dir, f"{job_id}_pred.jpg")
        download_binary(f"{service_url}{viz_url}", visualization_path, headers=headers)
        print(f"  [DINO] viz   -> {os.path.abspath(visualization_path)}")

    return {
        "job_id": job_id,
        "boxes_path": boxes_path,
        "boxes": boxes,
        "visualization_path": visualization_path,
    }


def run_sam(
    image_url: str,
    boxes: dict,
    output_dir: str,
    sam_base_url: str = SAM_BASE_URL,
    boxes_path: Optional[str] = None,
    min_score: Optional[float] = None,
    poll_interval: int = DEFAULT_POLL_INTERVAL,
    poll_max_attempts: int = DEFAULT_POLL_MAX_ATTEMPTS,
) -> dict:
    """
    Local mode: read bounding boxes from a local boxes.json and feed them to the SAM API.

    Args:
        image_url:        Publicly accessible URL of the panorama image.
        boxes_json_path:  Path to local boxes.json (GroundingDINO output).
        sam_base_url:     SAM API base URL.
        output_dir:       Output directory for downloaded masks.
        min_score:        Minimum confidence score filter.
        poll_interval:    Polling interval in seconds.
        poll_max_attempts: Maximum polling attempts.

    Returns:
        SAM result dict.
    """
    print(f"    Loading boxes from {boxes_path} ...")
    if boxes_path is not None:
        box_prompts = load_box_prompts_from_json(boxes_path, min_score=min_score)
    else:
        box_prompts = load_box_prompts_from_json(boxes, min_score=min_score)
    print(f"    Loaded {len(box_prompts)} box prompts for SAM")

    print("\n    Submitting SAM segmentation job ...")
    request_id = submit_mask_job(image_url, sam_base_url, box_prompts)
    result = wait_and_fetch_mask_job(
        request_id, sam_base_url, output_dir, poll_interval, poll_max_attempts
    )

    num_masks = len(result.get("data", {}).get("masks", []))
    print(f"   Submitted {len(box_prompts)} boxes")
    print(f"\n  Received {num_masks} masks from SAM API")
    return result


def run_sam3d(
    image_url: str,
    mask_urls: List[str],
    sam3d_base_url: str,
    output_dir: str,
    furniture_prompt: Optional[str] = 'furniture',
    export_textured_glb: bool = True,
    poll_interval: int = DEFAULT_POLL_INTERVAL,
    poll_max_attempts: int = DEFAULT_POLL_MAX_ATTEMPTS,
):
    """
    Feeds image_url and mask_dir 
    """
    print(f"\n      Generating 3D furniture from {len(mask_urls)} masks")

    print("\n      Submitting SAM3D generation job ...")
    request_id = submit_3d_job(
        image_url=image_url,
        sam3d_base_url=sam3d_base_url,
        mask_urls=mask_urls,
        prompt=furniture_prompt,
        export_textured_glb=export_textured_glb,
    )
    return wait_and_fetch_3d_job(
        request_id, sam3d_base_url, output_dir,
        interval=poll_interval, max_attempts=poll_max_attempts,
    )


def run_inpaint_anything(
    image_url: str,
    mask_dir: str,
    root_dir: str,
    service_url: str,
    crop_size: int = 512,
    num_masks: Optional[int] = None,
    api_key: Optional[str] = None,
) -> dict:
    """
    Call Inpaint-Anything /predict and download the inpainted image to root_dir.

    Args:
        image_url:   Publicly accessible panorama URL (fetched via GET).
        mask_dir:    Local directory containing mask_*.png files to upload.
        root_dir:    Directory used when saving the inpainted image.
        service_url: Inpaint-Anything service URL (default localhost:8002).
        crop_size:   Crop size for inpainting (default 512).
        num_masks:   Optional cap on how many mask_*.png files to upload.
        api_key:     X-API-KEY for the inpaint service.

    Returns:
        Dict with job_id and inpainted_path.
    """
    service_url = service_url.rstrip("/")
    os.makedirs(root_dir, exist_ok=True)

    headers = {}
    if api_key is not None:
        headers["X-API-KEY"] = api_key

    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "mask_*.png")))
    if num_masks is not None:
        mask_paths = mask_paths[:num_masks]
    if not mask_paths:
        raise FileNotFoundError(f"No mask_*.png files found in {os.path.abspath(mask_dir)}")

    image_response = requests.get(image_url, timeout=60)
    image_response.raise_for_status()
    image_bytes = image_response.content

    filename = os.path.basename(urlparse(image_url).path) or "pano.jpg"
    content_type = image_response.headers.get("Content-Type", "image/jpeg")
    if not content_type.startswith("image/"):
        content_type = "image/jpeg"

    files = [("image", (filename, image_bytes, content_type))]
    for mask_path in mask_paths:
        with open(mask_path, "rb") as f:
            files.append(
                ("masks", (os.path.basename(mask_path), f.read(), "image/png"))
            )

    form_data = {
        "crop_size": str(crop_size),
    }

    print(f"  [INPAINT] uploading image + {len(mask_paths)} mask(s) ...")
    predict_response = requests.post(
        f"{service_url}/predict",
        headers=headers,
        files=files,
        data=form_data,
        timeout=300,
    )
    predict_response.raise_for_status()
    predict_data = predict_response.json()

    job_id = predict_data.get("job_id")
    if not job_id:
        raise RuntimeError(f"Inpaint-Anything /predict returned no job_id: {predict_data}")

    image_path_url = predict_data.get("image_url") or f"/jobs/{job_id}/inpainted.png"
    inpainted_path = os.path.join(root_dir, f"{job_id}_inpainted.png")
    download_binary(f"{service_url}{image_path_url}", inpainted_path, headers=headers)
    print(f"  [INPAINT] image -> {os.path.abspath(inpainted_path)}")

    return {
        "job_id": job_id,
        "inpainted_path": inpainted_path,
    }


def run_lgt_net(
    image_url: str,
    root_dir: str,
    service_url: str,
    api_key: Optional[str] = None,
    post_processing: Literal["manhattan", "atalanta", "original"] = "manhattan",
    pre_processing: bool = True,
    output_mesh: bool = True,
    output_point_cloud: bool = False,
    mask_dir: Optional[str] = None,
    mesh_format: str = ".obj",
    room_height: Optional[float] = None,
    room_type: str = "living_room",
) -> dict:
    """Call LGT-Net /predict; optionally upload local mask_*.png for placements."""
    service_url = service_url.rstrip("/")
    os.makedirs(root_dir, exist_ok=True)

    headers = {}
    if api_key is not None:
        headers["X-API-KEY"] = api_key

    image_bytes, filename, content_type = load_image_bytes(image_url)
    files = [("image", (filename, image_bytes, content_type))]
    if mask_dir is not None:
        for mask_path in sorted(glob.glob(os.path.join(mask_dir, "mask_*.png"))):
            with open(mask_path, "rb") as f:
                files.append(("masks", (os.path.basename(mask_path), f.read(), "image/png")))

    form = {
        "post_processing": post_processing,
        "pre_processing": str(pre_processing).lower(),
        "output_mesh": str(output_mesh).lower(),
        "output_point_cloud": str(output_point_cloud).lower(),
        "room_type": room_type,
    }
    if room_height is not None:
        form["room_height"] = str(room_height)

    predict_response = requests.post(
        f"{service_url}/predict",
        headers=headers,
        files=files,
        data=form,
        timeout=900,
    )
    predict_response.raise_for_status()
    predict_data = predict_response.json()

    job_id = predict_data.get("job_id")
    if not job_id:
        raise RuntimeError(f"LGT-Net /predict returned no job_id: {predict_data}")

    print(f"  [LGT] job_id={job_id}")

    coordinates = predict_data.get("coordinates")
    placements = predict_data.get("placements")
    placement_success = int(predict_data.get("placement_success") or 0)
    placement_failure = int(predict_data.get("placement_failure") or 0)

    layout_path = None
    if coordinates is not None:
        layout_path = os.path.join(root_dir, f"{job_id}_layout.json")
        with open(layout_path, "w", encoding="utf-8") as f:
            json.dump(coordinates, f, indent=2)
        cam_h = coordinates.get("cameraHeight")
        layout_h = coordinates.get("layoutHeight")
        print(
            f"  [LGT] layout -> {os.path.abspath(layout_path)} "
            f"(cameraHeight={cam_h}, layoutHeight={layout_h})"
        )

    placements_path = None
    if placements is not None:
        placements_path = os.path.join(root_dir, f"{job_id}_placements.json")
        with open(placements_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "job_id": job_id,
                    "placements": placements,
                    "placement_success": placement_success,
                    "placement_failure": placement_failure,
                },
                f,
                indent=2,
            )
        print(
            f"  [LGT] placements -> {os.path.abspath(placements_path)} "
            f"(success={placement_success}, failure={placement_failure})"
        )

    mesh_path = None
    if output_mesh:
        mesh_url = predict_data.get("mesh_url") or f"/jobs/{job_id}/mesh"
        if not mesh_format.startswith("."):
            mesh_format = f".{mesh_format}"
        mesh_path = os.path.join(root_dir, f"{job_id}_3d{mesh_format}")
        download_binary(f"{service_url}{mesh_url}", mesh_path, headers=headers)
        print(f"  [LGT] mesh -> {os.path.abspath(mesh_path)}")

    point_cloud_path = None
    if output_point_cloud:
        pc_url = predict_data.get("point_cloud_url") or f"/jobs/{job_id}/point_cloud"
        point_cloud_path = os.path.join(root_dir, f"{job_id}_3d.ply")
        download_binary(f"{service_url}{pc_url}", point_cloud_path, headers=headers)
        print(f"  [LGT] point cloud -> {os.path.abspath(point_cloud_path)}")

    return {
        "job_id": job_id,
        "coordinates": coordinates,
        "placements": placements,
        "placement_success": placement_success,
        "placement_failure": placement_failure,
        "placements_path": placements_path,
        "mesh_path": mesh_path,
        "point_cloud_path": point_cloud_path,
        "layout_path": layout_path,
    }


def evaluate_placement_uprightness(placements: Optional[List[dict]]) -> List[dict]:
    """
    Evaluation-only uprightness for each successfully placed object.

    Skipped placements (missing translation or scale) are omitted. Rotation
    matrices still in the xyz2json frame are remapped to create_3d_obj / +Y-up
    before scoring so the metric matches the posed scene.
    """
    report: List[dict] = []
    if not placements:
        return report

    for placement in placements:
        if placement.get("translation") is None or placement.get("scale") is None:
            continue
        rotation = placement.get("rotation") or {}
        matrix = rotation.get("matrix") if isinstance(rotation, dict) else None
        if matrix is None:
            continue

        r = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
        if not (isinstance(rotation, dict) and rotation.get("frame") == "obj3d"):
            r = json_rotation_to_obj3d(r)

        metrics = uprightness_metrics(r)
        report.append(
            {
                "mask": placement.get("mask"),
                "surface": placement.get("surface"),
                "orientation_source": (
                    rotation.get("source") if isinstance(rotation, dict) else None
                ),
                **metrics,
            }
        )
    return report


def _mesh_path_for_mask(furniture_dir: str, mask_name: str) -> Optional[str]:
    """Maps masks to their corresponding 3D mesh files in the furniture directory
        mask_N.png -> furniture_dir/mesh_N.{glb,ply,obj}
     """
    match = re.fullmatch(r"mask_(\d+)\.png", os.path.basename(mask_name), flags=re.IGNORECASE)
    if not match:
        return None
    idx = match.group(1)
    for ext in (".glb", ".ply", ".obj"):
        path = os.path.join(furniture_dir, f"mesh_{idx}{ext}")
        if os.path.isfile(path):
            return path
    return None


def _placement_obj3d_matrix(placement: dict) -> Optional[np.ndarray]:
    """4×4 pose in the obj3d frame from translation / rotation / scale, or None."""
    translation = placement.get("translation")
    if translation is None:
        return None
    rotation = placement.get("rotation") or {}
    matrix = rotation.get("matrix") if isinstance(rotation, dict) else None
    R = np.asarray(matrix, dtype=float) if matrix is not None else None
    if R is not None and rotation.get("frame") != "obj3d":
        _, R = placement_to_obj3d_frame(np.zeros(3), R)
    transform = np.eye(4)
    transform[:3, :3] = (np.eye(3) if R is None else R) * float(
        (placement.get("scale") or {}).get("factor", 1.0)
    )
    transform[:3, 3] = json_frame_to_obj3d(np.asarray(translation, dtype=float))
    return transform


def export_placed_furniture_assets(
    placements: List[dict],
    furniture_dir: str,
    output_dir: str,
    mask_dir: Optional[str] = None,
) -> List[dict]:
    """Export QC-approved meshes with final scale, rotation, and translation."""
    import trimesh

    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    assets = []
    for i, placement in enumerate(placements or []):
        mask = placement.get("mask") or f"mask_{i}.png"
        detection_id = placement.get("detection_id") or f"detection_{i:04d}"
        source = _mesh_path_for_mask(furniture_dir, mask)
        accepted = bool(
            source and placement.get("translation") is not None
            and not placement.get("error")
            and (placement.get("quality") or {}).get("keep", True)
        )
        quality = placement.get("quality") or {}
        transform = _placement_obj3d_matrix(placement)
        asset = {
            "detection_id": detection_id,
            "mask_index": placement.get("mask_index", i),
            "mask": os.path.abspath(os.path.join(mask_dir, mask)) if mask_dir else mask,
            "label": placement.get("label"),
            "raw_label": (placement.get("dino") or {}).get("raw_label"),
            "score": (placement.get("dino") or {}).get("score"),
            "accepted": accepted,
            "sam3d_mesh": os.path.abspath(source) if source else None,
            "posed_mesh": None,
            "surface": placement.get("surface"),
            "translation": placement.get("translation"),
            "rotation": placement.get("rotation"),
            "scale": placement.get("scale"),
            "matrix": None if transform is None else transform.tolist(),
            "quality": quality,
            "quality_message": quality.get("message", quality_check_message(quality.get("fails"))),
            "error": placement.get("error"),
        }
        if accepted and transform is not None:
            scene = trimesh.load_scene(source, process=False)
            scene.apply_transform(transform)
            if placement.get("surface") != "wall":
                floor_shift = np.eye(4)
                floor_shift[1, 3] = float(transform[1, 3] - scene.bounds[0, 1])
                scene.apply_transform(floor_shift)
            safe_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", detection_id)
            posed_path = os.path.abspath(os.path.join(output_dir, f"{safe_id}.glb"))
            scene.export(posed_path, file_type="glb")
            asset["posed_mesh"] = posed_path
            asset["translation_obj3d"] = transform[:3, 3].tolist()
        assets.append(asset)
    return assets


def load_pipeline_config(config_path: str) -> dict:
    """
    Load pipeline hyperparameters from a JSON file.

    Expected top-level keys: ``image_url``, ``root_dir``, ``api_key``,
    plus per-model sections ``dino``, ``sam``, ``sam3d``, ``inpaint``,
    ``lgt_net``, and ``visualize``. Missing sections default to ``{}``.
    """
    path = os.path.abspath(config_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Pipeline config not found: {path}")
    with open(path, encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Pipeline config must be a JSON object, got {type(config).__name__}")
    return config


def run_pipeline_from_config(config: dict) -> dict:
    """
    Run the full DINO → SAM → SAM3D → Inpaint → LGT → visualize pipeline
    using hyperparameters from ``load_pipeline_config`` (or an equivalent dict).
    """
    image_url = config.get("image_url")
    if not image_url:
        raise ValueError("config.image_url is required")

    root_dir = config.get("root_dir") or "."
    api_key = config.get("api_key")
    room_type = config.get("room_type", "living_room")
    dino_cfg = config.get("dino") or {}
    sam_cfg = config.get("sam") or {}
    sam3d_cfg = config.get("sam3d") or {}
    inpaint_cfg = config.get("inpaint") or {}
    lgt_cfg = config.get("lgt_net") or {}
    viz_cfg = config.get("visualize") or {}

    os.makedirs(root_dir, exist_ok=True)

    dino_result = run_dino(
        image_url=image_url,
        root_dir=root_dir,
        service_url=dino_cfg.get("service_url", "http://localhost:8001"),
        text_prompt=dino_cfg.get("text_prompt") or prompt_for_room(room_type),
        api_key=dino_cfg.get("api_key", api_key),
        box_threshold=float(dino_cfg.get("box_threshold", 0.3)),
        text_threshold=float(dino_cfg.get("text_threshold", 0.25)),
        visualize=bool(dino_cfg.get("visualize", False)),
        download_boxes=bool(dino_cfg.get("download_boxes", False)),
        token_spans=dino_cfg.get("token_spans"),
    )

    mask_dir = os.path.join(root_dir, "masks")
    shutil.rmtree(mask_dir, ignore_errors=True)
    sam_result = run_sam(
        image_url=image_url,
        boxes=dino_result["boxes"],
        output_dir=mask_dir,
        sam_base_url=sam_cfg.get("service_url", SAM_BASE_URL),
        boxes_path=dino_result.get("boxes_path"),
        min_score=sam_cfg.get("min_score"),
        poll_interval=int(sam_cfg.get("poll_interval", DEFAULT_POLL_INTERVAL)),
        poll_max_attempts=int(sam_cfg.get("poll_max_attempts", DEFAULT_POLL_MAX_ATTEMPTS)),
    )
    aligned_detections = detections_aligned_to_masks(
        dino_result["boxes"], min_score=sam_cfg.get("min_score")
    )
    masks = sam_result.get("data", {}).get("masks", [])
    for i, mask in enumerate(masks):
        if i < len(aligned_detections):
            mask["detection_id"] = aligned_detections[i]["detection_id"]
        mask["mask_index"] = i
        mask["mask_name"] = f"mask_{i}.png"
    print(f"  [SAM] masks saved under {os.path.abspath(mask_dir)}")
    print(
        f"  [SAM] request complete: "
        f"{len(sam_result.get('data', {}).get('masks', []))} mask(s)"
    )

    mask_urls = [mask["url"] for mask in masks]
    object_dir = os.path.join(root_dir, "objects")
    shutil.rmtree(object_dir, ignore_errors=True)
    sam3d_result = run_sam3d(
        image_url=image_url,
        mask_urls=mask_urls,
        sam3d_base_url=sam3d_cfg.get("service_url", SAM3D_BASE_URL),
        output_dir=object_dir,
        furniture_prompt=sam3d_cfg.get("furniture_prompt", "furniture"),
        export_textured_glb=bool(sam3d_cfg.get("export_textured_glb", True)),
        poll_interval=int(sam3d_cfg.get("poll_interval", 15)),
        poll_max_attempts=int(sam3d_cfg.get("poll_max_attempts", 120)),
    )
    print(
        f"    [SAM3D] request complete: "
        f"{len(sam3d_result.get('data', {}).get('individual_glbs', []))} object(s) saved"
    )

    inpaint_result = run_inpaint_anything(
        image_url=image_url,
        mask_dir=mask_dir,
        root_dir=root_dir,
        service_url=inpaint_cfg.get("service_url", "http://localhost:8002"),
        crop_size=int(inpaint_cfg.get("crop_size", 512)),
        num_masks=inpaint_cfg.get("num_masks"),
        api_key=inpaint_cfg.get("api_key", api_key),
    )
    print(
        f"    [INPAINT] request complete: {inpaint_result.get('job_id')} "
        f"inpainted image saved"
    )

    upload_masks = bool(lgt_cfg.get("upload_masks", True))
    lgt_net_result = run_lgt_net(
        image_url=inpaint_result.get("inpainted_path"),
        root_dir=os.path.join(root_dir, "lgt_net"),
        service_url=lgt_cfg.get("service_url", "http://localhost:8000"),
        api_key=lgt_cfg.get("api_key", api_key),
        post_processing=lgt_cfg.get("post_processing", "manhattan"),
        pre_processing=bool(lgt_cfg.get("pre_processing", True)),
        output_mesh=bool(lgt_cfg.get("output_mesh", True)),
        output_point_cloud=bool(lgt_cfg.get("output_point_cloud", False)),
        mask_dir=mask_dir if upload_masks else None,
        mesh_format=lgt_cfg.get("mesh_format", ".obj"),
        room_height=lgt_cfg.get("room_height"),
        room_type=room_type,
    )
    print(
        f"    [LGT-Net] request complete: {lgt_net_result.get('job_id')} "
        f"layout mesh saved "
        f"(placement success={lgt_net_result.get('placement_success', 0)}, "
        f"failure={lgt_net_result.get('placement_failure', 0)})"
    )

    # Hybrid pose: LGT translation + SAM3D upright/yaw + SAM3D native height.
    metadata_path = os.path.join(object_dir, "metadata.json")
    has_meta = os.path.isfile(metadata_path)
    placements = lgt_net_result.get("placements")
    if placements:
        if has_meta:
            placements = merge_sam3d_orientations_into_placements(
                placements, metadata_path
            )
        placements = merge_sam3d_scale_into_placements(
            placements,
            furniture_dir=object_dir,
            metadata=metadata_path if has_meta else None,
            mask_dir=mask_dir,
            layout=lgt_net_result.get("coordinates"),
            boxes=dino_result.get("boxes"),
            min_score=sam_cfg.get("min_score"),
        )
        scaled = [
            p for p in placements
            if (p.get("scale") or {}).get("method") in {
                "size_prior",
                "size_prior_moved",
                "class_typical",
            }
        ]
        if scaled:
            mean_f = float(np.mean([float(p["scale"]["factor"]) for p in scaled]))
            n_clip = sum(
                1
                for p in scaled
                if p["scale"]["method"] == "size_prior_moved"
            )
            print(
                f"    [SCALE] metric priors on {len(scaled)}/{len(placements)} "
                f"(mean factor={mean_f:.3f}, moved-inward={n_clip})"
            )
        lgt_net_result = {**lgt_net_result, "placements": placements}

    # Plan A + B + C quality: label/surface, layout geometry, inter-object clip.
    # QC rejects are folded into record['error'] and placement_success/failure.
    quality_placements = apply_placement_quality(
        lgt_net_result.get("placements"),
        boxes=dino_result.get("boxes"),
        layout=lgt_net_result.get("coordinates"),
        min_score=sam_cfg.get("min_score"),
        drop_rejected=False,
    )
    quality_placements, placement_success, placement_failure = mark_quality_rejections(
        quality_placements
    )
    quality_rejected = sum(
        1
        for p in quality_placements
        if not (p.get("quality") or {}).get("keep", True)
    )
    lgt_net_result = {
        **lgt_net_result,
        "placements": quality_placements,
        "placement_success": placement_success,
        "placement_failure": placement_failure,
        "quality_rejected": quality_rejected,
    }
    if quality_rejected:
        print(
            f"    [QC] rejected {quality_rejected}/"
            f"{len(quality_placements)} placement(s) "
            f"(success={placement_success}, failure={placement_failure})"
        )
        for p in quality_placements:
            q = p.get("quality") or {}
            if q.get("keep", True):
                continue
            print(
                f"    [QC]   reject {p.get('mask')} "
                f"label={p.get('label')} surface={p.get('surface')} "
                f"error={p.get('error')}"
            )

    uprightness_report = evaluate_placement_uprightness(
        lgt_net_result.get("placements")
    )
    lgt_net_result = {
        **lgt_net_result,
        "uprightness": uprightness_report,
    }

    placements_path = lgt_net_result.get("placements_path")
    if placements_path and lgt_net_result.get("placements") is not None:
        with open(placements_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "job_id": lgt_net_result.get("job_id"),
                    "placements": lgt_net_result.get("placements"),
                    "placement_success": placement_success,
                    "placement_failure": placement_failure,
                    "quality_rejected": quality_rejected,
                    "uprightness": uprightness_report,
                    "orientation_source": (
                        "sam3d_hybrid"
                        if os.path.isfile(metadata_path)
                        else "lgt"
                    ),
                    "scale_source": "metric_size_prior",
                },
                f,
                indent=2,
            )

    if uprightness_report:
        mean_cos = float(
            np.mean([row["cos"] for row in uprightness_report])
        )
        print(
            f"    [LGT-Net] uprightness: n={len(uprightness_report)} "
            f"mean_cos={mean_cos:.4f}"
        )

    posed_dir = os.path.join(root_dir, "posed_objects")
    assets = export_placed_furniture_assets(
        lgt_net_result.get("placements") or [],
        furniture_dir=object_dir,
        output_dir=posed_dir,
        mask_dir=mask_dir,
    )
    assets_by_id = {asset["detection_id"]: asset for asset in assets}
    detections = []
    for i, detection in enumerate(aligned_detections):
        detection_id = detection["detection_id"]
        asset = assets_by_id.get(detection_id, {})
        mask = masks[i] if i < len(masks) else {}
        detections.append({
            "detection_id": detection_id,
            "detection_index": next(
                (j for j, row in enumerate(dino_result["boxes"].get("detections", []))
                 if row.get("detection_id") == detection_id),
                i,
            ),
            "mask_index": i,
            "label": detection.get("label"),
            "score": detection.get("score"),
            "box": detection.get("box"),
            "mask_url": mask.get("url"),
            "mask": os.path.abspath(os.path.join(mask_dir, f"mask_{i}.png")),
            "sam3d_mesh": asset.get("sam3d_mesh"),
            "posed_mesh": asset.get("posed_mesh"),
            "accepted": asset.get("accepted", False),
            "matrix": asset.get("matrix"),
            "quality_message": asset.get("quality_message", ""),
        })
    manifest = {
        "schema_version": 1,
        "pipeline_job_id": lgt_net_result.get("job_id"),
        "room_type": room_type,
        "room": {
            "layout_json": (
                os.path.abspath(lgt_net_result["layout_path"])
                if lgt_net_result.get("layout_path") else None
            ),
            "layout_mesh": (
                os.path.abspath(lgt_net_result["mesh_path"])
                if lgt_net_result.get("mesh_path") else None
            ),
            "height_m": (lgt_net_result.get("coordinates") or {}).get("layoutHeight"),
        },
        "sam3d": {
            "metadata": os.path.abspath(metadata_path) if has_meta else None,
            "combined_mesh": (
                os.path.abspath(os.path.join(object_dir, "combined_scene.glb"))
                if os.path.isfile(os.path.join(object_dir, "combined_scene.glb")) else None
            ),
        },
        "detections": detections,
        "objects": assets,
    }
    manifest_path = os.path.abspath(os.path.join(root_dir, "assets_manifest.json"))
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"    [ASSETS] manifest -> {manifest_path}")

    viz_geometries = None
    if bool(viz_cfg.get("enabled", True)):
        viz_geometries = visualize_placed_furniture(
            lgt_net_result=lgt_net_result,
            furniture_dir=object_dir,
            show=bool(viz_cfg.get("show", True)),
            show_ground_grid=bool(viz_cfg.get("show_ground_grid", True)),
            show_coordinate_frame=bool(viz_cfg.get("show_coordinate_frame", True)),
            room_transparency=float(viz_cfg.get("room_transparency", 0.4)),
            save_screenshot=viz_cfg.get("save_screenshot"),
            screenshot_zoom=float(viz_cfg.get("screenshot_zoom", 1.0)),
            skip_rejected=bool(viz_cfg.get("skip_rejected", True)),
            verbose=bool(viz_cfg.get("verbose", True)),
        )

    return {
        "dino": dino_result,
        "sam": sam_result,
        "sam3d": sam3d_result,
        "inpaint": inpaint_result,
        "lgt_net": lgt_net_result,
        "mask_dir": mask_dir,
        "object_dir": object_dir,
        "placement_success": lgt_net_result.get("placement_success", 0),
        "placement_failure": lgt_net_result.get("placement_failure", 0),
        "quality_rejected": quality_rejected,
        "uprightness": uprightness_report,
        "assets": manifest,
        "assets_manifest_path": manifest_path,
        "visualize": viz_geometries,
    }


def visualize_placed_furniture(
    lgt_net_result: dict,
    furniture_dir: str,
    show: bool = True,
    show_ground_grid: bool = True,
    show_coordinate_frame: bool = True,
    room_transparency: float = 0.4,
    save_screenshot: Optional[str] = None,
    screenshot_zoom: float = 1.0,
    snap_to_floor: bool = True,
    skip_rejected: bool = True,
    verbose: bool = True,
) -> Optional[List]:
    """
    Apply LGT-Net placement translation and orientation to SAM3D meshes and
    visualize them with the predicted layout mesh.

    Orientation preference:
      1) ``placement.rotation.source == "sam3d"`` (already in obj3d frame)
      2) else ``furniture_dir/metadata.json`` SAM3D quaternion hybrid
      3) else wall-heuristic rotation from LGT (xyz2json → obj3d remap)

    Translation always comes from LGT and is remapped xyz2json → obj3d.
    Uniform ``placement.scale.factor`` is applied when present. When a local
    GLB is available, the factor is recomputed from the mesh AABB height and
    stored ``target_height`` (or α / R) so pose-time assumed heights are not
    double-applied.
    When ``snap_to_floor`` is True (default), freestanding meshes are raised /
    lowered after posing so their AABB bottom sits on the placement floor
    plane (SAM3D pivots are usually mesh-centered).
    When ``skip_rejected`` is True (default), placements with
    ``quality.keep is False`` are not drawn.

    Uses helpers from ``visualization.compare_layout_furniture``.

    Args:
        lgt_net_result: Output of ``run_lgt_net`` (needs ``mesh_path`` + ``placements``).
        furniture_dir: Directory with ``mesh_N.glb`` files matching ``mask_N.png``.
        show: If True, open an interactive Open3D window.
        show_ground_grid: Draw a y=0 reference grid.
        show_coordinate_frame: Draw RGB axes.
        room_transparency: Room wall opacity blend in ``[0, 1]``.
        save_screenshot: Optional path to capture a still after display setup.
        screenshot_zoom: Multiplier on default capture zoom; smaller is farther.
        snap_to_floor: Snap freestanding AABB bottoms to the floor after pose.
        skip_rejected: Skip placements rejected by quality control.
        verbose: Print load / placement progress.

    Returns:
        List of Open3D geometries used in the scene, or None if nothing loaded.
    """
    import open3d as o3d

    room_path = lgt_net_result.get("mesh_path")
    placements = lgt_net_result.get("placements") or []
    if not placements and lgt_net_result.get("placements_path"):
        with open(lgt_net_result["placements_path"], encoding="utf-8") as f:
            payload = json.load(f)
        placements = payload.get("placements") or []

    if not room_path and not placements:
        raise ValueError("lgt_net_result has neither mesh_path nor placements")

    metadata_path = os.path.join(furniture_dir, "metadata.json")
    sam3d_meta = None
    if os.path.isfile(metadata_path):
        try:
            sam3d_meta = load_sam3d_metadata(metadata_path)
            placements = merge_sam3d_orientations_into_placements(placements, sam3d_meta)
            if verbose:
                print(
                    f"  [VIZ] SAM3D hybrid orientations from {os.path.abspath(metadata_path)}"
                )
        except Exception as exc:
            if verbose:
                print(f"  [VIZ] SAM3D metadata unused ({exc})")

    geometries: List = []

    layout = lgt_net_result.get("coordinates")
    if not layout and lgt_net_result.get("layout_path"):
        layout_path = lgt_net_result["layout_path"]
        if os.path.isfile(layout_path):
            with open(layout_path, encoding="utf-8") as f:
                layout = json.load(f)

    try:
        placements = merge_sam3d_scale_into_placements(
            placements,
            furniture_dir=furniture_dir,
            metadata=sam3d_meta,
            layout=layout,
        )
    except Exception as exc:
        if verbose:
            print(f"  [VIZ] SAM3D scale unused ({exc})")

    room_mesh = None
    if layout:
        try:
            room_mesh, edges = layout_json_to_hull(layout)
            if room_transparency < 1.0:
                set_room_transparency(room_mesh, alpha=max(room_transparency, 0.35))
            geometries.append(room_mesh)
            geometries.append(edges)
            if verbose:
                print("  [VIZ] layout hull from LGT JSON (exterior walls)")
        except Exception as exc:
            if verbose:
                print(f"  [VIZ] layout hull unused ({exc})")
            room_mesh = None
    if room_mesh is None and room_path:
        if verbose:
            print(f"\n  [VIZ] loading layout mesh: {os.path.abspath(room_path)}")
        room_mesh = load_room_mesh(room_path, verbose=verbose)
        if room_mesh is not None:
            if not room_mesh.has_textures() and not room_mesh.has_vertex_colors():
                apply_uniform_color(room_mesh, ROOM_DEFAULT_COLOR)
            if room_transparency < 1.0:
                set_room_transparency(room_mesh, alpha=room_transparency)
            geometries.append(room_mesh)

    placed = 0
    for i, placement in enumerate(placements):
        mask_name = placement.get("mask") or f"mask_{i}.png"
        quality = placement.get("quality") or {}
        if skip_rejected and quality.get("keep") is False:
            if verbose:
                fails = ",".join(quality.get("fails") or []) or "quality"
                label = placement.get("label") or "?"
                print(f"  [VIZ] skip {mask_name}: rejected ({fails}, label={label})")
            continue

        translation = placement.get("translation")
        rotation = placement.get("rotation") or {}
        if translation is None:
            if verbose:
                print(f"  [VIZ] skip {mask_name}: no translation")
            continue

        mesh_path = _mesh_path_for_mask(furniture_dir, mask_name)
        if mesh_path is None:
            if verbose:
                print(f"  [VIZ] skip {mask_name}: no mesh in {furniture_dir}")
            continue

        mesh = load_mesh(mesh_path, verbose=verbose)
        if mesh is None:
            continue

        if not mesh.has_textures():
            color = FURNITURE_COLORS[placed % len(FURNITURE_COLORS)]
            apply_uniform_color(mesh, color)

        rot_matrix = None
        if isinstance(rotation, dict) and rotation.get("matrix") is not None:
            rot_matrix = np.asarray(rotation["matrix"], dtype=np.float64)
        translation_vec = np.asarray(translation, dtype=np.float64).reshape(3)

        scale_info = placement.get("scale") or {}
        if not isinstance(scale_info, dict):
            scale_info = {"factor": float(scale_info), "method": "placement"}
        scale_factor = float(scale_info.get("factor", 1.0) or 1.0)
        scale_method = scale_info.get("method") or "sam3d"

        # Recover a missing stored factor directly from local mesh dimensions.
        try:
            extents = mesh.get_axis_aligned_bounding_box().get_extent()
            h_mesh = float(extents[1])
            existing_factor = scale_info.get("factor")
            if h_mesh > 1e-4 and existing_factor is None:
                prior = size_prior(placement.get("label"))
                if prior:
                    scale_factor, dimensions = scale_from_prior(extents, prior)
                    scale_method = "size_prior"
                    scale_info = {
                        **scale_info,
                        "factor": scale_factor,
                        "method": scale_method,
                        "object_height": h_mesh,
                        "target_height": dimensions[2],
                        "target_dimensions": dimensions,
                        "prior_category": prior["category"],
                    }
        except Exception:
            pass

        # Translations are always xyz2json → obj3d. SAM3D rotations are already
        # authored in the obj3d / Open3D Y-up frame; wall heuristics are not.
        translation_vec = json_frame_to_obj3d(translation_vec)
        if rot_matrix is not None and not (
            isinstance(rotation, dict) and rotation.get("frame") == "obj3d"
        ):
            _, rot_matrix = placement_to_obj3d_frame(
                np.zeros(3, dtype=np.float64), rot_matrix
            )

        transform_mesh(
            mesh,
            translation=translation_vec,
            rotation=rot_matrix,
            scale=scale_factor,
        )

        # Placement puts the mesh origin on the floor; SAM3D GLBs are usually
        # centered, so half the body would clip below the floor. Snap the
        # post-pose AABB bottom onto the contact plane for freestanding items.
        snapped = False
        if snap_to_floor and placement.get("surface") != "wall":
            floor_y = float(translation_vec[1])
            before_min = float(mesh.get_axis_aligned_bounding_box().min_bound[1])
            align_furniture_to_room(mesh, floor_y=floor_y)
            after_min = float(mesh.get_axis_aligned_bounding_box().min_bound[1])
            snapped = abs(after_min - before_min) > 1e-6

        geometries.append(mesh)
        placed += 1
        if verbose:
            yaw = rotation.get("yaw") if isinstance(rotation, dict) else None
            src = rotation.get("source") if isinstance(rotation, dict) else None
            yaw_str = f", yaw={yaw:.3f}" if isinstance(yaw, (int, float)) else ""
            src_str = f", ori={src}" if src else ""
            snap_str = ", floor_snap" if snapped else ""
            scale_str = (
                f", scale={scale_factor:.3f}({scale_method})"
                if abs(scale_factor - 1.0) > 1e-3 or scale_method != "sam3d"
                else ""
            )
            print(
                f"  [VIZ] placed {os.path.basename(mesh_path)} "
                f"t={translation_vec.tolist()}{yaw_str}{src_str}{scale_str}{snap_str}"
            )

    if not geometries:
        print("  [VIZ] nothing to visualize")
        return None

    if show_coordinate_frame:
        geometries.append(create_coordinate_frame(size=0.35))

    screenshot_geoms = list(geometries)

    if show_ground_grid:
        origin = np.zeros(3, dtype=np.float64)
        size = 6.0
        if room_mesh is not None:
            box = room_mesh.get_axis_aligned_bounding_box()
            lo = np.asarray(box.min_bound, dtype=np.float64)
            hi = np.asarray(box.max_bound, dtype=np.float64)
            origin = np.array(
                [0.5 * (lo[0] + hi[0]), float(lo[1]), 0.5 * (lo[2] + hi[2])]
            )
            size = float(max(hi[0] - lo[0], hi[2] - lo[2], 2.0) * 1.4)
        geometries.extend(create_ground_grid(size=size, origin=origin))

    if verbose:
        print(
            f"  [VIZ] scene ready: room={'yes' if room_mesh else 'no'}, "
            f"furniture={placed}"
        )

    if show:
        o3d.visualization.draw_geometries(
            geometries,
            window_name="LGT-Net layout × placed furniture",
            width=1280,
            height=720,
            left=50,
            top=50,
            mesh_show_back_face=True,
        )

    if save_screenshot:
        capture_fitted_screenshots(
            screenshot_geoms,
            save_screenshot,
            padding=1.22,
            verbose=verbose,
        )
        if layout:
            dest = save_screenshot
            if os.path.splitext(dest)[1].lower() in {".png", ".jpg", ".jpeg"}:
                dest = os.path.dirname(os.path.abspath(dest))
            elif os.path.basename(os.path.abspath(dest)).lower() == "screenshots":
                dest = os.path.dirname(os.path.abspath(dest))
            fp_path = os.path.join(dest, "floorplan.png")
            try:
                save_layout_floorplan(layout, fp_path)
                if verbose:
                    print(f"  [VIZ] floorplan -> {os.path.abspath(fp_path)}")
            except Exception as exc:
                if verbose:
                    print(f"  [VIZ] floorplan unused ({exc})")

    return geometries


if __name__ == "__main__":
    default_config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "pipeline_config.json",
    )
    parser = argparse.ArgumentParser(
        description="Run full panorama furniture + layout pipeline from JSON config"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=default_config_path,
        help=f"Path to pipeline hyperparameters JSON (default: {default_config_path})",
    )
    parser.add_argument(
        "--image_url",
        type=str,
        default=None,
        help="Override config.image_url (public panorama URL or local path for LGT stage)",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=None,
        help="Override config.root_dir for saving results",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Override config.api_key used by DINO / Inpaint / LGT services",
    )
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    if args.image_url is not None:
        config["image_url"] = args.image_url
    if args.root_dir is not None:
        config["root_dir"] = args.root_dir
    if args.api_key is not None:
        config["api_key"] = args.api_key

    run_pipeline_from_config(config)
