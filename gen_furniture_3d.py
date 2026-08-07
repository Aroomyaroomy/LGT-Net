"""
Bridge script: generate 3D furniture models from segmentation masks via SAM3D 3D API.

Supports two modes:
  1. from_masks — feed existing mask URLs + image URL -> SAM3D 3D API -> .glb files
  2. full_pipeline — GroundingDINO -> SAM masks -> SAM3D 3D (end-to-end)

Usage:
  # Mode 1: from existing mask URLs
  python gen_furniture_3d.py from_masks \\
      --image_url "https://example.com/pano.jpg" \\
      --mask_urls "https://example.com/mask_0.png" "https://example.com/mask_1.png" \\
      --sam3d_base_url "https://ai-test.aroomy.com/api/sam3d" \\
      --output_dir ./furniture

  # Mode 2: full pipeline (DINO -> masks -> 3D)
  python gen_furniture_3d.py full_pipeline \\
      --image_path path/to/pano.jpg \\
      --dino_base_url "http://localhost:8001" \\
      --dino_api_key "your-key" \\
      --text_prompt "sofa. chair. table." \\
      --sam_base_url "https://ai-test.aroomy.com/api/sam3d/masks" \\
      --sam3d_base_url "https://ai-test.aroomy.com/api/sam3d" \\
      --image_url_for_sam "https://example.com/pano.jpg" \\
      --output_dir ./furniture

  Can also be imported as a module:
      from gen_furniture_3d import run_from_masks, run_full_pipeline
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTIVE_STATUSES = {"IN_QUEUE", "QUEUED", "PENDING", "IN_PROGRESS", "PROCESSING", "RUNNING"}
DONE_STATUSES = {"COMPLETED", "SUCCEEDED"}
FAILED_STATUSES = {"FAILED", "ERROR", "CANCELED", "CANCELLED"}

DEFAULT_POLL_INTERVAL = 15  # 3D generation takes longer than masks
DEFAULT_POLL_MAX_ATTEMPTS = 60  # up to 15 minutes


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def download_binary(url: str, output_path: str, max_retries: int = 5) -> None:
    """Download a binary file from a URL to a local path with retries."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, timeout=(30, 120))
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


# ---------------------------------------------------------------------------
# SAM3D 3D API interaction
# ---------------------------------------------------------------------------

def submit_3d_job(
    image_url: str,
    sam3d_base_url: str,
    mask_urls: Optional[List[str]] = None,
    box_prompts: Optional[List[dict]] = None,
    prompt: str = "furniture",
    export_textured_glb: bool = True,
) -> str:
    """
    Submit a 3D furniture generation job to the SAM3D API.

    Priority: mask_urls > box_prompts > prompt (text-only auto-seg).

    Args:
        image_url: Publicly accessible URL of the panorama image.
        sam3d_base_url: SAM3D API base URL, e.g. https://ai-test.aroomy.com/api/sam3d.
        mask_urls: List of mask image URLs (from prior SAM mask step).
        box_prompts: List of bounding box dicts with xMin, yMin, xMax, yMax.
        prompt: Text prompt for auto-segmentation fallback (default "furniture").
        export_textured_glb: Whether to export textured GLB files.

    Returns:
        request_id — used for subsequent status polling and result retrieval.
    """
    payload: dict = {
        "imageUrl": image_url,
        "exportTexturedGlb": export_textured_glb,
    }

    if mask_urls:
        payload["maskUrls"] = mask_urls
    elif box_prompts:
        payload["boxPrompts"] = [
            {
                "xMin": int(b["xMin"]),
                "xMax": int(b["xMax"]),
                "yMin": int(b["yMin"]),
                "yMax": int(b["yMax"]),
            }
            for b in box_prompts
        ]
    else:
        payload["prompt"] = prompt

    response = requests.post(
        f"{sam3d_base_url}/submit",
        json=payload,
        timeout=60,
    )
    if not response.ok:
        print(f"  [SAM3D ERROR] HTTP {response.status_code}: {response.text[:500]}")
    response.raise_for_status()
    data = response.json()
    request_id = data.get("requestId")
    if not request_id:
        raise RuntimeError(f"Submit response missing requestId: {data}")
    return request_id


def poll_3d_job_status(request_id: str, sam3d_base_url: str) -> str:
    """Query the status of a SAM3D 3D generation job."""
    response = requests.get(
        f"{sam3d_base_url}/status",
        params={"requestId": request_id},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    status = data.get("status")
    if not status:
        raise RuntimeError(f"Status response missing status: {data}")
    return status


def get_3d_job_result(request_id: str, sam3d_base_url: str) -> dict:
    """Retrieve the final result of a SAM3D 3D generation job."""
    response = requests.get(
        f"{sam3d_base_url}/result",
        params={"requestId": request_id},
        timeout=30,
    )
    if not response.ok:
        print(f"  [SAM3D ERROR] HTTP {response.status_code}: {response.text[:500]}")
    if response.status_code == 409:
        raise RuntimeError("Job result not ready yet.")
    response.raise_for_status()
    data = response.json()
    if "error" in data:
        raise RuntimeError(f"{data['error']} | Details: {data.get('details')}")
    if "data" not in data:
        raise RuntimeError(f"Malformed result payload: {data}")
    return data


def wait_and_fetch_3d_job(
    request_id: str,
    sam3d_base_url: str,
    output_dir: str,
    interval: int = DEFAULT_POLL_INTERVAL,
    max_attempts: int = DEFAULT_POLL_MAX_ATTEMPTS,
) -> dict:
    """
    Poll until the SAM3D 3D job completes, then download all result files.

    Downloads:
      - individual_glbs: per-object textured GLB files
      - model_glb: combined scene GLB (if present)
      - metadata.json: per-object pose metadata

    Args:
        request_id: SAM3D job request ID.
        sam3d_base_url: SAM3D API base URL.
        output_dir: Output directory (created automatically).
        interval: Polling interval in seconds.
        max_attempts: Maximum number of polling attempts.

    Returns:
        Full SAM3D result response dict.
    """
    os.makedirs(output_dir, exist_ok=True)

    NETWORK_RETRIES = 3
    network_failures = 0

    for attempt in range(1, max_attempts + 1):
        try:
            status = poll_3d_job_status(request_id, sam3d_base_url)
            network_failures = 0  # reset on success
        except (requests.ConnectionError, requests.Timeout) as e:
            network_failures += 1
            if network_failures > NETWORK_RETRIES:
                raise RuntimeError(
                    f"Network error after {NETWORK_RETRIES} retries: {e}"
                ) from e
            print(f"  [SAM3D] network error (retry {network_failures}/{NETWORK_RETRIES}): {e}")
            time.sleep(interval)
            continue

        print(f"  [SAM3D] status={status} (attempt {attempt}/{max_attempts})")

        if status in FAILED_STATUSES:
            raise RuntimeError(
                f"3D job failed with status={status}, requestId={request_id}"
            )

        if status in DONE_STATUSES:
            result = get_3d_job_result(request_id, sam3d_base_url)
            result_data = result["data"]

            # Download individual per-object GLB files
            individual_glbs = result_data.get("individual_glbs", [])
            for idx, glb in enumerate(individual_glbs):
                filename = glb.get("file_name") or f"object_{idx}.glb"
                filepath = os.path.abspath(os.path.join(output_dir, filename))
                download_binary(glb["url"], filepath)
                print(f"  [SAM3D] Downloaded: {filepath}")

            # Download combined model GLB (if present)
            model_glb = result_data.get("model_glb")
            if model_glb and model_glb.get("url"):
                filename = model_glb.get("file_name") or "model_combined.glb"
                filepath = os.path.abspath(os.path.join(output_dir, filename))
                download_binary(model_glb["url"], filepath)
                print(f"  [SAM3D] Downloaded combined model: {filepath}")

            # Save metadata (rotation, scale, translation per object)
            metadata = result_data.get("metadata")
            if metadata:
                metadata_path = os.path.abspath(os.path.join(output_dir, "metadata.json"))
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                print(f"  [SAM3D] Saved metadata: {metadata_path}")

            return result

        if status not in ACTIVE_STATUSES:
            raise RuntimeError(
                f"Unexpected status '{status}' for requestId={request_id}"
            )

        time.sleep(interval)

    raise TimeoutError(
        f"Failed to fetch 3D job result after {max_attempts} attempts "
        f"(requestId={request_id})"
    )


# ---------------------------------------------------------------------------
# High-level entry points
# ---------------------------------------------------------------------------

def run_from_masks(
    image_url: str,
    mask_urls: List[str],
    sam3d_base_url: str,
    output_dir: str,
    prompt: str = "furniture",
    export_textured_glb: bool = True,
    poll_interval: int = DEFAULT_POLL_INTERVAL,
    poll_max_attempts: int = DEFAULT_POLL_MAX_ATTEMPTS,
) -> dict:
    """
    Generate 3D furniture from existing mask URLs.

    Use this when you already have segmentation masks (e.g. from a prior
    ``get_pano_masks.py`` run) and want to turn them into 3D models.

    Args:
        image_url: Publicly accessible URL of the original panorama image.
        mask_urls: List of mask image URLs (each URL should be publicly accessible).
        sam3d_base_url: SAM3D API base URL, e.g. https://ai-test.aroomy.com/api/sam3d.
        output_dir: Output directory for downloaded .glb files and metadata.
        prompt: Text prompt fallback (default "furniture").
        export_textured_glb: Whether to export textured GLB.
        poll_interval: Polling interval in seconds.
        poll_max_attempts: Maximum polling attempts.

    Returns:
        SAM3D result dict.

    Example:
        >>> result = run_from_masks(
        ...     image_url="https://example.com/pano.jpg",
        ...     mask_urls=["https://example.com/mask_0.png"],
        ...     sam3d_base_url="https://ai-test.aroomy.com/api/sam3d",
        ...     output_dir="./furniture",
        ... )
    """
    print(f"\n[from_masks] Generating 3D furniture from {len(mask_urls)} mask(s) ...")
    print(f"  image_url: {image_url}")
    for i, mu in enumerate(mask_urls):
        print(f"  mask_url[{i}]: {mu}")

    print("\n[SAM3D] Submitting 3D generation job ...")
    request_id = submit_3d_job(
        image_url=image_url,
        sam3d_base_url=sam3d_base_url,
        mask_urls=mask_urls,
        prompt=prompt,
        export_textured_glb=export_textured_glb,
    )
    print(f"  [SAM3D] requestId={request_id}")

    return wait_and_fetch_3d_job(
        request_id, sam3d_base_url, output_dir,
        interval=poll_interval, max_attempts=poll_max_attempts,
    )


def run_full_pipeline(
    image_path: str,
    dino_base_url: str,
    dino_api_key: str,
    text_prompt: str,
    sam_base_url: str,
    sam3d_base_url: str,
    output_dir: str,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    poll_interval: int = DEFAULT_POLL_INTERVAL,
    poll_max_attempts: int = DEFAULT_POLL_MAX_ATTEMPTS,
    image_url_for_sam: Optional[str] = None,
    furniture_prompt: str = "furniture",
) -> dict:
    """
    Full pipeline: GroundingDINO -> SAM masks -> SAM3D 3D furniture.

    Steps:
        1. Call GroundingDINO API to detect objects -> boxes
        2. Call SAM mask API to generate segmentation masks -> mask URLs
        3. Call SAM3D 3D API to generate 3D furniture models -> .glb files

    Args:
        image_path: Local panorama image path.
        dino_base_url: GroundingDINO service URL.
        dino_api_key: GroundingDINO API key.
        text_prompt: Detection prompt, e.g. "sofa. chair. table."
        sam_base_url: SAM mask API base URL.
        sam3d_base_url: SAM3D 3D API base URL.
        output_dir: Output directory for masks, boxes, and 3D models.
        box_threshold: GroundingDINO box confidence threshold.
        text_threshold: GroundingDINO text matching threshold.
        poll_interval: Polling interval in seconds.
        poll_max_attempts: Maximum polling attempts.
        image_url_for_sam: Public image URL (required — SAM APIs need a reachable URL).
        furniture_prompt: Text prompt for SAM3D 3D generation (default "furniture").

    Returns:
        SAM3D result dict with extra keys: ``dino_job_id``, ``mask_request_id``.

    Example:
        >>> result = run_full_pipeline(
        ...     image_path="pano.jpg",
        ...     dino_base_url="http://localhost:8001",
        ...     dino_api_key="dev-key",
        ...     text_prompt="sofa. chair. table.",
        ...     sam_base_url="https://ai-test.aroomy.com/api/sam3d/masks",
        ...     sam3d_base_url="https://ai-test.aroomy.com/api/sam3d",
        ...     image_url_for_sam="https://example.com/pano.jpg",
        ...     output_dir="./output",
        ... )
    """
    # Import here to avoid circular dependency; get_pano_masks lives in same dir
    from get_pano_masks import (
        fetch_boxes_from_dino,
        submit_mask_job,
        wait_and_fetch_mask_job,
    )

    image_url = image_url_for_sam
    if not image_url:
        raise ValueError(
            "image_url_for_sam is required for full_pipeline mode. "
            "SAM APIs need a publicly accessible image URL."
        )

    # ---- Step 1: GroundingDINO ----
    print(f"\n{'='*60}")
    print(f"  Step 1/3: GroundingDINO — object detection")
    print(f"{'='*60}")
    print(f"  image:  {image_path}")
    print(f"  prompt: {text_prompt}")

    box_prompts, dino_job_id = fetch_boxes_from_dino(
        dino_base_url, dino_api_key, image_path, text_prompt,
        box_threshold=box_threshold, text_threshold=text_threshold,
    )
    print(f"  [DINO] {len(box_prompts)} objects detected")

    if not box_prompts:
        print("[SKIP] No objects detected, stopping pipeline")
        return {"dino_job_id": dino_job_id, "box_prompts": [], "skipped": True}

    # ---- Step 2: SAM masks ----
    print(f"\n{'='*60}")
    print(f"  Step 2/3: SAM3D — mask generation")
    print(f"{'='*60}")

    mask_output_dir = os.path.join(output_dir, "masks")
    print(f"  [SAM] Submitting mask job with {len(box_prompts)} box(es) ...")
    mask_request_id = submit_mask_job(image_url, sam_base_url, box_prompts)
    print(f"  [SAM] requestId={mask_request_id}")

    mask_result = wait_and_fetch_mask_job(
        mask_request_id, sam_base_url, mask_output_dir,
        interval=poll_interval, max_attempts=poll_max_attempts,
    )

    # Collect mask URLs from result
    mask_urls: List[str] = []
    result_data = mask_result.get("data", {})
    for mask in result_data.get("masks", []):
        mask_urls.append(mask["url"])

    print(f"  [SAM] {len(mask_urls)} mask(s) generated")

    # ---- Step 3: SAM3D 3D generation ----
    print(f"\n{'='*60}")
    print(f"  Step 3/3: SAM3D — 3D furniture generation")
    print(f"{'='*60}")

    furniture_output_dir = os.path.join(output_dir, "furniture")
    print(f"  [SAM3D] Submitting 3D job with {len(mask_urls)} mask(s) ...")
    request_3d_id = submit_3d_job(
        image_url=image_url,
        sam3d_base_url=sam3d_base_url,
        mask_urls=mask_urls,
        prompt=furniture_prompt,
    )
    print(f"  [SAM3D] requestId={request_3d_id}")

    result_3d = wait_and_fetch_3d_job(
        request_3d_id, sam3d_base_url, furniture_output_dir,
        interval=poll_interval, max_attempts=poll_max_attempts,
    )
    result_3d["dino_job_id"] = dino_job_id
    result_3d["mask_request_id"] = mask_request_id

    return result_3d


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SAM3D 3D Furniture Generation — mask/boxes -> 3D .glb models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From existing mask URLs
  python gen_furniture_3d.py from_masks \\
      --image_url "https://example.com/pano.jpg" \\
      --mask_urls "https://example.com/mask_0.png" "https://example.com/mask_1.png" \\
      --sam3d_base_url "https://ai-test.aroomy.com/api/sam3d" \\
      --output_dir ./furniture

  # Full pipeline
  python gen_furniture_3d.py full_pipeline \\
      --image_path pano.jpg \\
      --dino_base_url "http://localhost:8001" \\
      --dino_api_key "dev-key" \\
      --text_prompt "sofa. chair. table." \\
      --sam_base_url "https://ai-test.aroomy.com/api/sam3d/masks" \\
      --sam3d_base_url "https://ai-test.aroomy.com/api/sam3d" \\
      --image_url_for_sam "https://example.com/pano.jpg" \\
      --output_dir ./output
        """,
    )

    subparsers = parser.add_subparsers(dest="mode", help="Run mode")
    subparsers.required = True

    # ---- from_masks subcommand ----
    masks_parser = subparsers.add_parser(
        "from_masks", help="Generate 3D from existing mask URLs -> SAM3D 3D"
    )
    masks_parser.add_argument("--image_url", required=True,
                              help="Publicly accessible URL of the original image")
    masks_parser.add_argument("--mask_urls", required=True, nargs="+",
                              help="One or more mask image URLs (space-separated)")
    masks_parser.add_argument("--sam3d_base_url", required=True,
                              help="SAM3D API base URL, e.g. https://ai-test.aroomy.com/api/sam3d")
    masks_parser.add_argument("--output_dir", required=True,
                              help="Output directory for .glb files and metadata")
    masks_parser.add_argument("--prompt", default="furniture",
                              help="Text prompt fallback (default: 'furniture')")
    masks_parser.add_argument("--no_textured_glb", action="store_true",
                              help="Disable textured GLB export (faster, less detail)")
    masks_parser.add_argument("--poll_interval", type=int, default=DEFAULT_POLL_INTERVAL,
                              help=f"Polling interval in seconds (default: {DEFAULT_POLL_INTERVAL})")
    masks_parser.add_argument("--poll_max_attempts", type=int, default=DEFAULT_POLL_MAX_ATTEMPTS,
                              help=f"Maximum polling attempts (default: {DEFAULT_POLL_MAX_ATTEMPTS})")

    # ---- full_pipeline subcommand ----
    full_parser = subparsers.add_parser(
        "full_pipeline", help="GroundingDINO -> SAM masks -> SAM3D 3D (end-to-end)"
    )
    full_parser.add_argument("--image_path", required=True,
                             help="Local panorama image path")
    full_parser.add_argument("--dino_base_url", required=True,
                             help="GroundingDINO service URL")
    full_parser.add_argument("--dino_api_key", required=True,
                             help="GroundingDINO API key (X-API-KEY)")
    full_parser.add_argument("--text_prompt", required=True,
                             help="Detection prompt, e.g. 'sofa. chair. table.'")
    full_parser.add_argument("--box_threshold", type=float, default=0.3,
                             help="Box confidence threshold (default: 0.3)")
    full_parser.add_argument("--text_threshold", type=float, default=0.25,
                             help="Text matching threshold (default: 0.25)")
    full_parser.add_argument("--sam_base_url", required=True,
                             help="SAM mask API base URL")
    full_parser.add_argument("--sam3d_base_url", required=True,
                             help="SAM3D 3D API base URL")
    full_parser.add_argument("--output_dir", required=True,
                             help="Output directory for masks, boxes, and 3D models")
    full_parser.add_argument("--image_url_for_sam", default=None,
                             help="Public image URL for SAM APIs (required!)")
    full_parser.add_argument("--furniture_prompt", default="furniture",
                             help="Text prompt for 3D generation (default: 'furniture')")
    full_parser.add_argument("--poll_interval", type=int, default=DEFAULT_POLL_INTERVAL,
                             help=f"Polling interval in seconds (default: {DEFAULT_POLL_INTERVAL})")
    full_parser.add_argument("--poll_max_attempts", type=int, default=DEFAULT_POLL_MAX_ATTEMPTS,
                             help=f"Maximum polling attempts (default: {DEFAULT_POLL_MAX_ATTEMPTS})")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "from_masks":
        run_from_masks(
            image_url=args.image_url,
            mask_urls=args.mask_urls,
            sam3d_base_url=args.sam3d_base_url.rstrip("/"),
            output_dir=args.output_dir,
            prompt=args.prompt,
            export_textured_glb=not args.no_textured_glb,
            poll_interval=args.poll_interval,
            poll_max_attempts=args.poll_max_attempts,
        )
    elif args.mode == "full_pipeline":
        run_full_pipeline(
            image_path=args.image_path,
            dino_base_url=args.dino_base_url.rstrip("/"),
            dino_api_key=args.dino_api_key,
            text_prompt=args.text_prompt,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            sam_base_url=args.sam_base_url.rstrip("/"),
            sam3d_base_url=args.sam3d_base_url.rstrip("/"),
            output_dir=args.output_dir,
            image_url_for_sam=args.image_url_for_sam,
            furniture_prompt=args.furniture_prompt,
            poll_interval=args.poll_interval,
            poll_max_attempts=args.poll_max_attempts,
        )
    else:
        parser.print_help()
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
