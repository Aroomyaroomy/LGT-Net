"""
Bridge script: feed original image + bounding boxes into Aroomy SAM3D mask API
to generate segmentation masks.

Supports two modes:
  1. local  — read boxes.json from local disk + remote image URL → SAM API
  2. remote — call GroundingDINO API to generate boxes → SAM API

Usage:
  # Local mode (use existing GroundingDINO boxes.json)
  python get_pano_masks.py local \
      --image_url "https://example.com/pano.jpg" \
      --boxes_json path/to/boxes.json \
      --sam_base_url "https://ai-test.aroomy.com/api/sam3d/masks" \
      --output_dir ./masks

  # Remote mode (auto-detect objects via GroundingDINO, then segment)
  python get_pano_masks.py remote \
      --image_path path/to/pano.jpg \
      --dino_base_url "http://localhost:8001" \
      --dino_api_key "your-key" \
      --text_prompt "sofa. chair. table." \
      --sam_base_url "https://ai-test.aroomy.com/api/sam3d/masks" \
      --output_dir ./masks

  Can also be imported as a module:
      from get_pano_masks import run_local_mode, run_remote_mode
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from typing import Optional

import requests
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTIVE_STATUSES = {"IN_QUEUE", "QUEUED", "PENDING", "IN_PROGRESS", "PROCESSING", "RUNNING"}
DONE_STATUSES = {"COMPLETED", "SUCCEEDED"}
FAILED_STATUSES = {"FAILED", "ERROR", "CANCELED", "CANCELLED"}

# Default polling parameters
DEFAULT_POLL_INTERVAL = 10       # seconds
DEFAULT_POLL_MAX_ATTEMPTS = 30  # max 30 attempts = 5 minutes total


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_image_frame(image_url: str) -> tuple[int, int]:
    """Download image and return (width, height) for coordinate validation."""
    response = requests.get(image_url, timeout=30)
    response.raise_for_status()
    image = Image.open(io.BytesIO(response.content))
    return image.size  # (width, height)


def download_binary(url: str, output_path: str) -> None:
    """Download a binary file from a URL to a local path."""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(response.content)


# ---------------------------------------------------------------------------
# Box loading — supports local JSON and GroundingDINO API sources
# ---------------------------------------------------------------------------

def load_box_prompts_from_json(
    boxes_json_path: str,
    min_score: Optional[float] = None,
) -> list[dict]:
    """
    Extract box_prompts from a GroundingDINO boxes.json file.

    Prefers the pre-formatted ``box_prompts`` field when available;
    otherwise falls back to extracting ``box`` from each entry in ``detections``.

    Args:
        boxes_json_path: Path to the boxes.json file.
        min_score:       Minimum confidence score; detections below this are filtered out.

    Returns:
        List of box_prompt dicts, each with keys xMin, yMin, xMax, yMax (int).
    """
    with open(boxes_json_path, encoding="utf-8") as f:
        data = json.load(f)

    # Prefer pre-formatted box_prompts
    if "box_prompts" in data and min_score is None:
        raw_boxes = data["box_prompts"]
    else:
        detections = data.get("detections", [])
        if min_score is not None:
            detections = [
                d for d in detections
                if d.get("score") is not None and d["score"] >= min_score
            ]
        raw_boxes = [d["box"] for d in detections if "box" in d]

    box_prompts = []
    for box in raw_boxes:
        prompt = {
            "xMin": int(box["xMin"]),
            "yMin": int(box["yMin"]),
            "xMax": int(box["xMax"]),
            "yMax": int(box["yMax"]),
        }
        # Skip invalid boxes
        if prompt["xMin"] >= prompt["xMax"] or prompt["yMin"] >= prompt["yMax"]:
            continue
        box_prompts.append(prompt)

    if not box_prompts:
        raise ValueError(f"No valid box prompts found in {boxes_json_path}")

    return box_prompts


def fetch_boxes_from_dino(
    dino_base_url: str,
    dino_api_key: str,
    image_path: str,
    text_prompt: str,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
) -> tuple[list[dict], str]:
    """
    Call GroundingDINO API to detect objects and return box_prompts.

    Steps:
        1. Upload image to GroundingDINO /predict
        2. Download the generated boxes.json
        3. Parse out box_prompts

    Args:
        dino_base_url:  GroundingDINO service URL.
        dino_api_key:   API key (X-API-KEY header).
        image_path:     Local panorama image path.
        text_prompt:    Detection prompt, e.g. "sofa. chair. table."
        box_threshold:  Bounding box confidence threshold.
        text_threshold: Text matching threshold.

    Returns:
        (box_prompts, job_id) — list of box_prompt dicts + GroundingDINO job_id.
    """
    # 1. Upload image to GroundingDINO
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    predict_response = requests.post(
        f"{dino_base_url}/predict",
        headers={"X-API-KEY": dino_api_key},
        files={"image": (os.path.basename(image_path), image_bytes, "image/jpeg")},
        data={
            "text_prompt": text_prompt,
            "box_threshold": str(box_threshold),
            "text_threshold": str(text_threshold),
        },
        timeout=120,
    )
    predict_response.raise_for_status()
    predict_data = predict_response.json()
    job_id = predict_data.get("job_id")
    if not job_id:
        raise RuntimeError(f"GroundingDINO /predict returned no job_id: {predict_data}")

    num_detections = predict_data.get("num_detections", 0)
    print(f"  [DINO] job_id={job_id}, detections={num_detections}")

    if num_detections == 0:
        print("  [DINO] No objects detected, returning empty box_prompts")
        return [], job_id

    # 2. Download boxes.json
    boxes_response = requests.get(
        f"{dino_base_url}/jobs/{job_id}/boxes",
        headers={"X-API-KEY": dino_api_key},
        timeout=30,
    )
    boxes_response.raise_for_status()
    boxes_data = boxes_response.json()

    # 3. Extract box_prompts from the API response dict
    box_prompts = _parse_box_prompts_from_data(boxes_data)

    if not box_prompts:
        print("  [DINO] No valid box_prompts in boxes.json")

    return box_prompts, job_id


def _parse_box_prompts_from_data(data: dict) -> list[dict]:
    """Extract box_prompts from a boxes.json dict (same logic as load_box_prompts_from_json)."""
    if "box_prompts" in data:
        raw_boxes = data["box_prompts"]
    else:
        detections = data.get("detections", [])
        raw_boxes = [d["box"] for d in detections if "box" in d]

    box_prompts = []
    for box in raw_boxes:
        prompt = {
            "xMin": int(box["xMin"]),
            "yMin": int(box["yMin"]),
            "xMax": int(box["xMax"]),
            "yMax": int(box["yMax"]),
        }
        if prompt["xMin"] < prompt["xMax"] and prompt["yMin"] < prompt["yMax"]:
            box_prompts.append(prompt)
    return box_prompts


# ---------------------------------------------------------------------------
# SAM3D Mask API interaction
# ---------------------------------------------------------------------------

def submit_mask_job(
    image_url: str,
    sam_base_url: str,
    box_prompts: list[dict],
) -> str:
    """
    Submit a segmentation job to the Aroomy SAM3D mask API.

    Args:
        image_url:   Publicly accessible URL of the panorama image.
        sam_base_url: SAM API base URL, e.g. https://ai-test.aroomy.com/api/sam3d/masks.
        box_prompts: List of bounding box dicts.

    Returns:
        request_id — used for subsequent status polling and result retrieval.
    """
    if not box_prompts:
        raise ValueError("box_prompts must contain at least one box")

    response = requests.post(
        f"{sam_base_url}/submit",
        json={"imageUrl": image_url, "boxPrompts": box_prompts},
        timeout=30,
    )
    if not response.ok:
        print(f"  [SAM ERROR] HTTP {response.status_code}: {response.text[:500]}")
    response.raise_for_status()
    data = response.json()
    request_id = data.get("requestId")
    if not request_id:
        raise RuntimeError(f"Submit response missing requestId: {data}")
    return request_id


def poll_mask_job_status(request_id: str, sam_base_url: str) -> str:
    """Query the status of a SAM mask job."""
    response = requests.get(
        f"{sam_base_url}/status",
        params={"requestId": request_id},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    status = data.get("status")
    if not status:
        raise RuntimeError(f"Status response missing status: {data}")
    return status


def get_mask_job_result(request_id: str, sam_base_url: str) -> dict:
    """Retrieve the final result of a SAM mask job (includes mask download URLs)."""
    response = requests.get(
        f"{sam_base_url}/result",
        params={"requestId": request_id},
        timeout=30,
    )
    if not response.ok:
        print(f"  [SAM ERROR] HTTP {response.status_code}: {response.text[:500]}")
    if response.status_code == 409:
        raise RuntimeError("Job result not ready yet.")
    response.raise_for_status()
    data = response.json()
    if "error" in data:
        details = data.get("details")
        raise RuntimeError(f"{data['error']} | Details: {details}")
    if "data" not in data:
        raise RuntimeError(f"Malformed result payload: {data}")
    return data


def wait_and_fetch_mask_job(
    request_id: str,
    sam_base_url: str,
    output_dir: str,
    interval: int = DEFAULT_POLL_INTERVAL,
    max_attempts: int = DEFAULT_POLL_MAX_ATTEMPTS,
) -> dict:
    """
    Poll until the SAM mask job completes, then download all mask files locally.

    Args:
        request_id:   SAM job request ID.
        sam_base_url: SAM API base URL.
        output_dir:   Output directory (created automatically).
        interval:     Polling interval in seconds.
        max_attempts: Maximum number of polling attempts.

    Returns:
        Full SAM result response dict.
    """
    os.makedirs(output_dir, exist_ok=True)

    NETWORK_RETRIES = 3
    network_failures = 0

    for attempt in range(1, max_attempts + 1):
        try:
            status = poll_mask_job_status(request_id, sam_base_url)
            network_failures = 0  # reset on success
        except (requests.ConnectionError, requests.Timeout) as e:
            network_failures += 1
            if network_failures > NETWORK_RETRIES:
                raise RuntimeError(
                    f"Network error after {NETWORK_RETRIES} retries: {e}"
                ) from e
            print(f"  [SAM] network error (retry {network_failures}/{NETWORK_RETRIES}): {e}")
            time.sleep(interval)
            continue

        print(f"  [SAM] status={status} (attempt {attempt}/{max_attempts})")

        if status in FAILED_STATUSES:
            raise RuntimeError(
                f"Mask job failed with status={status}, requestId={request_id}"
            )

        if status in DONE_STATUSES:
            result = get_mask_job_result(request_id, sam_base_url)
            result_data = result["data"]

            # Download each mask file
            for idx, mask in enumerate(result_data.get("masks", [])):
                filename = mask.get("file_name") or f"mask_{idx}.png"
                filepath = os.path.abspath(os.path.join(output_dir, filename))
                download_binary(mask["url"], filepath)
                print(f"  [SAM] Downloaded: {filepath}")

            # Download annotated image (if present)
            image_obj = result_data.get("image")
            if image_obj and image_obj.get("url"):
                image_filename = image_obj.get("file_name") or "annotated.png"
                filepath = os.path.abspath(os.path.join(output_dir, image_filename))
                download_binary(image_obj["url"], filepath)
                print(f"  [SAM] Downloaded annotated image: {filepath}")

            return result

        if status not in ACTIVE_STATUSES:
            raise RuntimeError(
                f"Unexpected status '{status}' for requestId={request_id}"
            )

        time.sleep(interval)

    raise TimeoutError(
        f"Failed to fetch mask job result after {max_attempts} attempts "
        f"(requestId={request_id})"
    )


# ---------------------------------------------------------------------------
# High-level entry points — two modes
# ---------------------------------------------------------------------------

def run_local_mode(
    image_url: str,
    boxes_json_path: str,
    sam_base_url: str,
    output_dir: str,
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
    print(f"[local] Loading boxes from {boxes_json_path} ...")
    box_prompts = load_box_prompts_from_json(boxes_json_path, min_score=min_score)
    print(f"  [boxes] {len(box_prompts)} box prompts loaded")

    print("[SAM] Submitting segmentation job ...")
    request_id = submit_mask_job(image_url, sam_base_url, box_prompts)
    print(f"  [SAM] requestId={request_id}")

    return wait_and_fetch_mask_job(
        request_id, sam_base_url, output_dir,
        interval=poll_interval, max_attempts=poll_max_attempts,
    )


def run_remote_mode(
    image_path: str,
    dino_base_url: str,
    dino_api_key: str,
    text_prompt: str,
    sam_base_url: str,
    output_dir: str,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    poll_interval: int = DEFAULT_POLL_INTERVAL,
    poll_max_attempts: int = DEFAULT_POLL_MAX_ATTEMPTS,
    image_url_for_sam: Optional[str] = None,
) -> dict:
    """
    Remote mode: call GroundingDINO API → get boxes → feed to SAM API.

    Note: the SAM API requires a publicly accessible image URL.  If only a
    local image_path is provided, SAM may not be able to reach it.  Use
    ``image_url_for_sam`` to pass a pre-uploaded S3/CDN URL.

    Args:
        image_path:       Local panorama image path.
        dino_base_url:    GroundingDINO service URL.
        dino_api_key:     GroundingDINO API key.
        text_prompt:      Detection prompt, e.g. "sofa. chair. table."
        sam_base_url:     SAM API base URL.
        output_dir:       Output directory for masks and boxes.
        box_threshold:    GroundingDINO box confidence threshold.
        text_threshold:   GroundingDINO text matching threshold.
        poll_interval:    Polling interval in seconds.
        poll_max_attempts: Maximum polling attempts.
        image_url_for_sam: Public image URL passed to SAM (falls back to image_path).

    Returns:
        SAM result dict (includes ``dino_job_id`` for traceability).
    """
    # Step 0: prepare image_url (SAM needs a URL it can reach)
    image_url = image_url_for_sam or image_path
    if image_url == image_path:
        print(f"[WARNING] image_url is a local path '{image_path}', "
              f"SAM API may not be able to access it!")
        print("  Consider uploading the image to S3/CDN first, "
              "then pass --image_url_for_sam")

    # Step 1: call GroundingDINO
    print("[remote] Calling GroundingDINO to detect objects ...")
    print(f"  image:  {image_path}")
    print(f"  prompt: {text_prompt}")
    box_prompts, dino_job_id = fetch_boxes_from_dino(
        dino_base_url, dino_api_key, image_path, text_prompt,
        box_threshold=box_threshold, text_threshold=text_threshold,
    )
    print(f"  [DINO] {len(box_prompts)} box prompts returned")

    # Step 2: save boxes.json to output_dir for later inspection
    boxes_out_path = os.path.join(output_dir, f"dino_{dino_job_id}_boxes.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(boxes_out_path, "w", encoding="utf-8") as f:
        json.dump({
            "dino_job_id": dino_job_id,
            "image_path": image_path,
            "text_prompt": text_prompt,
            "box_prompts": box_prompts,
            "num_boxes": len(box_prompts),
        }, f, indent=2, ensure_ascii=False)
    print(f"  [boxes] Saved to {os.path.abspath(boxes_out_path)}")

    # Step 3: skip SAM if no objects detected
    if not box_prompts:
        print("[SKIP] No objects detected, skipping SAM segmentation")
        return {"dino_job_id": dino_job_id, "box_prompts": [], "skipped": True}

    # Step 4: call SAM API
    print("[SAM] Submitting segmentation job ...")
    request_id = submit_mask_job(image_url, sam_base_url, box_prompts)
    print(f"  [SAM] requestId={request_id}")

    result = wait_and_fetch_mask_job(
        request_id, sam_base_url, output_dir,
        interval=poll_interval, max_attempts=poll_max_attempts,
    )
    result["dino_job_id"] = dino_job_id
    return result


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GroundingDINO → SAM3D Mask bridge tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local mode
  python get_pano_masks.py local \\
      --image_url "https://example.com/pano.jpg" \\
      --boxes_json boxes.json \\
      --sam_base_url "https://ai-test.aroomy.com/api/sam3d/masks" \\
      --output_dir ./masks

  # Remote mode
  python get_pano_masks.py remote \\
      --image_path pano.jpg \\
      --dino_base_url "http://localhost:8001" \\
      --dino_api_key "your-key" \\
      --text_prompt "sofa. chair. table." \\
      --sam_base_url "https://ai-test.aroomy.com/api/sam3d/masks" \\
      --output_dir ./masks
        """,
    )

    subparsers = parser.add_subparsers(dest="mode", help="Run mode")
    subparsers.required = True

    # ---- local subcommand ----
    local_parser = subparsers.add_parser("local", help="Load boxes from local JSON → SAM")
    local_parser.add_argument("--image_url", required=True,
                              help="Publicly accessible URL of the panorama image")
    local_parser.add_argument("--boxes_json", required=True,
                              help="Path to GroundingDINO boxes.json output")
    local_parser.add_argument("--sam_base_url", required=True,
                              help="SAM3D mask API base URL")
    local_parser.add_argument("--output_dir", required=True,
                              help="Output directory for masks")
    local_parser.add_argument("--min_score", type=float, default=None,
                              help="Minimum confidence score (filter low-quality detections)")
    local_parser.add_argument("--poll_interval", type=int, default=DEFAULT_POLL_INTERVAL,
                              help=f"Polling interval in seconds (default: {DEFAULT_POLL_INTERVAL})")
    local_parser.add_argument("--poll_max_attempts", type=int, default=DEFAULT_POLL_MAX_ATTEMPTS,
                              help=f"Maximum polling attempts (default: {DEFAULT_POLL_MAX_ATTEMPTS})")

    # ---- remote subcommand ----
    remote_parser = subparsers.add_parser("remote", help="Call GroundingDINO API → SAM")
    remote_parser.add_argument("--image_path", required=True,
                               help="Local panorama image path")
    remote_parser.add_argument("--dino_base_url", required=True,
                               help="GroundingDINO service URL")
    remote_parser.add_argument("--dino_api_key", required=True,
                               help="GroundingDINO API key")
    remote_parser.add_argument("--text_prompt", required=True,
                               help="Detection prompt, e.g. 'sofa. chair. table.'")
    remote_parser.add_argument("--box_threshold", type=float, default=0.3,
                               help="Box confidence threshold (default: 0.3)")
    remote_parser.add_argument("--text_threshold", type=float, default=0.25,
                               help="Text matching threshold (default: 0.25)")
    remote_parser.add_argument("--sam_base_url", required=True,
                               help="SAM3D mask API base URL")
    remote_parser.add_argument("--output_dir", required=True,
                               help="Output directory for masks and boxes")
    remote_parser.add_argument("--image_url_for_sam", default=None,
                               help="Public image URL for SAM (falls back to image_path, "
                               "which SAM may not be able to reach)")
    remote_parser.add_argument("--poll_interval", type=int, default=DEFAULT_POLL_INTERVAL,
                               help=f"Polling interval in seconds (default: {DEFAULT_POLL_INTERVAL})")
    remote_parser.add_argument("--poll_max_attempts", type=int, default=DEFAULT_POLL_MAX_ATTEMPTS,
                               help=f"Maximum polling attempts (default: {DEFAULT_POLL_MAX_ATTEMPTS})")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "local":
        run_local_mode(
            image_url=args.image_url,
            boxes_json_path=args.boxes_json,
            sam_base_url=args.sam_base_url.rstrip("/"),
            output_dir=args.output_dir,
            min_score=args.min_score,
            poll_interval=args.poll_interval,
            poll_max_attempts=args.poll_max_attempts,
        )
    elif args.mode == "remote":
        run_remote_mode(
            image_path=args.image_path,
            dino_base_url=args.dino_base_url.rstrip("/"),
            dino_api_key=args.dino_api_key,
            text_prompt=args.text_prompt,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            sam_base_url=args.sam_base_url.rstrip("/"),
            output_dir=args.output_dir,
            image_url_for_sam=args.image_url_for_sam,
            poll_interval=args.poll_interval,
            poll_max_attempts=args.poll_max_attempts,
        )
    else:
        parser.print_help()
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
