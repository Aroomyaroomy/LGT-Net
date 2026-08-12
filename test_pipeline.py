#!/usr/bin/env python3
"""
Minimal pipeline test — debug data formats end-to-end.
Run directly: python test_pipeline.py
"""

import asyncio
import httpx
import json
import sys
import time
from typing import Any, Dict, List, Optional

# =============================================================================
# CONFIG — edit these
# =============================================================================
IMAGE_URL = (
    "https://this-is-a-valid-bucket-name-651697298829-ap-northeast-1-an"
    ".s3.ap-northeast-1.amazonaws.com/kitchen.jpg"
)
TEXT_PROMPT = "furniture"

DINO_URL = "http://localhost:8001"
INPAINT_URL = "http://localhost:8002"
LGT_URL = "http://localhost:8000"
SAM_MASK_URL = "https://ai-test.aroomy.com/api/sam3d/masks"
SAM3D_URL = "https://ai-test.aroomy.com/api/sam3d"
API_KEY = "dev-local-test-key-16chars"

# =============================================================================
# HELPERS
# =============================================================================


async def download_image(url: str) -> bytes:
    print(f"[download] {url[:80]}...")
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as c:
        r = await c.get(url)
        r.raise_for_status()
        return r.content


async def step1_dino(image_bytes: bytes) -> dict:
    """Call GroundingDINO, return boxes dict."""
    print("[DINO] POST /predict ...")
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(
            f"{DINO_URL}/predict",
            headers={"X-API-KEY": API_KEY},
            files={"image": ("pano.jpg", image_bytes, "image/jpeg")},
            data={"text_prompt": TEXT_PROMPT, "box_threshold": "0.3", "text_threshold": "0.25"},
        )
        r.raise_for_status()
        data = r.json()

    job_id = data["job_id"]
    boxes_url = data.get("boxes_url") or f"/jobs/{job_id}/boxes"
    print(f"[DINO] job_id={job_id}, fetching boxes from {boxes_url}")

    async with httpx.AsyncClient(timeout=30) as c:
        r2 = await c.get(
            boxes_url if boxes_url.startswith("http") else f"{DINO_URL}{boxes_url}",
            headers={"X-API-KEY": API_KEY},
        )
        r2.raise_for_status()
        boxes = r2.json()

    # DEBUG: show structure
    print(f"[DINO] boxes keys: {list(boxes.keys())}")
    if "detections" in boxes:
        d = boxes["detections"]
        print(f"[DINO] detections: {len(d)}")
        if d and isinstance(d[0], dict):
            print(f"[DINO] first detection keys: {list(d[0].keys())}")
            if "box" in d[0]:
                print(f"[DINO] first box: {d[0]['box']}")
    if "box_prompts" in boxes:
        bp = boxes["box_prompts"]
        print(f"[DINO] box_prompts: {len(bp)}")
        if bp:
            print(f"[DINO] first box_prompt: {bp[0]}")

    return boxes


def build_box_prompts(boxes: dict) -> list:
    """Convert DINO boxes → SAM box_prompts."""
    if "box_prompts" in boxes:
        raw = boxes["box_prompts"]
    elif "detections" in boxes:
        raw = [d["box"] for d in boxes["detections"] if "box" in d]
    else:
        raise ValueError(f"Cannot find boxes in keys: {list(boxes.keys())}")

    prompts = []
    for box in raw:
        p = {
            "xMin": int(box["xMin"]),
            "yMin": int(box["yMin"]),
            "xMax": int(box["xMax"]),
            "yMax": int(box["yMax"]),
        }
        if p["xMin"] < p["xMax"] and p["yMin"] < p["yMax"]:
            prompts.append(p)

    print(f"[boxes] {len(prompts)} valid prompts: {prompts[:2]}...")
    return prompts


async def step2_sam(image_url: str, box_prompts: list) -> List[str]:
    """Submit SAM → poll → return mask URLs."""
    print(f"[SAM] submit with {len(box_prompts)} prompts ...")
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(
            f"{SAM_MASK_URL}/submit",
            json={"imageUrl": image_url, "boxPrompts": box_prompts},
        )
        print(f"[SAM] submit response: {r.status_code}")
        if not r.is_success:
            print(f"[SAM ERROR] {r.text[:500]}")
        r.raise_for_status()
        data = r.json()
    rid = data.get("requestId")
    print(f"[SAM] requestId={rid}, polling ...")

    for attempt in range(1, 60):
        await asyncio.sleep(5)
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(f"{SAM_MASK_URL}/status", params={"requestId": rid})
            r.raise_for_status()
            s = str(r.json().get("status", "")).upper()
        print(f"[SAM]   poll {attempt}: {s}")
        if s in ("COMPLETED", "SUCCEEDED"):
            async with httpx.AsyncClient(timeout=30) as c:
                r = await c.get(f"{SAM_MASK_URL}/result", params={"requestId": rid})
                r.raise_for_status()
                result = r.json()
            masks = result.get("data", {}).get("masks", [])
            urls = [m["url"] for m in masks if m.get("url")]
            print(f"[SAM] done — {len(urls)} masks")
            return urls
        if s in ("ERROR", "FAILED", "CANCELED"):
            raise RuntimeError(f"SAM job failed: {r.json()}")

    raise TimeoutError("SAM job timed out")


async def step3_inpaint(image_bytes: bytes, mask_urls: List[str]) -> bytes:
    """Inpaint masks → return inpainted image bytes."""
    print(f"[INPAINT] downloading {len(mask_urls)} masks ...")
    mask_files = []
    async with httpx.AsyncClient(timeout=60) as c:
        for i, url in enumerate(mask_urls):
            r = await c.get(url)
            r.raise_for_status()
            mask_files.append((f"mask_{i}.png", r.content))
            print(f"[INPAINT]   mask {i}: {len(r.content)} bytes")

    print(f"[INPAINT] POST /predict with {len(mask_files)} masks ...")
    files = [("image", ("pano.jpg", image_bytes, "image/jpeg"))]
    for name, data in mask_files:
        files.append(("masks", (name, data, "image/png")))

    async with httpx.AsyncClient(timeout=300) as c:
        r = await c.post(
            f"{INPAINT_URL}/predict",
            headers={"X-API-KEY": API_KEY},
            files=files,
            data={"crop_size": "512"},
        )
        r.raise_for_status()
        data = r.json()

    job_id = data["job_id"]
    img_url = data.get("image_url") or f"/jobs/{job_id}/inpainted.png"
    print(f"[INPAINT] job_id={job_id}, downloading inpainted image ...")
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.get(
            img_url if img_url.startswith("http") else f"{INPAINT_URL}{img_url}",
            headers={"X-API-KEY": API_KEY},
        )
        r.raise_for_status()
        return r.content


async def step4_lgt(image_bytes: bytes) -> dict:
    """Call LGT-Net → return layout result."""
    print("[LGT] POST /predict ...")
    async with httpx.AsyncClient(timeout=300) as c:
        r = await c.post(
            f"{LGT_URL}/predict",
            headers={"X-API-KEY": API_KEY},
            files={"image": ("inpainted.jpg", image_bytes, "image/jpeg")},
            data={
                "post_processing": "manhattan",
                "pre_processing": "true",
                "output_mesh": "true",
                "output_3d": "true",
            },
        )
        r.raise_for_status()
        data = r.json()
    print(f"[LGT] job_id={data.get('jobId')}, coordinates keys: {list(data.get('coordinates', {}).keys())}")
    return data


# =============================================================================
# MAIN
# =============================================================================


async def main():
    t0 = time.time()

    print("=" * 60)
    print("Step 1/4: DINO")
    print("=" * 60)
    image_bytes = await download_image(IMAGE_URL)
    boxes = await step1_dino(image_bytes)

    print()
    print("=" * 60)
    print("Step 2/4: SAM")
    print("=" * 60)
    box_prompts = build_box_prompts(boxes)
    if not box_prompts:
        print("FAIL: no box prompts — check DINO format above")
        return
    mask_urls = await step2_sam(IMAGE_URL, box_prompts)
    if not mask_urls:
        print("FAIL: no mask URLs")
        return

    print()
    print("=" * 60)
    print("Step 3/4: Inpaint")
    print("=" * 60)
    inpainted = await step3_inpaint(image_bytes, mask_urls)
    print(f"[INPAINT] result: {len(inpainted)} bytes")

    print()
    print("=" * 60)
    print("Step 4/4: LGT-Net")
    print("=" * 60)
    lgt_result = await step4_lgt(inpainted)

    print()
    print("=" * 60)
    print(f"PIPELINE DONE — {round(time.time() - t0, 1)}s")
    print("=" * 60)
    print(json.dumps(lgt_result.get("coordinates", {}), indent=2, default=str)[:2000])


if __name__ == "__main__":
    asyncio.run(main())
