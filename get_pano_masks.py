import io
import json
import os
import time

import requests
from PIL import Image


ACTIVE_STATUSES = {"IN_QUEUE", "QUEUED", "PENDING", "IN_PROGRESS", "PROCESSING", "RUNNING"}
DONE_STATUSES = {"COMPLETED", "SUCCEEDED"}
FAILED_STATUSES = {"FAILED", "ERROR", "CANCELED", "CANCELLED"}


def get_image_frame(image_url: str) -> tuple[int, int]:
    response = requests.get(image_url, timeout=30)
    response.raise_for_status()
    image = Image.open(io.BytesIO(response.content))
    width, height = image.size
    return width, height


def load_box_prompts_from_json(boxes_json_path: str, min_score: float = None) -> list[dict]:
    """Read GroundingDINO boxes.json and format prompts for mask job submission."""
    with open(boxes_json_path, encoding="utf-8") as f:
        data = json.load(f)

    if "box_prompts" in data and min_score is None:
        raw_boxes = data["box_prompts"]
    else:
        detections = data.get("detections", [])
        if min_score is not None:
            detections = [d for d in detections if d.get("score") is not None and d["score"] >= min_score]
        raw_boxes = [d["box"] for d in detections if "box" in d]

    box_prompts = []
    for box in raw_boxes:
        prompt = {
            "xMin": int(box["xMin"]),
            "yMin": int(box["yMin"]),
            "xMax": int(box["xMax"]),
            "yMax": int(box["yMax"]),
        }
        if prompt["xMin"] >= prompt["xMax"] or prompt["yMin"] >= prompt["yMax"]:
            continue
        box_prompts.append(prompt)

    if not box_prompts:
        raise ValueError(f"No valid box prompts found in {boxes_json_path}")

    return box_prompts


def submit_mask_job(image_url: str, base_url: str, box_prompts: list[dict]) -> str:
    if not box_prompts:
        raise ValueError("box_prompts must contain at least one box")

    response = requests.post(
        f"{base_url}/submit",
        json={"imageUrl": image_url, "boxPrompts": box_prompts},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    request_id = data.get("requestId")
    if not request_id:
        raise RuntimeError(f"Submit response missing requestId: {data}")
    return request_id


def poll_mask_job_status(request_id: str, base_url: str) -> str:
    response = requests.get(
        f"{base_url}/status",
        params={"requestId": request_id},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    status = data.get("status")
    if not status:
        raise RuntimeError(f"Status response missing status: {data}")
    return status


def get_mask_job_result(request_id: str, base_url: str) -> dict:
    response = requests.get(
        f"{base_url}/result",
        params={"requestId": request_id},
        timeout=30,
    )
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


def download_binary(url: str, output_path: str) -> None:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(response.content)


def wait_and_fetch_mask_job(
    request_id: str,
    base_url: str,
    output_dir: str,
    interval: int = 10,
    max_attempts: int = 30,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    for _ in range(max_attempts):
        status = poll_mask_job_status(request_id, base_url)

        if status in FAILED_STATUSES:
            raise RuntimeError(f"Mask job failed with status={status}, requestId={request_id}")

        if status in DONE_STATUSES:
            result = get_mask_job_result(request_id, base_url)
            result_data = result["data"]

            for idx, mask in enumerate(result_data.get("masks", [])):
                filename = mask.get("file_name") or f"mask_{idx}.png"
                download_binary(mask["url"], os.path.join(output_dir, filename))

            image_obj = result_data.get("image")
            if image_obj and image_obj.get("url"):
                image_filename = image_obj.get("file_name") or "annotated.png"
                download_binary(image_obj["url"], os.path.join(output_dir, image_filename))

            return result

        if status not in ACTIVE_STATUSES:
            raise RuntimeError(f"Unexpected status '{status}' for requestId={request_id}")

        time.sleep(interval)

    raise TimeoutError(f"Failed to fetch mask job result | Job ID {request_id}")


if __name__ == "__main__":
    BASE_URL = "https://ai-test.aroomy.com/api/sam3d/masks"
    IMAGE_URL = "https://this-is-a-valid-bucket-name-651697298829-ap-northeast-1-an.s3.ap-northeast-1.amazonaws.com/small_square_room.jpg?response-content-disposition=inline&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEBsaDmFwLW5vcnRoZWFzdC0xIkcwRQIhAMCVgi5xFA3we1wsvAocFY8R6%2BBrkuREg7ntwURgSjMlAiAxca7E%2BfvdgmvwRJuX2lgsZf%2BGBfg8dplXblURFpsJ1Cq%2BAwjl%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAAaDDY1MTY5NzI5ODgyOSIMXgM4Oqs9ToEDFdOMKpID4gdnwy2SzWZ24AWEI4x3i%2F%2FT2tXWKwkBRkxRRTY14Qv0opnTZY5X4P42BxdyuSFy%2F%2FNBzU6sFfHXMVh%2Fz1u4tTmcZ8ktZ7UaLqHSXtb1nHDq2X5ZEyfZYj1iRatQTYl%2FTBsgQCRNXbKYaTc6ySo0ny8JAib4vzy%2B8iHNEYaV7KVIkS6BVFA1873Cog7H9HzHPv%2BIKngfUhBR9qxlkwY9jq2%2BkmwL72joxNgjOmuh3NMLrtju4fWu4U6fNeLwzPDWaNzOF4FhkIs4tesSsnga64WoEI6p8Ka1pQS5zIedwvnNpGqGU%2FcfOokbcucxMlZVvmhQ69q1rFAmddTAuazjAmanshhKMz8DVvtMYe542qimxkrZ4BChH7nRnWb8GTLXrDnX65pbnszqZu9itszDuzvYM%2Fylii4o2h7MOeRJHysH8k0sZKaQTw6jx5pM1ZLha0M%2FMscEy9m8PKxmXUSxxT18RaGqEtpU5Y5E4ummBKjzOD2UsqS%2FCAKdHM0h30k6gTPHG%2Br%2BdChH%2B78sFjqBja37MJX3zdIGOt4CW8y6fBmMdtYmRdg7PAz9bvtDOPm5%2BWbxvy6%2FNYEr0TAXGl%2BGRBKZOzdbA%2FiJisUjui%2Bb%2FIbvQyN4UHDJUDROZjhNMFMWEP2p98o8IEiK%2Fjd5RHiE04wBiVW1B2TjX6IXYoakpLodi1c5Rtiax8efw%2F4iLDMxdU4b7uZ2BBxqtA925ToiHLvEoIDcXOiBHJrqd15g81inYgWWRVeVHJzuJIH%2FWxj1smYybbuA%2FdqQrkh12QUAGr10yUtbB2cI8OHrDXy%2Bv1b5akn8bypzhdy1rdWKAtaHb481EApAGR9oq11sbvoLo%2B6%2Fa4o100jziJgmFgJs4ZWqy9mW2Tcbgj3ro%2FVvmXt4uY4VnVSUTUH99BMLEZ3PrlZFwx2BYMTEpeW7nb3c5Fjcs7YP4xym6EudKmgvFxveUTr4N83iFbyuQ%2F7YXnrYm63XvaXKULMT6DDH9qipib7NLNn4hKTlJ8Q%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIAZPPBPTWG23V7FH2Z%2F20260712%2Fap-northeast-1%2Fs3%2Faws4_request&X-Amz-Date=20260712T113805Z&X-Amz-Expires=7200&X-Amz-SignedHeaders=host&X-Amz-Signature=ee60f44b52fa0eb575c515d934c5277624362a95a137f03943a5e1822fb654c5"
    OUTPUT_DIR = "src/segmented_panos/small_square_room/"
    BOXES_JSON = "testing_src/dino_boxes/small_square_room/boxes.json"

    box_prompts = load_box_prompts_from_json(BOXES_JSON)
    request_id = submit_mask_job(IMAGE_URL, BASE_URL, box_prompts)
    wait_and_fetch_mask_job(request_id, BASE_URL, OUTPUT_DIR)