# Grounding DINO - Aroomy

This is an altered inference service based on the wonderful paper "[Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://arxiv.org/abs/2303.05499)". [[Official Repo](https://github.com/IDEA-Research/GroundingDINO)] [[Demo](https://huggingface.co/spaces/ShilongLiu/Grounding_DINO_demo)]

It exposes a FastAPI endpoint for zero-shot object detection from an image and a text prompt (for example furniture categories in an interior photo).

---

**Model**

Open-set detector that takes an `(image, text)` pair and returns bounding boxes for phrases in the prompt. Category names in the text prompt should be separated with `.` (for example `sofa . chair . table .`).

This service defaults to **GroundingDINO-T** (Swin-T backbone). You can switch to GroundingDINO-B by changing the config and checkpoint paths (see Downloading Pre-trained Weights).

---

# Installation

Set a secret API key before starting the service. Create a `.env` file in the `GroundingDINO` project root:

```shell
DINO_SECRET_KEY=your-secret-key-at-least-16-chars
```

For local development only, you can set `DINO_DEV_MODE=true` to use a built-in insecure key instead (by default the container runs in dev-mode).

#### Please see Downloading Pre-trained Weights section first

An NVIDIA GPU with the NVIDIA Container Toolkit is required (`gpus: all` in Compose). Run the following script to build and start Docker from the `GroundingDINO` directory:

```shell
docker compose up --build
```

By default the service is hosted locally at `127.0.0.1:8001`.

Optional build override if your GPU is older than Ampere (for example T4 / V100):

```shell
TORCH_CUDA_ARCH_LIST=7.5;8.0 docker compose up --build
```

---

# Inference

After the service is running, you can run a health check to see if the model is loaded:

```shell
curl.exe http://127.0.0.1:8001/health
```

`/health` is unauthenticated and returns `200` when healthy or `503` when the model is not loaded.

To run detection, include the `X-API-KEY` header:

```shell
curl.exe -s -X POST "http://127.0.0.1:8001/predict" `
  -H "X-API-KEY: your-secret-key-at-least-16-chars" `
  -F "image=@path/to/image.jpg" `
  -F "text_prompt=sofa . chair . table . lamp . bookshelf" `
  -F "box_threshold=0.3" `
  -F "text_threshold=0.25" `
  -F "visualize=true"
```

Flags:
- `X-API-KEY`: required API key (must match `DINO_SECRET_KEY`, or the dev key when `DINO_DEV_MODE=true`)
- `image`: path to the input image (absolute or relative)
- `text_prompt`: detection prompt; separate categories with `.`
- `box_threshold`: box confidence threshold (`0.3` by default)
- `text_threshold`: text/phrase threshold (`0.25` by default); ignored when `token_spans` is set
- `token_spans`: optional JSON string of character spans into the caption, for example `"[[[2, 5]]]"` for a phrase in the prompt (see official demo notes)
- `visualize`: if `true`, draw boxes on the image and return `visualization_url` (`false` by default)

Example JSON response:

```json
{
  "job_id": "25acb4ae93a3419db2fcc91b3c1ee584",
  "boxes_url": "/jobs/{job_id}/boxes",
  "visualization_url": "/jobs/{job_id}/visualization",
  "num_detections": 14
}
```

Download detection boxes JSON:

```shell
curl.exe -H "X-API-KEY: your-secret-key-at-least-16-chars" `
  "http://127.0.0.1:8001/jobs/{job_id}/boxes" `
  -o boxes.json
```

Download visualization (only if `visualize=true`):

```shell
curl.exe -H "X-API-KEY: your-secret-key-at-least-16-chars" `
  "http://127.0.0.1:8001/jobs/{job_id}/visualization" `
  -o pred.jpg
```

Job outputs are written under `./data` on the host (mounted to `/data` in the container) and cleaned up after a TTL (default 600 seconds).

---

# Downloading Pre-trained Weights

Before the model can be loaded, download a checkpoint into `./weights`. You must manually create this folder either by hand or using the script below. By default the service expects GroundingDINO-T:

(Script is run from GroundingDINO directory)
```shell
mkdir weights
cd weights
curl.exe -L -o groundingdino_swint_ogc.pth `
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..
```

Weights are available for download here (please make use of these links if script above does not work):

| Name | Backbone | Checkpoint | Config |
| --- | --- | --- | --- |
| GroundingDINO-T (default) | Swin-T | [GitHub](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth) \| [HF](https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth) | `GroundingDINO_SwinT_OGC.py` |
| GroundingDINO-B | Swin-B | [GitHub](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth) \| [HF](https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth) | `GroundingDINO_SwinB_cfg.py` |

Make sure the default weight file is stored as follows:

```
weights
|-- groundingdino_swint_ogc.pth
```

To use GroundingDINO-B instead, place `groundingdino_swinb_cogcoor.pth` under `weights/` and set in `compose.yml` (or your environment):

```shell
DINO_CONFIG_DIR=/opt/groundingdino/groundingdino/config/GroundingDINO_SwinB_cfg.py
DINO_CKPT_DIR=/opt/groundingdino/weights/groundingdino_swinb_cogcoor.pth
```

No image rebuild is required for swapping mounted weights or these env vars; restart the container after changing them.

---

# Acknowledgements

Upstream implementation: [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO).

Related projects from the original authors and community:

- [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)
- [Grounded SAM 2](https://github.com/IDEA-Research/Grounded-SAM-2)
- [Hugging Face Grounding DINO](https://huggingface.co/docs/transformers/model_doc/grounding-dino)

# Citation

If you use this code for your research, please cite

```
@article{liu2023grounding,
  title={Grounding dino: Marrying dino with grounded pre-training for open-set object detection},
  author={Liu, Shilong and Zeng, Zhaoyang and Ren, Tianhe and Li, Feng and Zhang, Hao and Yang, Jie and Li, Chunyuan and Yang, Jianwei and Su, Hang and Zhu, Jun and others},
  journal={arXiv preprint arXiv:2303.05499},
  year={2023}
}
```
