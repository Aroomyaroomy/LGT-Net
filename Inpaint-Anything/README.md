# Inpaint-Anything - Aroomy

This is an altered inference service based on the wonderful project "[Inpaint Anything: Segment Anything Meets Image Inpainting](https://arxiv.org/abs/2304.06790)". [[Official Repo](https://github.com/geekyutao/Inpaint-Anything)] [[Demo](https://huggingface.co/spaces/VIPLab/Inpaint-Anything)]

It exposes a FastAPI endpoint for **mask-guided inpainting** (LaMa fill path only): given an image and a directory of `mask_*.png` files, it removes / fills the masked regions and returns the inpainted image.

---

**Model**

This service loads **big-lama** and applies dilated masks sequentially with optional crop-based filling (`crop_size`). SAM / Remove-Anything CLI tooling from upstream is not required for the `/predict` API path.

---

# Installation

Set a secret API key before starting the service. Create a `.env` file in the `Inpaint-Anything` project root:

```shell
INPAINT_SECRET_KEY=your-secret-key-at-least-16-chars
```

For local development only, you can set `INPAINT_DEV_MODE=true` to use a built-in insecure key instead (by default the Compose file runs with `INPAINT_DEV_MODE=true`).

#### Please see Downloading Pre-trained Weights section first

An NVIDIA GPU with the NVIDIA Container Toolkit is required (`gpus: all` in Compose). Run the following from the `Inpaint-Anything` directory:

```shell
docker compose up --build
```

By default the service is hosted locally at `127.0.0.1:8002`.

---

# Inference

After the service is running, you can run a health check to see if the model is loaded:

```shell
curl.exe http://127.0.0.1:8002/health
```

`/health` is unauthenticated and returns `200` when healthy or `503` when the model is not loaded.

`mask_dir` is a **path on the server** (container filesystem), not an uploaded archive. Masks must already be visible inside the container as `mask_*.png`. Compose mounts `./data` to `/data`, so put masks under `./data/...` on the host and pass the matching `/data/...` path.

To run inpainting, include the `X-API-KEY` header:

```shell
curl.exe -s -X POST "http://127.0.0.1:8002/predict" `
  -H "X-API-KEY: your-secret-key-at-least-16-chars" `
  -F "image=@path/to/image.jpg" `
  -F "mask_dir=/data/masks" `
  -F "crop_size=512"
```

Flags:
- `X-API-KEY`: required API key (must match `INPAINT_SECRET_KEY`, or the dev key when `INPAINT_DEV_MODE=true`)
- `image`: path to the input image (absolute or relative on the client)
- `mask_dir`: directory on the **server** containing one or more `mask_*.png` files
- `crop_size`: crop size used for fill pre/post processing (`512` by default)

Example JSON response:

```json
{
  "job_id": "25acb4ae93a3419db2fcc91b3c1ee584",
  "image_url": "/jobs/25acb4ae93a3419db2fcc91b3c1ee584/inpainted.png"
}
```

Download the inpainted image:

```shell
curl.exe -H "X-API-KEY: your-secret-key-at-least-16-chars" `
  "http://127.0.0.1:8002/jobs/{job_id}/inpainted.png" `
  -o inpainted.png
```

Job outputs are written under `./data` on the host (mounted to `/data` in the container) and cleaned up after a TTL (default 600 seconds).

---

# Downloading Pre-trained Weights

When you build with Docker, **big-lama** is downloaded automatically from Hugging Face during the image build (`smartywu/big-lama`). No manual download is required for the Compose workflow.

For a non-Docker / local run, download and unpack the checkpoint so it matches:

```
pretrained_models
|-- big-lama
|   |-- config.yaml
|   |-- models
|       |-- best.ckpt
```

Manual download (Hugging Face mirror of the official Yandex release):

```shell
mkdir pretrained_models
curl.exe -L -o big-lama.zip `
  https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip
tar -xf big-lama.zip -C pretrained_models
# or: Expand-Archive big-lama.zip -DestinationPath pretrained_models
```

Default settings (override via env / `compose.yml`):

| Setting | Default |
| --- | --- |
| `INPAINT_CONFIG_DIR` | `lama/configs/prediction/default.yaml` (Compose: absolute path under `/opt/inpaint-anything/...`) |
| `INPAINT_CKPT_DIR` | `pretrained_models/big-lama/` |
| `INPAINT_DEVICE` | `cuda` |
| `INPAINT_OUTPUT_DIR` | `data/` (Compose: `/data`) |

---

# Acknowledgements

Upstream implementation: [geekyutao/Inpaint-Anything](https://github.com/geekyutao/Inpaint-Anything).

Related projects:

- [LaMa](https://github.com/advimman/lama)
- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)

# Citation

If you use this code for your research, please cite

```
@article{yu2023inpaint,
  title={Inpaint Anything: Segment Anything Meets Image Inpainting},
  author={Yu, Tao and Feng, Runseng and Feng, Ruoyu and Liu, Jinming and Jin, Xin and Zeng, Wenjun and Chen, Zhibo},
  journal={arXiv preprint arXiv:2304.06790},
  year={2023}
}
```
