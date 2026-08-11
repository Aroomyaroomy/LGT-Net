# Inpaint-Anything FastAPI inference (LaMa fill path only).
#
# Build / run (from this directory):
#   docker compose up --build
#
# Or without Compose:
#   docker build -t inpaint-anything:inference .
#   docker run --rm --gpus all -p 8002:8002 -v ${PWD}/data:/data inpaint-anything:inference

FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive

ENV DEBIAN_FRONTEND=${DEBIAN_FRONTEND} \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_HOME=/opt/torch-cache \
    INPAINT_ANYTHING_ROOT=/opt/inpaint-anything \
    LAMA_CONFIG=/opt/inpaint-anything/lama/configs/prediction/default.yaml \
    LAMA_CKPT=/opt/inpaint-anything/pretrained_models/big-lama \
    PYTHONPATH=/opt/inpaint-anything:/opt/inpaint-anything/lama

WORKDIR /opt/inpaint-anything

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
        curl \
        unzip \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/* \
    && python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# Minimal code for app.py + LaMa inpainting.
COPY lama ./lama
COPY utils ./utils
COPY app.py lama_inpaint.py ./

# LaMa requirements from upstream, minus packages provided by the base image
# and tensorflow (not required for LaMa inference).
# Pin numpy<2: torch 2.1 wheels expect NumPy 1.x (_ARRAY_API).
RUN grep -vE '^(torch|torchvision|torchaudio|tensorflow|numpy|scikit-learn|scikit-image)([=<[:space:]]|$)' lama/requirements.txt \
        > /tmp/lama.requirements.txt \
    && python -m pip install --no-cache-dir "setuptools" "numpy<2" "scikit-learn" "scikit-image" -r /tmp/lama.requirements.txt \
    && rm /tmp/lama.requirements.txt \
    && echo "from .utils import *" > utils/__init__.py \
    && python -c "from utils import load_img_to_array; print('deps OK')"

# FastAPI inference server (app.py) — not in upstream LaMa requirements.
RUN python -m pip install --no-cache-dir \
        "fastapi" \
        "uvicorn[standard]" \
        "pydantic-settings" \
        "python-multipart" \
    && python -c "import fastapi, uvicorn, pydantic_settings; print('api deps OK')"

# big-lama checkpoint (Hugging Face mirror of the official Yandex release).
RUN mkdir -p pretrained_models \
    && wget -q -O /tmp/big-lama.zip \
         https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip \
    && unzip -q /tmp/big-lama.zip -d pretrained_models \
    && rm /tmp/big-lama.zip \
    && test -f pretrained_models/big-lama/models/best.ckpt \
    && test -f pretrained_models/big-lama/config.yaml \
    && echo "big-lama ready"

EXPOSE 8002

CMD ["python", "app.py"]
