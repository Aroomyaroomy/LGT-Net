# syntax=docker/dockerfile:1.7
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates git \
    libglib2.0-0 libgomp1 libgl1 \
    && update-ca-certificates

RUN python -m pip install --upgrade pip uv

COPY requirements-uv.txt /app/requirements-uv.txt

# The PyTorch base image already provides torch 2.4.1 + torchvision 0.19.1.
# Install the remaining app dependencies without redownloading the CUDA stack.
RUN --mount=type=cache,target=/root/.cache/uv \
    grep -Ev "^(torch|torchvision)==" /app/requirements-uv.txt > /tmp/requirements-no-torch.txt && \
    python -c "import torch, torchvision; assert torch.__version__.startswith('2.4.1'), torch.__version__; assert torchvision.__version__.startswith('0.19.1'), torchvision.__version__" && \
    uv pip install --system \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    --index-strategy unsafe-best-match \
    -r /tmp/requirements-no-torch.txt

COPY app.py inference.py /app/
COPY config /app/config
COPY dataset /app/dataset
COPY evaluation /app/evaluation
COPY loss /app/loss
COPY models /app/models
COPY postprocessing /app/postprocessing
COPY preprocessing /app/preprocessing
COPY utils /app/utils
COPY visualization /app/visualization
COPY src/config /app/src/config

# explicitly create output and checkpoint dirs
RUN mkdir -p /app/src/output /app/checkpoints

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]