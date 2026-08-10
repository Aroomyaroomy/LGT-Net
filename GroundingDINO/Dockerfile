# syntax=docker/dockerfile:1

ARG CUDA_IMAGE_TAG=12.1.1-cudnn8
ARG UBUNTU_VARIANT=ubuntu22.04

# ---------------------------------------------------------------------------
# build groundingdino._C CUDA extension
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_IMAGE_TAG}-devel-${UBUNTU_VARIANT} AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9+PTX"
ARG TORCH_VERSION=2.1.2
ARG TORCHVISION_VERSION=0.16.2
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121

# libc10.so for groundingdino._C lives under torch/lib (pip torch wheel).
ENV DEBIAN_FRONTEND=${DEBIAN_FRONTEND} \
    CUDA_HOME=/usr/local/cuda \
    PATH=${CUDA_HOME}/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:${CUDA_HOME}/lib64 \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    SETUPTOOLS_USE_DISTUTILS=stdlib \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/groundingdino

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        python-is-python3 \
        build-essential \
        ninja-build \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && nvcc --version \
    && python -m pip install --upgrade pip setuptools wheel

# Torch 2.1 needs NumPy 1.x. Pin before and after torch so torchvision cannot
# resolve NumPy 2.x from PyPI.
RUN printf 'numpy<2\n' > /tmp/constraints.txt \
    && python -m pip install -c /tmp/constraints.txt "numpy<2" \
    && python -m pip install -c /tmp/constraints.txt \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
        --index-url "${TORCH_INDEX_URL}" \
    && python -m pip install -c /tmp/constraints.txt "numpy<2" \
    && python -c "import torch, numpy as np; \
assert torch.version.cuda, 'torch build has no CUDA'; \
assert int(np.__version__.split('.', 1)[0]) < 2, np.__version__; \
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'numpy', np.__version__)"

COPY setup.py LICENSE ./
COPY groundingdino ./groundingdino

# Build/install groundingdino._C. Import-check from /tmp (avoid WORKDIR shadow)
# and import torch first so its shared libs are resolved.
RUN python -m pip install --no-build-isolation --no-deps . \
    && cd /tmp \
    && python -c "import torch; import groundingdino, groundingdino._C as c; \
assert 'dist-packages' in groundingdino.__file__, groundingdino.__file__; \
print('groundingdino._C OK', c, groundingdino.__file__)" \
    && mkdir -p /opt/export \
    && for p in /usr/local/lib/python3.10/dist-packages/*; do \
         case "$(basename "$p")" in \
           pip|pip-*|setuptools|setuptools-*|wheel|wheel-*|pkg_resources|_distutils*|easy_install.py|distutils-precedence.pth) continue ;; \
           *) cp -a "$p" /opt/export/ ;; \
         esac; \
       done \
    && test -d /opt/export/torch \
    && test -d /opt/export/torchvision \
    && ls -la /opt/export/groundingdino/_C*

# ---------------------------------------------------------------------------
# slim runtime (no compiler / CUDA toolkit); torch copied from builder
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_IMAGE_TAG}-runtime-${UBUNTU_VARIANT}

ARG DEBIAN_FRONTEND=noninteractive
ARG TORCH_VERSION=2.1.2
ARG TORCHVISION_VERSION=0.16.2

ENV DEBIAN_FRONTEND=${DEBIAN_FRONTEND} \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:${CUDA_HOME}/lib64 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    GROUNDINGDINO_ROOT=/opt/groundingdino \
    DINO_CONFIG_DIR=/opt/groundingdino/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    DINO_CKPT_DIR=/opt/groundingdino/weights/groundingdino_swint_ogc.pth \
    DINO_OUTPUT_DIR=/data

WORKDIR /opt/groundingdino

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python-is-python3 \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --upgrade pip \
    && mkdir -p /opt/groundingdino/weights /data /opt/hf-cache

# CUDA torch + torchvision + groundingdino._C (+ torch pip deps) from builder.
COPY --from=builder /opt/export/ /usr/local/lib/python3.10/dist-packages/

COPY requirements.txt ./

# Constrain torch/numpy so transformers/timm/opencv cannot replace CUDA torch
# with a CPU wheel or pull NumPy 2.x. Verify imports from /tmp (no cwd shadow).
RUN printf 'torch==%s\ntorchvision==%s\nnumpy<2\n' \
        "${TORCH_VERSION}" "${TORCHVISION_VERSION}" > /tmp/constraints.txt \
    && python -m pip install -c /tmp/constraints.txt -r requirements.txt \
    && find /usr/local/lib/python3*/dist-packages -name '*.pyc' -delete \
    && (find /usr/local/lib/python3*/dist-packages -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true) \
    && ln -sfn /usr/local/lib/python3.10/dist-packages/groundingdino /opt/groundingdino/groundingdino \
    && cd /tmp \
    && python -c "import torch; import numpy as np; import groundingdino, groundingdino._C as c; \
assert torch.version.cuda, 'runtime torch has no CUDA'; \
assert int(np.__version__.split('.', 1)[0]) < 2, np.__version__; \
assert 'dist-packages' in groundingdino.__file__, groundingdino.__file__; \
print('runtime OK', c, 'torch', torch.__version__, 'cuda', torch.version.cuda, 'numpy', np.__version__)"

COPY app.py app_utils.py ./

# ---------------------------------------------------------------------------
# Weights are NOT baked into the image — mount ./weights:/opt/groundingdino/weights at runtime.
# Optional pre-download: docker compose build --build-arg PRELOAD_MODELS=true
# ---------------------------------------------------------------------------
RUN mkdir -p /opt/groundingdino/weights
ARG PRELOAD_MODELS=false
ARG DINO_WEIGHT_URL=https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth
RUN if [ "$PRELOAD_MODELS" = "true" ]; then \
        echo "Downloading GroundingDINO weights (662 MB) ..." && \
        wget -q --show-progress \
            -O /opt/groundingdino/weights/groundingdino_swint_ogc.pth \
            "${DINO_WEIGHT_URL}" && \
        ls -lh /opt/groundingdino/weights/ && \
        echo "GroundingDINO weights ready."; \
    else \
        echo "PRELOAD_MODELS=false — weights must be mounted at /opt/groundingdino/weights"; \
    fi

EXPOSE 8001

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
