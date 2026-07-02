FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# root working directory
WORKDIR /app

# install OS-level depedencies 
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 python3.8-dev python3-pip python3-setuptools python3-wheel \
    curl ca-certificates git \
    libglib2.0-0 libgomp1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# explicitly set python3 to python3.8
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

RUN python3 -m pip install --upgrade pip uv

COPY requirements-uv-cu110.txt /app/requirements-uv-cu110.txt

# install torch + torchvision
RUN uv pip install --system \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cu110 \
    torch==1.7.1+cu110 torchvision==0.8.2+cu110

# create temp requirements file without torch + torchvision then install
RUN grep -Ev "^(torch|torchvision)==" /app/requirements-uv-cu110.txt > /tmp/requirements-no-torch.txt && \
    uv pip install --system \
    --index-url https://pypi.org/simple \
    -r /tmp/requirements-no-torch.txt

COPY . /app

# explicitly create output and checkpoint dirs
RUN mkdir -p /app/src/output /app/checkpoints

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]