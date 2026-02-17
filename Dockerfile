# ── Base image: CUDA 12.8 + cuDNN, Ubuntu 22.04 ──────────────────────────────
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

LABEL org.opencontainers.image.title="vLLM 0.14.0 Pascal Patch" \
      org.opencontainers.image.description="vLLM 0.14.0 patched for Pascal-architecture GPUs (P40, GTX 10xx)" \
      org.opencontainers.image.source="https://github.com/shu1qin9/vllm-0.14.0-pascal"

# ── Build-time args ────────────────────────────────────────────────────────────
# Override RELEASE_TAG to target a different release
ARG RELEASE_TAG=v0.14.0-pascal
ARG RELEASE_BASE=https://github.com/shu1qin9/vllm-0.14.0-pascal/releases/download/${RELEASE_TAG}

ARG VLLM_WHL=vllm-0.14.0rc1.dev122+g42826bbcc.d20251226.cu128-cp312-cp312-linux_x86_64.whl
ARG TORCH_WHL=torch-2.9.1a0+gitd38164a-cp312-cp312-linux_x86_64.whl
ARG TRITON_WHL=triton-3.6.0+git9e449252-cp312-cp312-linux_x86_64.whl
ARG PYTHON_VERSION=3.12

# ── System dependencies ────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python${PYTHON_VERSION} \
      python${PYTHON_VERSION}-venv \
      python3-pip \
      curl \
      git \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
 && update-alternatives --install /usr/bin/python  python  /usr/bin/python${PYTHON_VERSION} 1

# ── Download patched wheels from GitHub Releases ───────────────────────────────
WORKDIR /wheels
RUN curl -fSL "${RELEASE_BASE}/${VLLM_WHL}"   -o vllm.whl   \
 && curl -fSL "${RELEASE_BASE}/${TORCH_WHL}"  -o torch.whl  \
 && curl -fSL "${RELEASE_BASE}/${TRITON_WHL}" -o triton.whl

# ── Install in the correct order ───────────────────────────────────────────────
# 1. vllm first (it pulls in stock torch + triton as deps)
# 2. uninstall stock torch/triton
# 3. install the patched builds
RUN pip install --no-cache-dir vllm.whl \
 && pip uninstall -y torch triton \
 && pip install --no-cache-dir --no-deps triton.whl \
 && pip install --no-cache-dir --no-deps torch.whl \
 && rm -rf /wheels

# ── Runtime ────────────────────────────────────────────────────────────────────
WORKDIR /app
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]
CMD ["--help"]