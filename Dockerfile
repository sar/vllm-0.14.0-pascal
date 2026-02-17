# ── Base image: CUDA 12.8 + cuDNN, Ubuntu 24.04 ──────────────────────────────
# Ubuntu 24.04 ships Python 3.12 natively — no PPA needed
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

LABEL org.opencontainers.image.title="vLLM 0.14.0 Pascal Patch" \
      org.opencontainers.image.description="vLLM 0.14.0 patched for Pascal-architecture GPUs (P40, GTX 10xx)" \
      org.opencontainers.image.source="https://github.com/shu1qin9/vllm-0.14.0-pascal"

# ── Build-time args ────────────────────────────────────────────────────────────
ARG RELEASE_TAG=v0.14.0-pascal
ARG RELEASE_BASE=https://github.com/shu1qin9/vllm-0.14.0-pascal/releases/download/${RELEASE_TAG}

ARG VLLM_WHL=vllm-0.14.0rc1.dev122+g42826bbcc.d20251226.cu128-cp312-cp312-linux_x86_64.whl
ARG TORCH_WHL=torch-2.9.1a0+gitd38164a-cp312-cp312-linux_x86_64.whl
ARG TRITON_WHL=triton-3.6.0+git9e449252-cp312-cp312-linux_x86_64.whl

# ── System dependencies ────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 \
      python3-venv \
      python3-pip \
      curl \
      git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

# ── Download patched wheels from GitHub Releases ───────────────────────────────
# Keep original filenames so pip can validate the wheel format
WORKDIR /wheels
RUN curl -fSL "${RELEASE_BASE}/${VLLM_WHL}"   -o "${VLLM_WHL}"   \
 && curl -fSL "${RELEASE_BASE}/${TORCH_WHL}"  -o "${TORCH_WHL}"  \
 && curl -fSL "${RELEASE_BASE}/${TRITON_WHL}" -o "${TRITON_WHL}"

# ── Install in the correct order ───────────────────────────────────────────────
RUN pip install --no-cache-dir --break-system-packages "${VLLM_WHL}" \
 && pip uninstall -y torch triton \
 && pip install --no-cache-dir --break-system-packages --no-deps "${TRITON_WHL}" \
 && pip install --no-cache-dir --break-system-packages --no-deps "${TORCH_WHL}" \
 && rm -rf /wheels

# ── Runtime ────────────────────────────────────────────────────────────────────
WORKDIR /app
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]
CMD ["--help"]