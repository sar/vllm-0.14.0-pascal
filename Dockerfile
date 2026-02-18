# ── Base image: CUDA 12.8 + cuDNN runtime, Ubuntu 24.04 ──────────────────────
# The -runtime variant includes cudart + cudnn but NOT cupti, nvrtc, cufft etc.
# We install the missing CUDA 12.8 apt packages below from NVIDIA's repo,
# which is pre-configured in all nvidia/cuda Docker images.
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

ENV VENV=/opt/venv
ENV DEBIAN_FRONTEND=noninteractive

# ── System + CUDA runtime dependencies ────────────────────────────────────────
# The patched torch (installed with --no-deps) links against system CUDA libs
# rather than the pip-bundled nvidia-*-cu12 packages. We install the full set
# from NVIDIA's apt repo (pre-configured in all nvidia/cuda base images).
RUN apt-get update && apt-get install -y --no-install-recommends --allow-change-held-packages \
    # ── Python ──────────────────────────────────────────────────────────────
    python3 \
    python3-venv \
    python3-pip \
    curl \
    git \
    # ── Core system runtime libs ─────────────────────────────────────────────
    libgomp1 \
    libstdc++6 \
    libgfortran5 \
    libnuma1 \
    libibverbs1 \
    libopenmpi3 \
    libtinfo6 \
    procps \
    # ── OpenCV runtime (opencv-python-headless is a vLLM dep) ────────────────
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    # ── Image codecs (Pillow dep) ────────────────────────────────────────────
    libjpeg-turbo8 \
    libpng16-16 \
    libwebp7 \
    # ── CUDA 12.8 libraries missing from the -runtime base image ─────────────
    # libcupti.so.12  — CUDA Profiling Tools, required by torch on import
    cuda-cupti-12-8 \
    # libnvrtc.so.12  — CUDA Runtime Compilation, required by torch/triton JIT
    cuda-nvrtc-12-8 \
    # libcufft.so.11  — CUDA FFT
    libcufft-12-8 \
    # libcurand.so.10 — CUDA Random Number Generation
    libcurand-12-8 \
    # libcusolver.so.11 — CUDA Dense/Sparse linear algebra
    libcusolver-12-8 \
    # libcusparse.so.12 — CUDA Sparse BLAS
    libcusparse-12-8 \
    # libnccl.so.2    — NVIDIA Collective Communications (multi-GPU)
    libnccl2 \
    # libnvToolsExt   — NVTX profiling markers used by torch
    cuda-nvtx-12-8 \
  && rm -rf /var/lib/apt/lists/*

# ── Update dynamic linker cache so all newly installed .so files are found ────
RUN ldconfig

# ── Create virtualenv — sidesteps PEP 668 externally-managed-environment ──────
RUN python3 -m venv ${VENV}
ENV PATH="${VENV}/bin:${PATH}"

# ── Download patched wheels from GitHub Releases ───────────────────────────────
WORKDIR /wheels
RUN curl -fSL "${RELEASE_BASE}/${VLLM_WHL}"   -o "${VLLM_WHL}"   \
 && curl -fSL "${RELEASE_BASE}/${TORCH_WHL}"  -o "${TORCH_WHL}"  \
 && curl -fSL "${RELEASE_BASE}/${TRITON_WHL}" -o "${TRITON_WHL}"

# ── Install in the correct order ───────────────────────────────────────────────
# 1. vllm (pulls in stock torch + triton as deps, plus all nvidia-*-cu12 pkgs)
# 2. uninstall stock torch/triton (keep the nvidia-*-cu12 pip packages)
# 3. install patched torch/triton with --no-deps
RUN pip install --no-cache-dir "${VLLM_WHL}" \
 && pip uninstall -y torch triton \
 && pip install --no-cache-dir --no-deps "${TRITON_WHL}" \
 && pip install --no-cache-dir --no-deps "${TORCH_WHL}" \
 && rm -rf /wheels

# ── Build-time sanity check ────────────────────────────────────────────────────
# Imports torch and triton to verify all .so files resolve.
# vllm cannot be imported here — it requires a real GPU on import.
# If any shared lib is still missing (libcupti, libnvrtc, etc.) this fails fast.
RUN python -c "import torch; print('torch:', torch.__version__); print('CUDA available (no GPU at build time):', torch.cuda.is_available()); print('Shared library check OK')"
RUN python -c "import triton; print('triton:', triton.__version__); print('triton OK')"

# ── Runtime ────────────────────────────────────────────────────────────────────
WORKDIR /app
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]
CMD ["--help"]