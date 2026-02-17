# Overview

This document covers the changes made to vLLM and related packages to add support for Pascal-architecture GPUs, specifically:

- [x] vLLM 0.14.0 — patched & built
- [x] PyTorch 2.9.1 — patched & built
- [x] Triton 3.6.0 — patched & built

The patching approach is based on [vllm-pascal](https://github.com/ampir-nn/vllm-pascal), updated to handle syntax changes in the newer versions.

# Details

## Background

I run a local compute setup using an NVIDIA P40. When trying to deploy models with vLLM, the P40's older Pascal architecture causes a number of compatibility errors out of the box.

I found an existing project — [vllm-pascal](https://github.com/ampir-nn/vllm-pascal) — that patches vLLM 0.10.2 and 0.11.0 for Pascal, but those older versions hit API compatibility issues with newer models like `AutoGLM-Phone-9B`. So I applied the same patching approach to build the latest vLLM from scratch.

---

## Option 1: Docker (Recommended)

A Docker image is automatically built and published to the GitHub Container Registry (GHCR) via GitHub Actions whenever a new release is tagged.

The image is based on `nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04` and comes with all three patched wheels pre-installed in the correct order — no manual setup required.

### Prerequisites

Your host machine needs `nvidia-container-toolkit` installed so Docker can access the GPU:

```shell
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Pull the image

```shell
docker pull ghcr.io/<your-github-username>/vllm-0.14.0-pascal:latest
```

> If the package is private, authenticate first:
> ```shell
> echo "<your-github-pat>" | docker login ghcr.io -u <your-github-username> --password-stdin
> ```
> Or go to your repo → **Packages** → the image → **Package settings** → set visibility to **Public**.

### Run

```shell
docker run --rm --gpus all \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/<your-github-username>/vllm-0.14.0-pascal:latest \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dtype float16
```

Key flags:
- `--gpus all` — exposes your Pascal GPU(s) to the container
- `-p 8000:8000` — exposes the OpenAI-compatible REST API on the host
- `-v ~/.cache/huggingface:...` — mounts your local model cache so models aren't re-downloaded on every run

### Test it's working

```shell
curl http://localhost:8000/v1/models
```

### How the CI build works

The GitHub Actions workflow (`.github/workflows/docker-build.yml`) triggers on every push to `master` and on any `v*` tag. It:

1. Resolves which release tag to pull wheels from (defaults to `v0.14.0-pascal`, or the pushed git tag if triggered by a tag push)
2. Verifies all three `.whl` assets are accessible on the release before starting the build
3. Builds the Docker image — the `Dockerfile` downloads the wheels directly from the GitHub Release via `curl` during the build, so no large files need to be checked into the repo
4. Pushes the image to GHCR tagged with `latest`, the release version (`0.14.0-pascal`), and a short git SHA for traceability

You can also trigger a build manually from the **Actions** tab in GitHub, with the option to specify a different release tag or skip pushing the image.

---

## Option 2: Manual Installation

If you'd prefer to install the wheels directly into a local Python environment:

### Download the wheels

Grab the three `.whl` files from the [latest release](https://github.com/shu1qin9/vllm-0.14.0-pascal/releases/tag/v0.14.0-pascal):

![image.png](https://lsq-markdown.oss-cn-hangzhou.aliyuncs.com/wiki20251226171434.png)

### Install

```shell
# Create virtual environment
python3 -m venv pyenv-P40
source pyenv-p40/bin/activate

# Install patched vLLM
pip install vllm-0.14.0rc1.dev122+g42826bbcc.d20251226.cu128-cp312-cp312-linux_x86_64.whl
```

> **Important:** Installing vLLM will automatically pull in stock versions of torch and triton as dependencies. You need to replace these with the patched builds:

```shell
# Remove stock torch and triton, install patched versions
pip uninstall torch triton -y

pip install triton-3.6.0+git9e449252-cp312-cp312-linux_x86_64.whl
pip install torch-2.9.1a0+gitd38164a-cp312-cp312-linux_x86_64.whl
```

> **Note:** You may see a dependency version error when installing the patched triton — this can be safely ignored.

Once everything is installed, run `pip list` and verify the three patched packages are present:

```shell
torch                             2.9.1a0+gitd38164a
triton                            3.6.0+git9e449252
vllm                              0.14.0rc1.dev122+g42826bbcc.d20251226.cu128
```

![image.png](https://lsq-markdown.oss-cn-hangzhou.aliyuncs.com/wiki20251226171940.png)

You can now launch vLLM and serve models as normal.