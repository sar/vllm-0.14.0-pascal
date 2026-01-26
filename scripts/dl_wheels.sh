#!/bin/bash

# ==========================================
# CONFIGURATION
# ==========================================

# The name of the requirements file you created in Step 1
REQUIREMENTS_FILE="vllm-requirements.txt"

# The folder where wheels will be saved
WHEEL_DIR="./vllm_offline_wheels"

# Python version on the target system (e.g., 310, 311, 312)
# Ensure you download for the Python version you intend to use on the offline machine.
PYTHON_VERSION="310"

# --- CUDA CONFIGURATION ---
# If your offline machine has GPU, you likely need the CUDA wheels for PyTorch.
# Uncomment the line below that matches your offline CUDA version (e.g., cu121, cu124)
# If you leave it empty, pip will try to download standard wheels (often CPU-only for Torch).

# PYTORCH_INDEX_URL="--extra-index-url https://download.pytorch.org/whl/cu121"
PYTORCH_INDEX_URL="" 

# ==========================================
# SCRIPT LOGIC
# ==========================================

echo "Starting dependency download..."
echo "Requirements file: $REQUIREMENTS_FILE"
echo "Destination directory: $WHEEL_DIR"
echo "Target Python Version: 3.$PYTHON_VERSION"

# Check if requirements file exists
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "Error: $REQUIREMENTS_FILE not found!"
    echo "Please create it with the dependencies list."
    exit 1
fi

# Create the destination directory
mkdir -p "$WHEEL_DIR"

echo "------------------------------------------------"
echo "Downloading wheels..."
echo "------------------------------------------------"

# Run pip download
# --only-binary=:all: ensures we get .whl files, not source tarballs (faster/cleaner)
# --python-version ensures we get wheels for the specific Python version
# --platform ensures we get wheels for the current architecture (Linux x86_64 usually)

pip download \
    -r "$REQUIREMENTS_FILE" \
    --dest "$WHEEL_DIR" \
    --only-binary=:all: \
    $PYTORCH_INDEX_URL

# Check if download was successful
if [ $? -eq 0 ]; then
    echo "------------------------------------------------"
    echo "Success! All dependencies downloaded to: $WHEEL_DIR"
    echo "You can now copy this folder to your offline system."
    echo ""
    echo "To install on the offline system, run:"
    echo "pip install --no-index --find-links=$WHEEL_DIR /path/to/your/custom_vllm.whl"
else
    echo "------------------------------------------------"
    echo "Error during download. Please check the output above."
    exit 1
fi