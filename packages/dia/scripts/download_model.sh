#!/bin/bash

# Exit on error
set -e

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed." >&2
    exit 1
fi

# Create and activate a virtual environment
echo "🔧 Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "🔄 Upgrading pip..."
pip install --upgrade pip

# Install required packages
echo "📦 Installing required packages..."
pip install huggingface-hub

# Run the download script
echo "🚀 Starting model download..."
python download_model.py

# Deactivate the virtual environment
deactivate

echo "✅ Done! Model download complete."
echo "To use the downloaded model, set the DIA_MODEL_PATH environment variable to:"
echo "$(pwd)/venv/lib/python3.$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[1:2])))')/site-packages/huggingface_hub/hf_hub"
