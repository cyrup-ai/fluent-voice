#!/usr/bin/env python3
"""
Script to download the Dia TTS model from Hugging Face Hub.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

def download_model(model_id: str, cache_dir: Path) -> Path:
    """
    Download the model from Hugging Face Hub.
    
    Args:
        model_id: The Hugging Face model ID (e.g., "nari-labs/Dia-1.6B-0626")
        cache_dir: Directory to download the model to
        
    Returns:
        Path to the downloaded model directory
    """
    print(f"Downloading model {model_id} to {cache_dir}...")
    
    # Create the cache directory if it doesn't exist
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download the model with progress bars
        model_path = snapshot_download(
            repo_id=model_id,
            local_dir=cache_dir,
            local_dir_use_symlinks=False,  # Download actual files, not symlinks
            resume_download=True,  # Resume interrupted downloads
            allow_patterns=["*.json", "*.bin", "*.safetensors", "*.md", "*.txt"],
            ignore_patterns=["*.h5", "*.ot", "*.msgpack", "*.tflite", "*.onnx"],
        )
        print(f"Model downloaded to: {model_path}")
        return Path(model_path)
    except Exception as e:
        print(f"Error downloading model: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    # Model ID for Dia TTS
    model_id = "nari-labs/Dia-1.6B-0626"
    
    # Default cache directory: ~/.cache/dia/models
    cache_dir = Path.home() / ".cache" / "dia" / "models"
    
    # Allow overriding the cache directory with an environment variable
    if "HF_HOME" in os.environ:
        cache_dir = Path(os.environ["HF_HOME"]) / "hub" / f"models--{model_id.replace('/', '--')}"
    
    print(f"Using cache directory: {cache_dir}")
    
    # Download the model
    model_path = download_model(model_id, cache_dir)
    
    # Verify the downloaded files
    required_files = [
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "tokenizer_config.json",
        "special_tokens_map.json"
    ]
    
    missing_files = []
    for file in required_files:
        if not (model_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"Warning: The following required files are missing: {', '.join(missing_files)}")
        print("The model may not work correctly without these files.")
    else:
        print("All required model files are present.")
    
    print("\nTo use this model with the Dia TTS system, set the following environment variable:")
    print(f"export DIA_MODEL_PATH=\"{model_path}\"")

if __name__ == "__main__":
    main()
