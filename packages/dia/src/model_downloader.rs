//! Model downloader module for dia
//!
//! This module handles downloading and managing AI models for the dia crate.

use std::path::Path;

/// Download a model using progresshub with resume support and validation
pub async fn download_model(
    model_name: &str,
    destination: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::process::Command;

    // Use progresshub CLI for efficient model downloading
    let output = Command::new("progresshub")
        .arg("download")
        .arg(model_name)
        .arg("--destination")
        .arg(destination)
        .arg("--resume") // Support interrupted downloads
        .arg("--validate") // Ensure file integrity
        .output()
        .map_err(|e| format!("Failed to execute progresshub: {e}"))?;

    if output.status.success() {
        tracing::info!(
            "Successfully downloaded model '{}' to {:?}",
            model_name,
            destination
        );
        Ok(())
    } else {
        let error_msg = String::from_utf8_lossy(&output.stderr);
        tracing::error!("Failed to download model '{model_name}': {error_msg}");
        Err(format!("Model download failed: {error_msg}").into())
    }
}

/// Check if a model exists locally with proper validation
pub fn model_exists(model_path: &Path) -> bool {
    if !model_path.exists() {
        return false;
    }

    // For neural network models, check for required files
    let required_files = [
        "config.json",       // Model configuration
        "model.safetensors", // Primary model weights
    ];

    // Alternative weight files if safetensors not available
    let alternative_files = ["pytorch_model.bin", "model.bin"];

    // Check required files exist
    let has_config = model_path.join("config.json").exists();
    let has_weights = required_files.iter().any(|f| model_path.join(f).exists())
        || alternative_files
            .iter()
            .any(|f| model_path.join(f).exists());

    has_config && has_weights
}
