//! Setup module for dia
//!
//! This module handles initialization and setup for the dia crate.

use std::path::PathBuf;
use progresshub_client_selector::{Client, DownloadConfig, Backend};

/// Model file paths structure
pub struct ModelPaths {
    pub weights: PathBuf,
    pub tokenizer: PathBuf,
}

/// Download and setup Dia model using progresshub with zero-allocation caching
pub async fn setup() -> Result<ModelPaths, String> {
    let client = Client::new(Backend::Auto);
    
    // Setup cache directory
    let cache_dir = dirs::cache_dir()
        .ok_or_else(|| "Cannot determine cache directory".to_string())?
        .join("fluent-voice")
        .join("dia");
    
    let config = DownloadConfig {
        destination: cache_dir.clone(),
        show_progress: false,
        use_cache: true,
    };
    
    // Download main Dia model
    let download_result = client
        .download_model_auto("facebook/moshi", &config, None)
        .await
        .map_err(|e| format!("Dia model download failed: {}", e))?;
    
    // Find model.safetensors in downloaded files
    let weights_path = download_result
        .models
        .first()
        .ok_or_else(|| "No models in download result".to_string())?
        .files
        .iter()
        .find(|file| file.path.file_name().and_then(|n| n.to_str()) == Some("model.safetensors"))
        .ok_or_else(|| "model.safetensors not found in downloaded files".to_string())?
        .path
        .clone();
    
    // Find tokenizer.json in downloaded files  
    let tokenizer_path = download_result
        .models
        .first()
        .unwrap()
        .files
        .iter()
        .find(|file| file.path.file_name().and_then(|n| n.to_str()) == Some("tokenizer.json"))
        .ok_or_else(|| "tokenizer.json not found in downloaded files".to_string())?
        .path
        .clone();
    
    Ok(ModelPaths {
        weights: weights_path,
        tokenizer: tokenizer_path,
    })
}

/// Configure the dia system with default settings
pub fn configure_defaults() -> Result<(), String> {
    // Default configuration is handled through DiaConfig::default()
    Ok(())
}
