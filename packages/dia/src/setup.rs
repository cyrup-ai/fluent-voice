use anyhow::Result;
use progresshub_client_selector::{Client as ProgressHubClient, MultiDownloadOrchestrator};
use progresshub_config::DownloadConfig;
use progresshub_progress::{DownloadProgress, subscribe_download_events};
use std::path::PathBuf;
use std::sync::mpsc;
use tokio::sync::broadcast;

use crate::app::ProgressUpdate;

// Model IDs
pub const DIA_MODEL: &str = "nari-labs/Dia-1.6B";
pub const ENCODEC: &str = "facebook/encodec_24khz";

/// Paths to model files that are needed for inference
#[derive(Debug, Clone)]
pub struct ModelPaths {
    pub weights: PathBuf,
    pub tokenizer: PathBuf,
    pub encodec: PathBuf,
}

/// Custom error type for model setup with semantic error handling
#[derive(Debug, thiserror::Error)]
pub enum ModelSetupError {
    #[error("Model download failed: {0}")]
    DownloadError(#[from] anyhow::Error),
    #[error("Progress reporting failed: {0}")]
    ProgressError(String),
    #[error("Model validation failed: {0}")]
    ValidationError(String),
    #[error("File system error: {0}")]
    FileSystemError(#[from] std::io::Error),
}

/// Zero-allocation progress aggregator for multi-model downloads
struct ProgressAggregator {
    dia_progress: Option<DownloadProgress>,
    encodec_progress: Option<DownloadProgress>,
    tx: mpsc::Sender<ProgressUpdate>,
}

impl ProgressAggregator {
    fn new(tx: mpsc::Sender<ProgressUpdate>) -> Self {
        Self {
            dia_progress: None,
            encodec_progress: None,
            tx,
        }
    }

    /// Update progress for a specific model and send aggregated progress
    fn update_progress(&mut self, model_id: &str, progress: DownloadProgress) -> Result<(), ModelSetupError> {
        match model_id {
            DIA_MODEL => self.dia_progress = Some(progress),
            ENCODEC => self.encodec_progress = Some(progress),
            _ => return Ok(()),
        }

        // Calculate aggregated progress
        let total_bytes = self.dia_progress.as_ref().map(|p| p.total_bytes).unwrap_or(0) +
                         self.encodec_progress.as_ref().map(|p| p.total_bytes).unwrap_or(0);
        
        let downloaded_bytes = self.dia_progress.as_ref().map(|p| p.bytes_downloaded).unwrap_or(0) +
                              self.encodec_progress.as_ref().map(|p| p.bytes_downloaded).unwrap_or(0);
        
        let avg_speed = match (&self.dia_progress, &self.encodec_progress) {
            (Some(dia), Some(encodec)) => (dia.speed_mbps + encodec.speed_mbps) / 2.0,
            (Some(dia), None) => dia.speed_mbps,
            (None, Some(encodec)) => encodec.speed_mbps,
            (None, None) => 0.0,
        };

        let aggregated_update = ProgressUpdate {
            path: format!("Downloading models ({}/{})", 
                if self.dia_progress.is_some() { 1 } else { 0 } + if self.encodec_progress.is_some() { 1 } else { 0 }, 2),
            bytes_downloaded: downloaded_bytes,
            total_bytes,
            speed_mbps: avg_speed,
        };

        self.tx.send(aggregated_update)
            .map_err(|e| ModelSetupError::ProgressError(format!("Failed to send progress update: {}", e)))?;
        
        Ok(())
    }
}

/// Zero-allocation, blazing-fast model setup with progresshub concurrent downloads
pub async fn setup(
    weights_path: Option<String>,
    tokenizer_path: Option<String>,
    tx: mpsc::Sender<ProgressUpdate>,
) -> Result<ModelPaths, ModelSetupError> {
    // Initialize progresshub client with zero-allocation configuration
    let client = ProgressHubClient::new()
        .map_err(|e| ModelSetupError::DownloadError(anyhow::anyhow!("Failed to create progresshub client: {}", e)))?;

    let download_config = DownloadConfig {
        destination: None, // Use progresshub's default cache location
        show_progress: false, // We handle progress ourselves
        use_cache: true, // Enable efficient caching and resume
    };

    // Send initial progress
    tx.send(ProgressUpdate {
        path: "Initializing model downloads".to_string(),
        bytes_downloaded: 0,
        total_bytes: 1000000000, // Approximate total size for both models
        speed_mbps: 0.0,
    }).map_err(|e| ModelSetupError::ProgressError(format!("Failed to send initial progress: {}", e)))?;

    // Set up progress aggregation with zero allocation
    let mut progress_aggregator = ProgressAggregator::new(tx.clone());
    
    // Subscribe to download events for real-time progress tracking
    let mut download_events = subscribe_download_events();

    // Create multi-download orchestrator for concurrent downloads
    let orchestrator = MultiDownloadOrchestrator::new(client);

    // Start concurrent downloads for both models
    let download_tasks = vec![
        (DIA_MODEL, download_config.clone()),
        (ENCODEC, download_config.clone()),
    ];

    let download_handle = tokio::spawn(async move {
        orchestrator.download_models(download_tasks).await
    });

    // Process progress events in real-time
    let progress_handle = tokio::spawn(async move {
        while let Ok(progress_event) = download_events.recv().await {
            if let Err(e) = progress_aggregator.update_progress(&progress_event.model_id, DownloadProgress {
                path: progress_event.file_name,
                bytes_downloaded: progress_event.bytes_downloaded,
                total_bytes: progress_event.total_bytes,
                speed_mbps: progress_event.download_speed,
            }) {
                eprintln!("Progress update error: {}", e);
            }
        }
    });

    // Wait for downloads to complete
    let download_results = download_handle.await
        .map_err(|e| ModelSetupError::DownloadError(anyhow::anyhow!("Download task failed: {}", e)))?
        .map_err(ModelSetupError::DownloadError)?;

    // Stop progress monitoring
    progress_handle.abort();

    // Extract file paths from download results
    let mut weights_file = None;
    let mut tokenizer_file = None;
    let mut encodec_file = None;

    for result in download_results {
        match result.model_id.as_str() {
            DIA_MODEL => {
                // Find weight and tokenizer files
                for path in &result.file_paths {
                    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                        if name.contains("model") && (name.ends_with(".bin") || name.ends_with(".safetensors")) {
                            weights_file = Some(path.clone());
                        } else if name.contains("tokenizer") && name.ends_with(".json") {
                            tokenizer_file = Some(path.clone());
                        }
                    }
                }
            }
            ENCODEC => {
                // Find encodec model file
                for path in &result.file_paths {
                    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                        if name == "model.safetensors" || name.contains("encodec") {
                            encodec_file = Some(path.clone());
                        }
                    }
                }
            }
            _ => continue,
        }
    }

    // Use provided paths or downloaded paths with validation
    let weights = if let Some(provided) = weights_path {
        PathBuf::from(provided)
    } else {
        weights_file.ok_or_else(|| ModelSetupError::ValidationError("DIA model weights not found in download".to_string()))?
    };

    let tokenizer = if let Some(provided) = tokenizer_path {
        PathBuf::from(provided)
    } else {
        tokenizer_file.ok_or_else(|| ModelSetupError::ValidationError("DIA tokenizer not found in download".to_string()))?
    };

    let encodec = encodec_file.ok_or_else(|| ModelSetupError::ValidationError("EnCodec model not found in download".to_string()))?;

    // Validate all files exist and are readable
    for (name, path) in [("weights", &weights), ("tokenizer", &tokenizer), ("encodec", &encodec)] {
        if !path.exists() {
            return Err(ModelSetupError::ValidationError(format!("{} file not found at: {}", name, path.display())));
        }
        if !path.is_file() {
            return Err(ModelSetupError::ValidationError(format!("{} path is not a file: {}", name, path.display())));
        }
    }

    // Send completion progress
    tx.send(ProgressUpdate {
        path: "Model setup complete".to_string(),
        bytes_downloaded: 1000000000,
        total_bytes: 1000000000,
        speed_mbps: 0.0,
    }).map_err(|e| ModelSetupError::ProgressError(format!("Failed to send completion progress: {}", e)))?;

    Ok(ModelPaths {
        weights,
        tokenizer,
        encodec,
    })
}
