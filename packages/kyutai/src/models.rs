//! Model downloading and management for Kyutai integration
//!
//! This module provides production-quality model downloading using progresshub
//! for the Kyutai Moshi language model and associated components.

use crate::error::MoshiError;
use anyhow::Result;
use progresshub::{DownloadResult, ModelResult, ProgressHub};
use std::path::PathBuf;
use tracing::{error, info, warn};

/// Configuration for Kyutai model downloading
#[derive(Debug, Clone)]
pub struct KyutaiModelConfig {
    /// Moshi language model repository
    pub moshi_repo: String,
    /// TTS voices repository for speaker embeddings
    pub tts_voices_repo: String,
    /// Force redownload even if cached
    pub force_download: bool,
}

impl Default for KyutaiModelConfig {
    fn default() -> Self {
        Self {
            moshi_repo: "kyutai/moshika-pytorch-bf16".to_string(),
            tts_voices_repo: "kyutai/tts-voices".to_string(),
            force_download: false,
        }
    }
}

/// Downloaded Kyutai model paths
#[derive(Debug, Clone)]
pub struct KyutaiModelPaths {
    /// Path to the language model file (lm_final.safetensors)
    pub lm_model_path: PathBuf,
    /// Path to the Mimi codec file (mimi-v0_1.safetensors)
    pub mimi_model_path: PathBuf,
    /// Path to the TTS voices directory
    pub tts_voices_path: PathBuf,
    /// Base directory where Moshi model was downloaded
    pub moshi_base_path: PathBuf,
}

/// Downloads and manages Kyutai models
pub struct KyutaiModelManager {
    config: KyutaiModelConfig,
}

impl KyutaiModelManager {
    /// Create a new model manager with default configuration
    pub fn new() -> Self {
        Self {
            config: KyutaiModelConfig::default(),
        }
    }

    /// Create a new model manager with custom configuration
    pub fn with_config(config: KyutaiModelConfig) -> Self {
        Self { config }
    }

    /// Download all required Kyutai models
    pub async fn download_models(&self) -> Result<KyutaiModelPaths, MoshiError> {
        info!("Starting Kyutai model downloads...");

        // Download Moshi language model
        let moshi_result = self.download_moshi_model().await?;
        let moshi_base_path = self.extract_model_path(&moshi_result)?;

        // Download TTS voices for speaker conditioning
        let tts_voices_result = self.download_tts_voices().await?;
        let tts_voices_path = self.extract_model_path(&tts_voices_result)?;

        // Construct paths to specific model files
        let lm_model_path = moshi_base_path.join("lm_final.safetensors");
        let mimi_model_path = moshi_base_path.join("mimi-v0_1.safetensors");

        // Validate that the expected files exist
        self.validate_model_files(&lm_model_path, &mimi_model_path, &tts_voices_path)?;

        let paths = KyutaiModelPaths {
            lm_model_path,
            mimi_model_path,
            tts_voices_path,
            moshi_base_path,
        };

        info!("Successfully downloaded all Kyutai models");
        info!("LM model: {}", paths.lm_model_path.display());
        info!("Mimi codec: {}", paths.mimi_model_path.display());
        info!("TTS voices: {}", paths.tts_voices_path.display());

        Ok(paths)
    }

    /// Download the Moshi language model
    async fn download_moshi_model(&self) -> Result<DownloadResult, MoshiError> {
        info!("Downloading Moshi model from {}", self.config.moshi_repo);

        let mut builder = ProgressHub::builder().model(&self.config.moshi_repo);

        if self.config.force_download {
            builder = builder.force(true);
        }

        let result = builder.with_cli_progress().download().await.map_err(|e| {
            error!("Failed to download Moshi model: {}", e);
            MoshiError::ModelLoad(format!("Moshi model download failed: {}", e))
        })?;

        // Extract the single result from OneOrMany
        match result.into_iter().next() {
            Some(download_result) => {
                info!(
                    "Moshi model downloaded successfully: {} bytes",
                    download_result.total_downloaded_bytes
                );
                Ok(download_result)
            }
            None => {
                error!("No Moshi model download result received");
                Err(MoshiError::ModelLoad(
                    "No Moshi model downloaded".to_string(),
                ))
            }
        }
    }

    /// Download TTS voices for speaker conditioning
    async fn download_tts_voices(&self) -> Result<DownloadResult, MoshiError> {
        info!(
            "Downloading TTS voices from {}",
            self.config.tts_voices_repo
        );

        let mut builder = ProgressHub::builder().model(&self.config.tts_voices_repo);

        if self.config.force_download {
            builder = builder.force(true);
        }

        let result = builder.with_cli_progress().download().await.map_err(|e| {
            error!("Failed to download TTS voices: {}", e);
            MoshiError::ModelLoad(format!("TTS voices download failed: {}", e))
        })?;

        // Extract the single result from OneOrMany
        match result.into_iter().next() {
            Some(download_result) => {
                info!(
                    "TTS voices downloaded successfully: {} bytes",
                    download_result.total_downloaded_bytes
                );
                Ok(download_result)
            }
            None => {
                error!("No TTS voices download result received");
                Err(MoshiError::ModelLoad(
                    "No TTS voices downloaded".to_string(),
                ))
            }
        }
    }

    /// Extract model cache path from download result
    fn extract_model_path(&self, result: &DownloadResult) -> Result<PathBuf, MoshiError> {
        match &result.models {
            progresshub::ZeroOneOrMany::One(model) => Ok(model.model_cache_path.clone()),
            progresshub::ZeroOneOrMany::Many(models) => {
                if let Some(model) = models.first() {
                    Ok(model.model_cache_path.clone())
                } else {
                    Err(MoshiError::ModelLoad(
                        "No models in download result".to_string(),
                    ))
                }
            }
            progresshub::ZeroOneOrMany::Zero => {
                Err(MoshiError::ModelLoad("No models downloaded".to_string()))
            }
        }
    }

    /// Validate that all required model files exist
    fn validate_model_files(
        &self,
        lm_model_path: &PathBuf,
        mimi_model_path: &PathBuf,
        tts_voices_path: &PathBuf,
    ) -> Result<(), MoshiError> {
        // Check LM model file
        if !lm_model_path.exists() {
            return Err(MoshiError::ModelLoad(format!(
                "Language model file not found: {}",
                lm_model_path.display()
            )));
        }

        // Check Mimi codec file
        if !mimi_model_path.exists() {
            return Err(MoshiError::ModelLoad(format!(
                "Mimi codec file not found: {}",
                mimi_model_path.display()
            )));
        }

        // Check TTS voices directory
        if !tts_voices_path.exists() {
            return Err(MoshiError::ModelLoad(format!(
                "TTS voices directory not found: {}",
                tts_voices_path.display()
            )));
        }

        // Validate TTS voices structure (should contain voice folders)
        let voice_folders = ["vctk", "expresso", "cml-tts", "ears", "voice-donations"];
        let mut found_voices = 0;

        for folder in &voice_folders {
            let voice_path = tts_voices_path.join(folder);
            if voice_path.exists() && voice_path.is_dir() {
                found_voices += 1;
            }
        }

        if found_voices == 0 {
            warn!("No expected voice folders found in TTS voices directory");
        } else {
            info!(
                "Found {} voice folders in TTS voices directory",
                found_voices
            );
        }

        Ok(())
    }
}

impl Default for KyutaiModelManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function to download Kyutai models with default configuration
pub async fn download_kyutai_models() -> Result<KyutaiModelPaths, MoshiError> {
    let manager = KyutaiModelManager::new();
    manager.download_models().await
}

/// Convenience function to download Kyutai models with custom configuration
pub async fn download_kyutai_models_with_config(
    config: KyutaiModelConfig,
) -> Result<KyutaiModelPaths, MoshiError> {
    let manager = KyutaiModelManager::with_config(config);
    manager.download_models().await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = KyutaiModelConfig::default();
        assert_eq!(config.moshi_repo, "kyutai/moshika-pytorch-bf16");
        assert_eq!(config.tts_voices_repo, "kyutai/tts-voices");
        assert!(!config.force_download);
    }

    #[test]
    fn test_custom_config() {
        let config = KyutaiModelConfig {
            moshi_repo: "kyutai/moshiko-pytorch-q8".to_string(),
            tts_voices_repo: "kyutai/tts-voices".to_string(),
            force_download: true,
        };

        let manager = KyutaiModelManager::with_config(config.clone());
        assert_eq!(manager.config.moshi_repo, config.moshi_repo);
        assert_eq!(manager.config.force_download, config.force_download);
    }
}
