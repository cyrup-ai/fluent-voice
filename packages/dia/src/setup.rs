use anyhow::{Context, Result, anyhow};
use progresshub::ModelDownloader;
use std::path::PathBuf;
use std::sync::mpsc;

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

/// Set up models for inference by downloading them if needed
pub async fn setup(
    weights_path: Option<String>,
    tokenizer_path: Option<String>,
    _tx: mpsc::Sender<ProgressUpdate>,
) -> Result<ModelPaths> {
    let download_results = ModelDownloader::new()
        .model(DIA_MODEL)
        .model(ENCODEC)
        .download()
        .await?;

    // Find DIA model result by looking through all models in all download results
    let dia_result = download_results
        .iter()
        .flat_map(|result| &result.models)
        .find(|model| model.path.to_string_lossy().contains("Dia-1.6B"))
        .with_context(|| format!("Model path for {} not found in download results", DIA_MODEL))?;

    // Find EnCodec result by looking through all models in all download results
    let encodec_result = download_results
        .iter()
        .flat_map(|result| &result.models)
        .find(|model| model.path.to_string_lossy().contains("encodec_24khz"))
        .with_context(|| format!("Model path for {} not found in download results", ENCODEC))?;

    let weights = match weights_path {
        Some(path) => PathBuf::from(path),
        None => {
            let safetensors_path = dia_result.path.join("model.safetensors");
            if safetensors_path.exists() {
                safetensors_path
            } else {
                return Err(anyhow!(
                    "Expected model.safetensors at {:?}",
                    safetensors_path
                ));
            }
        }
    };

    let tokenizer = match tokenizer_path {
        Some(path) => PathBuf::from(path),
        None => {
            let tokenizer_path = dia_result.path.join("tokenizer.json");
            if tokenizer_path.exists() {
                tokenizer_path
            } else {
                return Err(anyhow!("Expected tokenizer.json at {:?}", tokenizer_path));
            }
        }
    };

    let encodec = encodec_result.path.clone();

    Ok(ModelPaths {
        weights,
        tokenizer,
        encodec,
    })
}
