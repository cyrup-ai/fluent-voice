use anyhow::{Context, Result, anyhow};
// use progresshub::ModelDownloader;
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
    // TODO: Implement model downloading without progresshub
    Err(anyhow!("Model downloading not yet implemented without progresshub"))
}
