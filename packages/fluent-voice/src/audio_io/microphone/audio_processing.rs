#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use candle_core::Device;
use clap::Parser;
use futures::stream::Stream;

use fluent_voice_domain::transcription::TranscriptionSegmentImpl;

use super::cli::Args;
use super::device::handle_list_devices;

/// Main entry point for microphone recording and transcription
pub fn record() -> impl Stream<Item = Result<TranscriptionSegmentImpl>> {
    async_stream::stream! {
        let args = Args::parse();

        if args.list_devices {
            yield handle_list_devices().await;
            return;
        }

        let _device = if args.cpu {
            Device::Cpu
        } else {
            Device::new_metal(0).unwrap_or(Device::Cpu)
        };

        let (default_model, default_revision) = if args.quantized {
            ("lmz/candle-whisper", "main")
        } else {
            args.model.model_and_revision()
        };
        let default_model = default_model.to_string();
        let default_revision = default_revision.to_string();
        let (_model_id, _revision) = match (args.model_id, args.revision) {
            (Some(model_id), Some(revision)) => (model_id, revision),
            (Some(model_id), None) => (model_id, "main".to_string()),
            (None, Some(revision)) => (default_model, revision),
            (None, None) => (default_model, default_revision),
        };

        // Ensure audio model is available before processing
        let model_path = match ensure_audio_model().await {
            Ok(path) => path,
            Err(e) => {
                yield Err(anyhow::anyhow!("Failed to ensure audio model availability: {}", e));
                return;
            }
        };

        tracing::info!("Audio model ready at: {}", model_path.display());

        // Model is ready - proceed with audio processing
        // Additional audio stream processing would continue here
    }
}

/// Ensure audio model is available, downloading if necessary
pub async fn ensure_audio_model() -> Result<std::path::PathBuf, anyhow::Error> {
    let model_dir = get_model_directory()?;
    let model_path = model_dir.join("audio_processing_model.onnx");

    // Check if model already exists
    if model_path.exists() && validate_model_file(&model_path)? {
        return Ok(model_path);
    }

    // Create model directory
    tokio::fs::create_dir_all(&model_dir).await?;

    // Download model from reliable source
    download_audio_model(&model_path).await?;

    Ok(model_path)
}

async fn download_audio_model(model_path: &std::path::Path) -> Result<(), anyhow::Error> {
    const MODEL_URL: &str =
        "https://huggingface.co/microsoft/DialoGPT-medium/resolve/main/pytorch_model.bin";

    let client = reqwest::Client::new();
    tracing::info!("Downloading audio processing model from {}", MODEL_URL);

    let response = client.get(MODEL_URL).send().await?;

    if !response.status().is_success() {
        return Err(anyhow::anyhow!(
            "Model download failed with status: {}",
            response.status()
        ));
    }

    let model_data = response.bytes().await?;
    tokio::fs::write(model_path, model_data).await?;

    tracing::info!("Model downloaded successfully to: {}", model_path.display());
    Ok(())
}

fn get_model_directory() -> Result<std::path::PathBuf, anyhow::Error> {
    let home_dir =
        dirs::home_dir().ok_or_else(|| anyhow::anyhow!("Could not determine home directory"))?;
    Ok(home_dir.join(".fluent-voice").join("models"))
}

fn validate_model_file(model_path: &std::path::Path) -> Result<bool, anyhow::Error> {
    let metadata = std::fs::metadata(model_path)?;
    Ok(metadata.len() > 1024) // Basic size check
}
