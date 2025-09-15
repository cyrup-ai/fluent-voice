//! Koffee CLI Binary
//! Cross-platform wake-word detector using Candle ML

use anyhow::{Context, Result};
use clap::Parser;
use env_logger::Env;
use log::{error, info};
use std::sync::Arc;
use std::sync::Mutex;

mod cli;
use cli::{Cli, Commands, ModelType};

// Import koffee modules for wake word functionality
use koffee::Kfc;
use koffee::trainer;
use koffee::wakewords::{WakewordLoad, WakewordModel};

// Import cpal for audio device management
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

// Error handling
#[derive(Debug, thiserror::Error)]
pub enum WakeWordCliError {
    #[error("Training failed: {0}")]
    Training(String),
    #[error("Model loading failed: {0}")]
    ModelLoad(String),
    #[error("Audio device error: {0}")]
    Audio(#[from] cpal::BuildStreamError),
    #[error("Feature extraction failed: {0}")]
    FeatureExtraction(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Train(train_cmd) => {
            info!("Training wake word model...");
            train_model(train_cmd).await
        }
        Commands::Detect(detect_cmd) => {
            info!("Starting wake word detection...");
            detect_wake_words(detect_cmd).await
        }
        Commands::ListDevices => {
            info!("Listing audio devices...");
            list_audio_devices().await
        }
        Commands::Record(record_cmd) => {
            info!("Recording training samples...");
            record_samples(record_cmd).await
        }
        Commands::Inspect(inspect_cmd) => {
            info!("Inspecting model...");
            inspect_model(inspect_cmd).await
        }
        Commands::Generate(generate_cmd) => {
            info!("Generating synthetic training samples...");
            generate_samples(generate_cmd).await
        }
    }
}

async fn train_model(cmd: cli::TrainCommand) -> Result<()> {
    info!("Training model with config: {cmd:?}");

    // Convert CLI ModelType to u8 for trainer
    let model_type = match cmd.model_type {
        ModelType::Tiny => 0,
        ModelType::Small => 1,
        ModelType::Medium => 2,
        ModelType::Large => 3,
    };

    // Use existing trainer::train_dir function
    trainer::train_dir(
        cmd.data_dir.to_str().unwrap(),
        cmd.output.to_str().unwrap(),
        koffee::ModelType::from(model_type),
    )
    .map_err(|e| anyhow::anyhow!("Training failed: {}", e))?;

    info!("Model trained and saved to: {:?}", cmd.output);
    Ok(())
}

async fn detect_wake_words(cmd: cli::DetectCommand) -> Result<()> {
    info!("Detecting wake words with config: {cmd:?}");

    // Create Kfc detector with default config
    let kfc_config = koffee::config::KoffeeCandleConfig::default();
    let detector =
        Kfc::new(&kfc_config).map_err(|e| anyhow::anyhow!("Failed to create detector: {}", e))?;

    // Get audio host and device
    let host = cpal::default_host();
    let device = if let Some(device_name) = &cmd.device {
        // Find device by name
        host.input_devices()
            .map_err(|e| anyhow::anyhow!("Failed to enumerate devices: {}", e))?
            .find(|d| d.name().unwrap_or_default() == *device_name)
            .ok_or_else(|| anyhow::anyhow!("Device '{}' not found", device_name))?
    } else {
        // Use default input device
        host.default_input_device()
            .ok_or_else(|| anyhow::anyhow!("No default input device available"))?
    };

    // Get supported config
    let config = device
        .default_input_config()
        .map_err(|e| anyhow::anyhow!("Failed to get device config: {}", e))?;

    info!(
        "Using audio device: {}",
        device.name().unwrap_or("Unknown".to_string())
    );
    info!("Audio config: {:?}", config);

    // Build input stream for real-time detection
    let detector_arc = Arc::new(Mutex::new(detector));
    let detector_clone = detector_arc.clone();
    let stream = device.build_input_stream(
        &config.into(),
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            if let Ok(mut detector_guard) = detector_clone.lock() {
                if let Some(detection) = detector_guard.process_samples(data) {
                    info!(
                        "Wake word detected: {} (score: {:.3})",
                        detection.name, detection.score
                    );
                }
            }
        },
        |err| error!("Audio stream error: {}", err),
        None,
    )?;

    // Start the stream
    stream.play()?;
    info!("Listening for wake words... Press Ctrl+C to stop.");

    // Keep running until interrupted
    tokio::signal::ctrl_c().await?;
    info!("Stopping wake word detection.");

    Ok(())
}

async fn list_audio_devices() -> Result<()> {
    info!("Listing available audio devices...");

    // Use existing examples/list_audio_devices.rs logic
    let host = cpal::default_host();

    println!("Available input devices:");
    for device in host.input_devices()? {
        let name = device.name()?;
        println!("  - {}", name);

        // Show supported configurations
        if let Ok(config) = device.default_input_config() {
            println!("    Default: {:?}", config);
        }

        // Show supported input configs
        match device.supported_input_configs() {
            Ok(configs) => {
                for (i, config) in configs.enumerate() {
                    if i < 3 {
                        // Limit to first 3 configs to avoid spam
                        println!("    Config {}: {:?}", i + 1, config);
                    }
                }
            }
            Err(_) => println!("    No supported configs available"),
        }
    }

    Ok(())
}

async fn record_samples(cmd: cli::RecordCommand) -> Result<()> {
    info!("Recording samples with config: {cmd:?}");
    // TODO: Implement sample recording
    error!("Sample recording not yet implemented");
    Ok(())
}

async fn inspect_model(cmd: cli::InspectCommand) -> Result<()> {
    info!("Inspecting model: {:?}", cmd.model_path);

    // Load model using WakewordLoad trait
    let model = WakewordModel::load_from_file(&cmd.model_path)
        .map_err(|e| anyhow::anyhow!("Failed to load model: {}", e))?;

    // Display model metadata
    println!("Model Information:");
    println!("  Labels: {:?}", model.labels);

    // Display tensor information
    println!("  Weights:");
    match &model.weights {
        koffee::wakewords::ModelWeights::Map(tensors) => {
            for (name, tensor_data) in tensors {
                println!(
                    "    {}: dims {:?}, dtype {}, {} bytes",
                    name,
                    tensor_data.dims,
                    tensor_data.d_type,
                    tensor_data.bytes.len()
                );
            }
        }
        koffee::wakewords::ModelWeights::Raw(bytes) => {
            println!("    Raw weights: {} bytes", bytes.len());
        }
    }

    Ok(())
}

async fn generate_samples(cmd: cli::GenerateCommand) -> Result<()> {
    info!("Generating samples with config: {cmd:?}");

    // Setup dia models using progresshub
    info!("Setting up dia models...");
    let _model_paths = dia::setup::setup()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to setup dia models: {}", e))?;

    info!("Successfully set up dia models");

    // Create output directory
    std::fs::create_dir_all(&cmd.output_dir)
        .with_context(|| format!("Failed to create output directory: {:?}", cmd.output_dir))?;

    // Generate samples using simple placeholder approach for now
    for i in 0..cmd.count {
        let output_filename = format!("sample_{i:04}.wav");
        let output_path = cmd.output_dir.join(&output_filename);

        info!(
            "Generating sample {}/{}: {}",
            i + 1,
            cmd.count,
            output_filename
        );

        // Generate simple sine wave as placeholder (dia TTS integration to be completed)
        let sample_rate = 16000;
        let duration_seconds = 2.0;
        let frequency = 440.0; // A4 note
        let num_samples = (sample_rate as f32 * duration_seconds) as usize;

        let audio_data: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                let amplitude = 0.1; // Quiet volume
                amplitude * (2.0 * std::f32::consts::PI * frequency * t).sin()
            })
            .collect();

        // Save audio to WAV file
        save_audio_to_wav(&audio_data, &output_path).context("Failed to save audio to WAV file")?;

        info!("Saved sample to: {output_path:?}");
    }

    info!(
        "Successfully generated {} samples in {:?}",
        cmd.count, cmd.output_dir
    );
    Ok(())
}

fn save_audio_to_wav(audio_data: &[f32], output_path: &std::path::Path) -> Result<()> {
    use dia::audio::SAMPLE_RATE;
    use dia::audio_io::write_pcm_as_wav;
    use std::fs::File;

    // Convert f32 to i16 PCM
    let pcm_samples: Vec<i16> = audio_data
        .iter()
        .map(|&x| (x.clamp(-1.0, 1.0) * 32767.0) as i16)
        .collect();

    // Write WAV file
    let mut file = File::create(output_path)
        .with_context(|| format!("Failed to create file: {output_path:?}"))?;

    write_pcm_as_wav(&mut file, &pcm_samples, SAMPLE_RATE as u32)
        .context("Failed to write PCM data to WAV file")?;

    Ok(())
}
