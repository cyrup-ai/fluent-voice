//! Koffee CLI Binary
//! Cross-platform wake-word detector using Candle ML

use anyhow::{Context, Result};
use clap::Parser;
use env_logger::Env;
use log::{error, info};

mod cli;
use cli::{Cli, Commands};

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
    // TODO: Implement model training
    error!("Model training not yet implemented");
    Ok(())
}

async fn detect_wake_words(cmd: cli::DetectCommand) -> Result<()> {
    info!("Detecting wake words with config: {cmd:?}");
    // TODO: Implement wake word detection
    error!("Wake word detection not yet implemented");
    Ok(())
}

async fn list_audio_devices() -> Result<()> {
    info!("Listing available audio devices...");
    // TODO: Implement device listing
    error!("Audio device listing not yet implemented");
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
    // TODO: Implement model inspection
    error!("Model inspection not yet implemented");
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
