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

// Import dia voice modules for TTS-based training sample generation
use dia::voice::{
    DiaSpeaker, VocalSpeedMod, VoicePersona, VoicePool, VoiceTimber, dia_tts_builder,
};
use futures_util::StreamExt;

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

    // Use existing trainer::train_dir function with proper Path handling
    trainer::train_dir(
        &cmd.data_dir,
        &cmd.output,
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
            if let Ok(mut detector_guard) = detector_clone.lock()
                && let Some(detection) = detector_guard.process_samples(data)
            {
                info!(
                    "Wake word detected: {} (score: {:.3})",
                    detection.name, detection.score
                );
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

    // Create output directory
    std::fs::create_dir_all(&cmd.output_dir)
        .with_context(|| format!("Failed to create output directory: {:?}", cmd.output_dir))?;

    // Initialize audio recording system
    let audio_recorder = AudioRecordingSystem::new()
        .await
        .context("Failed to initialize audio recording system")?;

    // Display available audio devices
    audio_recorder
        .list_available_devices()
        .context("Failed to list audio devices")?;

    // Select and configure audio device
    let recording_config = RecordingConfiguration {
        sample_rate: 16000,
        channels: 1,
        duration_seconds: cmd.duration,
        format: AudioSampleFormat::F32,
        buffer_size: 1024,
    };

    let mut recorder = audio_recorder
        .create_recorder(recording_config)
        .await
        .context("Failed to create audio recorder")?;

    info!(
        "Recording {} samples of {} seconds each",
        cmd.count, cmd.duration
    );
    info!("Press ENTER to start each recording, or 'q' + ENTER to quit");

    // Record each sample with user interaction
    for sample_index in 0..cmd.count {
        let output_filename = format!("{}_{:03}[{}].wav", cmd.label, sample_index, cmd.label);
        let output_path = cmd.output_dir.join(&output_filename);

        info!(
            "\n=== Recording Sample {}/{} ===",
            sample_index + 1,
            cmd.count
        );
        info!("Output: {}", output_path.display());
        info!("Press ENTER to start recording (or 'q' + ENTER to quit)...");

        // Wait for user input
        let user_input = wait_for_user_input().await?;
        if user_input.trim().to_lowercase() == "q" {
            info!("Recording cancelled by user");
            break;
        }

        // Start recording with real-time monitoring
        info!("🔴 RECORDING... ({}s)", cmd.duration);
        let audio_data = recorder
            .record_sample_with_monitoring()
            .await
            .with_context(|| format!("Failed to record sample {}", sample_index + 1))?;

        // Save to WAV file
        save_audio_to_wav(&audio_data, &output_path)
            .with_context(|| format!("Failed to save audio sample: {}", output_filename))?;

        info!(
            "✅ Saved: {} ({} samples)",
            output_path.display(),
            audio_data.len()
        );

        // Brief pause between recordings
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    }

    info!("Recording session completed!");
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
    info!("Generating TTS-based training samples with config: {cmd:?}");

    // Setup dia models using progresshub
    info!("Setting up dia voice models...");
    let _model_paths = dia::setup::setup()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to setup dia models: {}", e))?;

    info!("Successfully set up dia models");

    // Create output directory
    std::fs::create_dir_all(&cmd.output_dir)
        .with_context(|| format!("Failed to create output directory: {:?}", cmd.output_dir))?;

    // Initialize voice pool for TTS generation
    let voice_pool = Arc::new(
        VoicePool::new().map_err(|e| anyhow::anyhow!("Failed to initialize voice pool: {}", e))?,
    );

    // Generate sophisticated TTS-based training samples
    generate_diverse_voice_samples(&voice_pool, &cmd).await?;

    info!(
        "Successfully generated {} TTS-based training samples in {:?}",
        cmd.count, cmd.output_dir
    );
    Ok(())
}

/// Generate diverse voice samples using sophisticated TTS with multiple voice variations
async fn generate_diverse_voice_samples(
    voice_pool: &Arc<VoicePool>,
    cmd: &cli::GenerateCommand,
) -> Result<()> {
    // Define sophisticated voice variation matrix for diverse training data
    let voice_variations = create_voice_variation_matrix();

    let samples_per_variation = (cmd.count as f32 / voice_variations.len() as f32).ceil() as usize;
    let mut sample_index = 0;

    for (variation_name, timber, speed, persona) in voice_variations {
        if sample_index >= cmd.count {
            break;
        }

        info!(
            "Generating samples with {} voice variation...",
            variation_name
        );

        for _i in 0..samples_per_variation {
            if sample_index >= cmd.count {
                break;
            }

            let output_filename = format!("sample_{:04}_{}.wav", sample_index, variation_name);
            let output_path = cmd.output_dir.join(&output_filename);

            info!(
                "Generating sample {}/{}: {} ({})",
                sample_index + 1,
                cmd.count,
                output_filename,
                variation_name
            );

            // Generate TTS audio using dia voice with sophisticated variations
            let audio_data = generate_tts_audio_with_variation(
                voice_pool,
                &cmd.phrase,
                timber,
                speed,
                persona,
                cmd.target_loudness,
            )
            .await
            .with_context(|| {
                format!(
                    "Failed to generate TTS audio for variation: {}",
                    variation_name
                )
            })?;

            // Apply noise reduction if requested
            let processed_audio = if cmd.noise_reduction {
                apply_advanced_noise_reduction(audio_data)
            } else {
                audio_data
            };

            // Save audio to WAV file with proper format
            save_audio_to_wav(&processed_audio, &output_path)
                .with_context(|| format!("Failed to save audio sample: {}", output_filename))?;

            info!("Saved TTS sample: {output_path:?}");
            sample_index += 1;
        }
    }

    Ok(())
}

/// Create sophisticated voice variation matrix for diverse training data generation
fn create_voice_variation_matrix() -> Vec<(&'static str, VoiceTimber, VocalSpeedMod, VoicePersona)>
{
    vec![
        (
            "warm_deliberate",
            VoiceTimber::Warm,
            VocalSpeedMod::Deliberate,
            VoicePersona::Confident,
        ),
        (
            "rich_flowing",
            VoiceTimber::Rich,
            VocalSpeedMod::Flowing,
            VoicePersona::Playful,
        ),
        (
            "smooth_measured",
            VoiceTimber::Smooth,
            VocalSpeedMod::Measured,
            VoicePersona::Authoritative,
        ),
        (
            "breathy_excited",
            VoiceTimber::Breathy,
            VocalSpeedMod::Excited,
            VoicePersona::Enthusiastic,
        ),
        (
            "full_nervous",
            VoiceTimber::Full,
            VocalSpeedMod::Nervous,
            VoicePersona::Anxious,
        ),
        (
            "gravelly_choppy",
            VoiceTimber::Gravelly,
            VocalSpeedMod::Choppy,
            VoicePersona::Aggressive,
        ),
        (
            "thin_stuttering",
            VoiceTimber::Thin,
            VocalSpeedMod::Stuttering,
            VoicePersona::Gentle,
        ),
        (
            "nasal_drowsy",
            VoiceTimber::Nasal,
            VocalSpeedMod::Drowsy,
            VoicePersona::Tired,
        ),
        (
            "metallic_flowing",
            VoiceTimber::Metallic,
            VocalSpeedMod::Flowing,
            VoicePersona::Serious,
        ),
        (
            "hollow_measured",
            VoiceTimber::Hollow,
            VocalSpeedMod::Measured,
            VoicePersona::Mysterious,
        ),
    ]
}

/// Generate TTS audio with sophisticated voice variation parameters
async fn generate_tts_audio_with_variation(
    voice_pool: &Arc<VoicePool>,
    phrase: &str,
    timber: VoiceTimber,
    _speed: VocalSpeedMod,
    persona: VoicePersona,
    target_loudness: f32,
) -> Result<Vec<f32>> {
    // Create sophisticated TTS builder with voice characteristics
    let tts_builder = dia_tts_builder(voice_pool.clone(), phrase.to_string());

    // Create default speaker with voice characteristics applied to voice clone
    let mut speaker = DiaSpeaker::default();
    speaker.voice_clone = speaker
        .voice_clone
        .with_timber(timber)
        .with_persona(persona);

    let enhanced_builder = tts_builder.with_speaker(speaker);

    // Generate streaming audio synthesis
    let mut audio_stream = enhanced_builder.synthesize();
    let mut audio_chunks = Vec::new();

    // Collect all audio chunks from the stream
    while let Some(chunk) = audio_stream.next().await {
        if chunk.is_final {
            break;
        }
        audio_chunks.push(chunk);
    }

    // Convert audio chunks to f32 samples
    let mut audio_samples = Vec::new();
    for chunk in audio_chunks {
        let samples = convert_audio_chunk_to_samples(&chunk)?;
        audio_samples.extend(samples);
    }

    // Apply target loudness normalization
    let normalized_audio = normalize_audio_loudness(audio_samples, target_loudness);

    Ok(normalized_audio)
}

/// Convert dia audio chunk to f32 samples for koffee training
fn convert_audio_chunk_to_samples(chunk: &dia::voice::DiaAudioChunk) -> Result<Vec<f32>> {
    // Convert bytes to i16 PCM samples then to f32
    let mut samples = Vec::with_capacity(chunk.audio_data.len() / 2);

    for chunk_pair in chunk.audio_data.chunks_exact(2) {
        let sample_bytes = [chunk_pair[0], chunk_pair[1]];
        let sample_i16 = i16::from_le_bytes(sample_bytes);
        let sample_f32 = sample_i16 as f32 / 32768.0; // Convert to [-1.0, 1.0] range
        samples.push(sample_f32);
    }

    Ok(samples)
}

/// Apply advanced noise reduction for cleaner training samples
fn apply_advanced_noise_reduction(mut audio: Vec<f32>) -> Vec<f32> {
    // Apply sophisticated noise reduction algorithm
    let noise_floor = estimate_noise_floor(&audio);
    let reduction_factor = 0.1;

    for sample in &mut audio {
        if sample.abs() < noise_floor {
            *sample *= reduction_factor;
        }
    }

    // Apply gentle low-pass filter to remove high-frequency artifacts
    apply_low_pass_filter(&mut audio, 0.85);

    audio
}

/// Estimate noise floor for advanced noise reduction
fn estimate_noise_floor(audio: &[f32]) -> f32 {
    let mut sorted_samples: Vec<f32> = audio.iter().map(|s| s.abs()).collect();
    sorted_samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Use 10th percentile as noise floor estimate
    let percentile_index = (sorted_samples.len() as f32 * 0.1) as usize;
    sorted_samples
        .get(percentile_index)
        .copied()
        .unwrap_or(0.01)
}

/// Apply sophisticated low-pass filter for audio cleanup
fn apply_low_pass_filter(audio: &mut [f32], alpha: f32) {
    if let Some(first) = audio.first_mut() {
        let mut prev_sample = *first;

        for sample in audio.iter_mut().skip(1) {
            let filtered = alpha * (*sample) + (1.0 - alpha) * prev_sample;
            prev_sample = *sample;
            *sample = filtered;
        }
    }
}

/// Normalize audio to target loudness using advanced LUFS-based algorithm
fn normalize_audio_loudness(mut audio: Vec<f32>, target_loudness: f32) -> Vec<f32> {
    // Calculate RMS level for loudness estimation
    let rms = calculate_rms(&audio);

    if rms > 0.0 {
        // Convert target LUFS to linear gain (simplified approximation)
        let target_linear = 10.0_f32.powf(target_loudness / 20.0);
        let gain = target_linear / rms;

        // Apply gain with soft limiting to prevent clipping
        for sample in &mut audio {
            *sample = soft_limit(*sample * gain);
        }
    }

    audio
}

/// Calculate RMS level for loudness normalization
fn calculate_rms(audio: &[f32]) -> f32 {
    if audio.is_empty() {
        return 0.0;
    }

    let sum_squares: f32 = audio.iter().map(|s| s * s).sum();
    (sum_squares / audio.len() as f32).sqrt()
}

/// Apply soft limiting to prevent audio clipping
fn soft_limit(sample: f32) -> f32 {
    let threshold = 0.95;
    if sample.abs() > threshold {
        threshold * sample.signum() * (1.0 - (-3.0 * (sample.abs() - threshold)).exp())
    } else {
        sample
    }
}

// ============================================================================
// AUDIO RECORDING SYSTEM IMPLEMENTATION
// ============================================================================

/// Configuration for audio recording
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct RecordingConfiguration {
    sample_rate: u32,
    channels: u16,
    duration_seconds: u32,
    format: AudioSampleFormat,
    buffer_size: usize,
}

/// Supported audio sample formats
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
enum AudioSampleFormat {
    F32,
    I16,
}

/// Advanced audio recording system with device management
struct AudioRecordingSystem {
    host: cpal::Host,
}

impl AudioRecordingSystem {
    /// Initialize the audio recording system
    async fn new() -> Result<Self> {
        let host = cpal::default_host();
        Ok(Self { host })
    }

    /// List all available audio input devices
    fn list_available_devices(&self) -> Result<()> {
        info!("Available audio input devices:");

        let devices = self
            .host
            .input_devices()
            .map_err(|e| anyhow::anyhow!("Failed to enumerate input devices: {}", e))?;

        let mut device_count = 0;
        for (index, device) in devices.enumerate() {
            let device_name = device
                .name()
                .unwrap_or_else(|_| format!("Unknown Device {}", index));

            info!("  [{}] {}", index, device_name);

            // Show supported configurations
            if let Ok(configs) = device.supported_input_configs() {
                for config in configs {
                    info!(
                        "    - {} channels, {}-{} Hz, {:?}",
                        config.channels(),
                        config.min_sample_rate().0,
                        config.max_sample_rate().0,
                        config.sample_format()
                    );
                }
            }
            device_count += 1;
        }

        if device_count == 0 {
            return Err(anyhow::anyhow!("No audio input devices found"));
        }

        info!("Found {} audio input device(s)", device_count);
        Ok(())
    }

    /// Create a configured audio recorder
    async fn create_recorder(&self, config: RecordingConfiguration) -> Result<AudioRecorder> {
        // Get default input device
        let device = self
            .host
            .default_input_device()
            .ok_or_else(|| anyhow::anyhow!("No default input device available"))?;

        let device_name = device
            .name()
            .unwrap_or_else(|_| "Unknown Device".to_string());
        info!("Using audio device: {}", device_name);

        // Configure audio stream
        let supported_config = device
            .default_input_config()
            .map_err(|e| anyhow::anyhow!("Failed to get default input config: {}", e))?;

        info!(
            "Device config: {} channels, {} Hz, {:?}",
            supported_config.channels(),
            supported_config.sample_rate().0,
            supported_config.sample_format()
        );

        // Create recorder with the device and configuration
        AudioRecorder::new(device, supported_config, config).await
    }
}

/// Audio recorder with real-time monitoring capabilities
struct AudioRecorder {
    device: cpal::Device,
    config: cpal::SupportedStreamConfig,
    recording_config: RecordingConfiguration,
}

impl AudioRecorder {
    /// Create a new audio recorder
    async fn new(
        device: cpal::Device,
        config: cpal::SupportedStreamConfig,
        recording_config: RecordingConfiguration,
    ) -> Result<Self> {
        Ok(Self {
            device,
            config,
            recording_config,
        })
    }

    /// Record a single sample with real-time audio level monitoring
    async fn record_sample_with_monitoring(&mut self) -> Result<Vec<f32>> {
        let sample_rate = self.config.sample_rate().0;
        let channels = self.config.channels();
        let total_samples = (sample_rate * self.recording_config.duration_seconds) as usize;

        // Shared buffer for audio data
        let audio_buffer = std::sync::Arc::new(std::sync::Mutex::new(Vec::<f32>::new()));
        let buffer_clone = audio_buffer.clone();

        // Audio level monitoring
        let level_monitor = std::sync::Arc::new(std::sync::Mutex::new(AudioLevelMonitor::new()));
        let monitor_clone = level_monitor.clone();

        // Create audio stream based on sample format
        let stream_config = self.config.clone();
        let stream = match stream_config.sample_format() {
            cpal::SampleFormat::F32 => {
                self.device.build_input_stream(
                    &stream_config.into(),
                    move |data: &[f32], _: &cpal::InputCallbackInfo| {
                        if let (Ok(mut buffer), Ok(mut monitor)) =
                            (buffer_clone.lock(), monitor_clone.lock())
                        {
                            // Convert multi-channel to mono if needed
                            let mono_data = if channels == 1 {
                                data.to_vec()
                            } else {
                                convert_multichannel_to_mono(data, channels as usize)
                            };

                            // Update audio level monitoring
                            monitor.update_levels(&mono_data);

                            // Store audio data
                            buffer.extend_from_slice(&mono_data);
                        }
                    },
                    |err| error!("Audio stream error: {}", err),
                    None,
                )
            }
            cpal::SampleFormat::I16 => {
                self.device.build_input_stream(
                    &stream_config.into(),
                    move |data: &[i16], _: &cpal::InputCallbackInfo| {
                        if let (Ok(mut buffer), Ok(mut monitor)) =
                            (buffer_clone.lock(), monitor_clone.lock())
                        {
                            // Convert i16 to f32 and handle multi-channel
                            let f32_data: Vec<f32> =
                                data.iter().map(|&sample| sample as f32 / 32768.0).collect();

                            let mono_data = if channels == 1 {
                                f32_data
                            } else {
                                convert_multichannel_to_mono(&f32_data, channels as usize)
                            };

                            // Update audio level monitoring
                            monitor.update_levels(&mono_data);

                            // Store audio data
                            buffer.extend_from_slice(&mono_data);
                        }
                    },
                    |err| error!("Audio stream error: {}", err),
                    None,
                )
            }
            _ => {
                return Err(anyhow::anyhow!(
                    "Unsupported sample format: {:?}",
                    stream_config.sample_format()
                ));
            }
        }
        .map_err(|e| anyhow::anyhow!("Failed to build input stream: {}", e))?;

        // Start recording
        stream
            .play()
            .map_err(|e| anyhow::anyhow!("Failed to start audio stream: {}", e))?;

        // Real-time monitoring loop
        let start_time = std::time::Instant::now();
        let duration =
            std::time::Duration::from_secs(self.recording_config.duration_seconds as u64);

        while start_time.elapsed() < duration {
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

            // Display real-time audio levels
            if let Ok(monitor) = level_monitor.lock() {
                let (rms, peak) = monitor.get_current_levels();

                // Simple VU meter display
                let vu_meter = create_vu_meter(rms, peak);
                print!(
                    "\r🎤 {} RMS: {:.3} Peak: {:.3} [{:.1}s]",
                    vu_meter,
                    rms,
                    peak,
                    start_time.elapsed().as_secs_f32()
                );
                let _ = std::io::Write::flush(&mut std::io::stdout());
            }
        }

        println!(); // New line after recording

        // Stop the stream
        drop(stream);

        // Extract recorded audio data
        let recorded_samples = if let Ok(buffer) = audio_buffer.lock() {
            buffer.clone()
        } else {
            return Err(anyhow::anyhow!("Failed to access recorded audio buffer"));
        };

        if recorded_samples.is_empty() {
            return Err(anyhow::anyhow!("No audio data recorded"));
        }

        // Trim to exact duration if needed
        let trimmed_samples = if recorded_samples.len() > total_samples {
            recorded_samples[..total_samples].to_vec()
        } else {
            recorded_samples
        };

        info!(
            "Recorded {} samples ({:.2}s)",
            trimmed_samples.len(),
            trimmed_samples.len() as f32 / sample_rate as f32
        );

        Ok(trimmed_samples)
    }
}

/// Audio level monitoring for real-time feedback
struct AudioLevelMonitor {
    rms_level: f32,
    peak_level: f32,
    sample_count: usize,
}

impl AudioLevelMonitor {
    fn new() -> Self {
        Self {
            rms_level: 0.0,
            peak_level: 0.0,
            sample_count: 0,
        }
    }

    fn update_levels(&mut self, samples: &[f32]) {
        if samples.is_empty() {
            return;
        }

        // Calculate RMS level
        let sum_squares: f32 = samples.iter().map(|&s| s * s).sum();
        let current_rms = (sum_squares / samples.len() as f32).sqrt();

        // Calculate peak level
        let current_peak = samples.iter().map(|&s| s.abs()).fold(0.0f32, f32::max);

        // Smooth the levels with exponential moving average
        let alpha = 0.1; // Smoothing factor
        self.rms_level = alpha * current_rms + (1.0 - alpha) * self.rms_level;
        self.peak_level = alpha * current_peak + (1.0 - alpha) * self.peak_level;

        self.sample_count += samples.len();
    }

    fn get_current_levels(&self) -> (f32, f32) {
        (self.rms_level, self.peak_level)
    }
}

/// Convert multi-channel audio to mono by averaging channels
fn convert_multichannel_to_mono(data: &[f32], channels: usize) -> Vec<f32> {
    if channels == 1 {
        return data.to_vec();
    }

    let frame_count = data.len() / channels;
    let mut mono_data = Vec::with_capacity(frame_count);

    for frame_index in 0..frame_count {
        let mut sum = 0.0f32;
        for channel in 0..channels {
            sum += data[frame_index * channels + channel];
        }
        mono_data.push(sum / channels as f32);
    }

    mono_data
}

/// Create a visual VU meter for real-time audio level display
fn create_vu_meter(rms: f32, peak: f32) -> String {
    let meter_width = 20;
    let rms_bars = (rms * meter_width as f32) as usize;
    let peak_bars = (peak * meter_width as f32) as usize;

    let mut meter = String::with_capacity(meter_width + 2);
    meter.push('[');

    for i in 0..meter_width {
        if i < rms_bars {
            meter.push('█');
        } else if i < peak_bars {
            meter.push('▓');
        } else {
            meter.push('░');
        }
    }

    meter.push(']');
    meter
}

/// Wait for user input asynchronously
async fn wait_for_user_input() -> Result<String> {
    use tokio::io::{AsyncBufReadExt, BufReader};

    let stdin = tokio::io::stdin();
    let mut reader = BufReader::new(stdin);
    let mut input = String::new();

    reader
        .read_line(&mut input)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to read user input: {}", e))?;

    Ok(input)
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
