//! Test Cyrup Wake Word Models
//!
//! This example demonstrates using both "cyrup" (wake) and "cyrup stop" (unwake)
//! models for accurate voice command detection.
//!
//! Usage:
//!   cargo run --example test_cyrup_models

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use koffee::{
    Kfc, KoffeeCandleDetection, ScoreMode,
    config::{AudioFmt, DetectorConfig, FiltersConfig, KoffeeCandleConfig, VADMode},
    wakewords::{WakewordLoad, WakewordModel},
};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Detection event with metadata
#[derive(Debug, Clone)]
struct DetectionEvent {
    model_name: String,
    score: f32,
    timestamp: Instant,
}

/// Application state with detection history
struct AppState {
    awake: Arc<AtomicBool>,
    detection_history: Arc<Mutex<VecDeque<DetectionEvent>>>,
    wake_detector: Arc<Mutex<Kfc>>,
    stop_detector: Arc<Mutex<Option<Kfc>>>, // Optional, might not exist
}

impl AppState {
    fn new(wake_detector: Kfc, stop_detector: Option<Kfc>) -> Self {
        Self {
            awake: Arc::new(AtomicBool::new(false)),
            detection_history: Arc::new(Mutex::new(VecDeque::with_capacity(10))),
            wake_detector: Arc::new(Mutex::new(wake_detector)),
            stop_detector: Arc::new(Mutex::new(stop_detector)),
        }
    }

    fn add_detection(&self, model_name: String, score: f32) {
        let event = DetectionEvent {
            model_name,
            score,
            timestamp: Instant::now(),
        };

        if let Ok(mut history) = self.detection_history.lock() {
            history.push_back(event);
            if history.len() > 10 {
                history.pop_front();
            }
        }
    }

    fn get_recent_detections(&self, duration: Duration) -> Vec<DetectionEvent> {
        let now = Instant::now();
        if let Ok(history) = self.detection_history.lock() {
            history
                .iter()
                .filter(|e| now.duration_since(e.timestamp) < duration)
                .cloned()
                .collect()
        } else {
            Vec::new()
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎤 Cyrup Wake Word Detection System");
    println!("==================================");
    println!();

    // Load wake word model from trained models
    let wake_model = match WakewordModel::load_from_file("syrup.rpw") {
        Ok(m) => {
            println!("✅ Loaded wake model: syrup.rpw");
            m
        }
        Err(e) => {
            eprintln!("❌ Failed to load wake model: {}", e);
            return Err("Wake model required".into());
        }
    };

    // Try to load a different model for stop detection
    let stop_model = match WakewordModel::load_from_file("cyrup_stop.rpw") {
        Ok(m) => {
            println!("✅ Loaded stop model: cyrup_stop.rpw");
            Some(m)
        }
        Err(e) => {
            println!(
                "⚠️  No stop model found - will use double-tap detection: {}",
                e
            );
            None
        }
    };

    // Configure detectors with optimized settings
    let config = KoffeeCandleConfig {
        detector: DetectorConfig {
            avg_threshold: 0.05, // Much lower threshold for testing
            threshold: 0.25,     // Much lower threshold for testing
            min_scores: 1,       // Only require 1 score for testing
            score_mode: ScoreMode::Max,
            score_ref: 0.22,
            vad_mode: Some(VADMode::Easy), // Enable VAD for better accuracy
            ..Default::default()
        },
        filters: FiltersConfig {
            band_pass: koffee::config::BandPassConfig {
                enabled: true,
                low_cutoff: 85.0, // Human voice frequency range
                high_cutoff: 255.0,
            },
            gain_normalizer: koffee::config::GainNormalizationConfig {
                enabled: true,
                gain_ref: Some(0.095),
                ..Default::default()
            },
        },
        fmt: AudioFmt {
            sample_rate: 16000,
            channels: 1,
            sample_format: koffee::SampleFormat::I16,
            endianness: koffee::Endianness::Little,
        },
        wake_unwake: koffee::WakeUnwakeConfig::default(), // Enable wake/unwake functionality
    };

    // Create wake detector
    let mut wake_detector = Kfc::new(&config)?;
    wake_detector.add_wakeword_model(wake_model)?;

    // Create stop detector if model exists
    let stop_detector = if let Some(model) = stop_model {
        let mut detector = Kfc::new(&config)?;
        detector.add_wakeword_model(model)?;
        Some(detector)
    } else {
        None
    };

    println!("✅ Detectors initialized");
    println!();

    // Audio setup
    let host = cpal::default_host();

    // Get the default input device
    let device = host
        .default_input_device()
        .ok_or("No input device available")?;

    println!("Audio device: {}", device.name()?);

    // Get all supported input configurations
    let mut supported_configs = device
        .supported_input_configs()
        .map_err(|e| format!("Failed to get supported input configs: {}", e))?
        .collect::<Vec<_>>();

    if supported_configs.is_empty() {
        return Err("No supported input configurations found".into());
    }

    // Sort by sample rate (prefer higher rates)
    supported_configs.sort_by(|a, b| b.max_sample_rate().0.cmp(&a.max_sample_rate().0));

    // Try to find a configuration with 1 channel (mono)
    let mono_config = supported_configs.iter().find(|c| c.channels() == 1);

    // Use the first available config (prefer mono if available)
    let config = match mono_config {
        Some(c) => c.clone(),
        None => supported_configs[0].clone(),
    };

    // Create a stream config with the highest supported sample rate
    let stream_config = config.with_sample_rate(config.max_sample_rate()).config();

    println!("Selected audio config: {:?}", stream_config);
    println!("Sample format: {:?}", config.sample_format());
    println!("Channels: {}", config.channels());
    println!("Sample rate: {} Hz", stream_config.sample_rate.0);

    // Print all supported configurations for debugging
    println!("\nAll supported input configurations:");
    for (i, cfg) in supported_configs.iter().enumerate() {
        println!(
            "  {}. {:?} (channels: {}, sample rate: {} - {} Hz, format: {:?})",
            i + 1,
            cfg,
            cfg.channels(),
            cfg.min_sample_rate().0,
            cfg.max_sample_rate().0,
            cfg.sample_format()
        );
    }

    // Create application state
    let state = AppState::new(wake_detector, stop_detector);

    // Status display
    println!();
    println!("📢 Commands:");
    println!("   • Say 'cyrup' to wake the system");
    if state.stop_detector.lock().unwrap().is_some() {
        println!("   • Say 'cyrup stop' to put it to sleep");
    } else {
        println!("   • Say 'cyrup' again within 2 seconds to sleep");
    }
    println!();
    println!("🔴 SLEEPING - Say 'cyrup' to wake");
    println!();

    // Audio processing
    let awake_clone = Arc::clone(&state.awake);
    let wake_detector_clone = Arc::clone(&state.wake_detector);
    let stop_detector_clone = Arc::clone(&state.stop_detector);
    let state_clone = state.clone();

    let mut audio_buffer: Vec<u8> = Vec::new();
    let chunk_size = (30.0 * stream_config.sample_rate.0 as f32 / 1000.0 * 2.0) as usize; // 30ms of audio in bytes (2 bytes per sample)

    // Create a callback that processes audio samples
    let data_callback = move |data: &[f32], _: &cpal::InputCallbackInfo| {
        // Convert the incoming samples to i16 format
        for &sample in data {
            // Convert from f32 to i16 (f32 is in range [-1.0, 1.0])
            let sample_i16 = (sample * i16::MAX as f32)
                .max(i16::MIN as f32)
                .min(i16::MAX as f32) as i16;
            audio_buffer.extend_from_slice(&sample_i16.to_le_bytes());
        }

        // Process complete chunks of audio data (chunk_size is in bytes)
        while audio_buffer.len() >= chunk_size {
            let chunk = &audio_buffer[..chunk_size];

            // Process the audio chunk
            let is_awake = awake_clone.load(Ordering::Relaxed);

            // Always check for wake word
            if let Ok(mut detector) = wake_detector_clone.try_lock() {
                if let Some(detection) = detector.process_bytes(chunk) {
                    state_clone.add_detection("wake".to_string(), detection.score);
                    handle_wake_detection(detection, &state_clone);
                }
            }

            // Check for stop word if awake and detector exists
            if is_awake {
                if let Ok(mut stop_opt) = stop_detector_clone.try_lock() {
                    if let Some(stop_det) = stop_opt.as_mut() {
                        if let Some(detection) = stop_det.process_bytes(chunk) {
                            state_clone.add_detection("stop".to_string(), detection.score);
                            handle_stop_detection(detection, &state_clone);
                        }
                    }
                }
            }

            audio_buffer.drain(..chunk_size);
        }
    };

    // Build and start the input stream with better error handling
    println!(
        "\nCreating audio input stream with config: {:?}",
        stream_config
    );

    // Create the error callback
    let error_callback = |err| {
        eprintln!("An error occurred on the audio stream: {}", err);
    };

    // Try to build the input stream with the supported config
    println!(
        "Attempting to build input stream with format: {:?}",
        config.sample_format()
    );

    let stream =
        match device.build_input_stream(&stream_config, data_callback, error_callback, None) {
            Ok(stream) => {
                println!("Successfully created audio input stream");
                stream
            }
            Err(e) => {
                eprintln!("Failed to create audio input stream: {}", e);
                eprintln!("Available input formats:");
                if let Ok(formats) = device.supported_input_configs() {
                    for format in formats {
                        eprintln!("  {:?}", format);
                    }
                } else {
                    eprintln!("  Could not retrieve supported input formats");
                }
                return Err(Box::new(e) as Box<dyn std::error::Error>);
            }
        };

    println!("Audio stream created successfully");

    stream.play()?;

    // Main loop with status updates
    let mut last_state = false;

    loop {
        std::thread::sleep(Duration::from_millis(100));

        let current_state = state.awake.load(Ordering::Relaxed);
        if current_state != last_state {
            last_state = current_state;
            println!();
            if current_state {
                println!("🟢 AWAKE - Listening for commands");
            } else {
                println!("🔴 SLEEPING - Say 'cyrup' to wake");
            }

            // Show recent detection history
            let recent = state.get_recent_detections(Duration::from_secs(5));
            if !recent.is_empty() {
                println!("   Recent detections:");
                for event in recent.iter().rev().take(3) {
                    println!("   - {} (score: {:.2})", event.model_name, event.score);
                }
            }
        }
    }
}

fn handle_wake_detection(detection: KoffeeCandleDetection, state: &AppState) {
    if detection.score < 0.5 {
        return; // Too low confidence
    }

    let is_awake = state.awake.load(Ordering::Relaxed);

    if !is_awake {
        // Wake up
        println!("🔊 Wake word detected! (score: {:.2})", detection.score);
        state.awake.store(true, Ordering::Relaxed);
    } else if state.stop_detector.lock().unwrap().is_none() {
        // No stop detector - use double-tap to sleep
        let recent = state.get_recent_detections(Duration::from_secs(2));
        let wake_count = recent
            .iter()
            .filter(|e| e.model_name == "wake" && e.score > 0.5)
            .count();

        if wake_count >= 2 {
            println!("🔊 Double wake word - going to sleep");
            state.awake.store(false, Ordering::Relaxed);
        }
    }
}

fn handle_stop_detection(detection: KoffeeCandleDetection, state: &AppState) {
    if detection.score > 0.5 && state.awake.load(Ordering::Relaxed) {
        println!("🔊 Stop command detected! (score: {:.2})", detection.score);
        state.awake.store(false, Ordering::Relaxed);
    }
}

// Implement Clone for AppState to use in closures
impl Clone for AppState {
    fn clone(&self) -> Self {
        // We only clone the Arc references, not the inner data
        Self {
            awake: Arc::clone(&self.awake),
            detection_history: Arc::clone(&self.detection_history),
            wake_detector: Arc::clone(&self.wake_detector),
            stop_detector: Arc::clone(&self.stop_detector),
        }
    }
}
