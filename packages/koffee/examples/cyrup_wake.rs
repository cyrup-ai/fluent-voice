//! Cyrup Wake Word Detection Example
//!
//! This example demonstrates using the existing "syrup" wake word model
//! for detecting "cyrup" as a wake word and "cyrup stop" as an unwake command.
//!
//! Usage:
//!   cargo run --example cyrup_wake

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleRate, StreamConfig};
use koffee::{
    Kfc, KoffeeCandleDetection,
    config::{AudioFmt, DetectorConfig, FiltersConfig, KoffeeCandleConfig},
    wakewords::{WakewordLoad, WakewordModel},
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Application state
struct AppState {
    awake: Arc<AtomicBool>,
    last_detection: Arc<Mutex<Option<Instant>>>,
}

impl AppState {
    fn new() -> Self {
        Self {
            awake: Arc::new(AtomicBool::new(false)),
            last_detection: Arc::new(Mutex::new(None)),
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎤 Cyrup Wake Word Detection");
    println!("============================");
    println!();
    println!("📢 Say 'cyrup' to wake, 'cyrup stop' to sleep");
    println!();

    // Load the pre-trained syrup wake word model
    let model_path = concat!(env!("CARGO_MANIFEST_DIR"), "/syrup.rpw");
    let model = match WakewordModel::load_from_file(model_path) {
        Ok(m) => {
            println!("✅ Loaded wake word model: {}", model_path);
            m
        }
        Err(e) => {
            eprintln!("❌ Failed to load model {}: {}", model_path, e);
            eprintln!("Make sure you're running from the koffee directory");
            return Err(Box::new(e));
        }
    };

    // Configure the detector
    let config = KoffeeCandleConfig {
        detector: DetectorConfig {
            avg_threshold: 0.2,
            threshold: 0.5,
            min_scores: 3,
            score_mode: koffee::ScoreMode::Max,
            ..Default::default()
        },
        filters: FiltersConfig {
            band_pass: koffee::config::BandPassConfig {
                enabled: true,
                low_cutoff: 80.0,
                high_cutoff: 400.0,
            },
            gain_normalizer: koffee::config::GainNormalizationConfig {
                enabled: true,
                gain_ref: Some(0.1),
                min_gain: 0.1,
                max_gain: 1.0,
            },
        },
        fmt: AudioFmt {
            sample_rate: 16000,
            channels: 1,
            sample_format: koffee::SampleFormat::I16,
            endianness: koffee::Endianness::Little,
        },
        wake_unwake: Default::default(),
    };

    // Create detector
    let mut detector = Kfc::new(&config)?;
    detector.add_wakeword_model(model)?;

    println!("✅ Wake word detector initialized");
    println!();

    // Set up audio capture
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or("No default input device available")?;

    println!("🎙️  Using audio device: {}", device.name()?);

    let supported_config = device.default_input_config()?;
    println!("   Format: {:?}", supported_config);

    let stream_config = StreamConfig {
        channels: 1,
        sample_rate: SampleRate(16000),
        buffer_size: cpal::BufferSize::Default,
    };

    // Application state
    let state = AppState::new();
    let detector = Arc::new(Mutex::new(detector));

    // Audio processing callback
    let detector_clone = Arc::clone(&detector);
    let awake_clone = Arc::clone(&state.awake);
    let last_detection_clone = Arc::clone(&state.last_detection);

    let mut audio_buffer = Vec::new();
    let chunk_size = 480; // 30ms at 16kHz

    let data_callback = move |data: &[f32], _: &cpal::InputCallbackInfo| {
        // Convert f32 to i16 samples
        for &sample in data {
            let sample_i16 = (sample * i16::MAX as f32) as i16;
            audio_buffer.extend_from_slice(&sample_i16.to_le_bytes());
        }

        // Process chunks
        while audio_buffer.len() >= chunk_size * 2 {
            let chunk: Vec<u8> = audio_buffer.drain(..chunk_size * 2).collect();

            // Process with detector
            if let Ok(mut det) = detector_clone.try_lock() {
                if let Some(detection) = det.process_bytes(&chunk) {
                    handle_detection(detection, &awake_clone, &last_detection_clone);
                }
            }
        }
    };

    let error_callback = |err| {
        eprintln!("Audio stream error: {}", err);
    };

    let stream = device.build_input_stream(&stream_config, data_callback, error_callback, None)?;

    stream.play()?;
    println!("🎧 Listening for wake words...");
    println!();

    // Main loop - check state and provide feedback
    loop {
        std::thread::sleep(Duration::from_millis(100));

        // Check if state changed
        static mut LAST_STATE: bool = false;
        let current_state = state.awake.load(Ordering::Relaxed);

        unsafe {
            if current_state != LAST_STATE {
                LAST_STATE = current_state;
                if current_state {
                    println!("🟢 AWAKE - System is listening");
                } else {
                    println!("🔴 SLEEPING - Say 'cyrup' to wake");
                }
            }
        }
    }
}

fn handle_detection(
    detection: KoffeeCandleDetection,
    awake: &Arc<AtomicBool>,
    last_detection: &Arc<Mutex<Option<Instant>>>,
) {
    let now = Instant::now();

    // Debounce detections (ignore if too close to last one)
    if let Ok(mut last) = last_detection.lock() {
        if let Some(last_time) = *last {
            if now.duration_since(last_time) < Duration::from_millis(500) {
                return;
            }
        }
        *last = Some(now);
    }

    // High confidence detection
    if detection.score > 0.6 {
        let is_awake = awake.load(Ordering::Relaxed);

        // Simple heuristic: if we're already awake and detect again quickly,
        // it might be "cyrup stop" (double detection)
        if is_awake {
            // Check if this is a potential "stop" command
            // In a real system, you'd have a separate model for "cyrup stop"
            println!(
                "🔊 Detected: {} (score: {:.2})",
                detection.name, detection.score
            );

            // For now, toggle on any high-confidence detection when awake
            awake.store(false, Ordering::Relaxed);
            println!("💤 Going to sleep...");
        } else {
            // Wake up on detection
            println!("🔊 Wake word detected! (score: {:.2})", detection.score);
            awake.store(true, Ordering::Relaxed);
        }
    } else if detection.score > 0.4 {
        // Medium confidence - just log it
        println!(
            "🔉 Possible detection: {} (score: {:.2})",
            detection.name, detection.score
        );
    }
}
