//! Test Cyrup Wake Word Models
//!
//! This example demonstrates using both "cyrup" (wake) and "cyrup stop" (unwake)
//! models for accurate voice command detection.
//!
//! Usage:
//!   cargo run --example test_cyrup_models

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleRate, StreamConfig};
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

    // Load wake word model (syrup.rpw for "cyrup")
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

    // Try to load stop model (cyrup_stop.rpw)
    let stop_model = match WakewordModel::load_from_file("cyrup_stop.rpw") {
        Ok(m) => {
            println!("✅ Loaded stop model: cyrup_stop.rpw");
            Some(m)
        }
        Err(_) => {
            println!("⚠️  No stop model found - will use double-tap detection");
            None
        }
    };

    // Configure detectors with optimized settings
    let config = KoffeeCandleConfig {
        detector: DetectorConfig {
            avg_threshold: 0.15, // Lower threshold for better sensitivity
            threshold: 0.45,
            min_scores: 2, // Require fewer scores for faster response
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
                gain_level: 0.095,
                ..Default::default()
            },
        },
        fmt: AudioFmt {
            sample_rate: 16000,
            channels: 1,
            sample_format: koffee::SampleFormat::I16,
            endianness: koffee::Endianness::Little,
        },
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
    let device = host
        .default_input_device()
        .ok_or("No default input device available")?;

    println!("🎙️  Using: {}", device.name()?);

    let stream_config = StreamConfig {
        channels: 1,
        sample_rate: SampleRate(16000),
        buffer_size: cpal::BufferSize::Default,
    };

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

    let mut audio_buffer = Vec::new();
    let chunk_size = 480; // 30ms at 16kHz

    let data_callback = move |data: &[f32], _: &cpal::InputCallbackInfo| {
        // Convert and buffer audio
        for &sample in data {
            let sample_i16 = (sample * i16::MAX as f32) as i16;
            audio_buffer.extend_from_slice(&sample_i16.to_le_bytes());
        }

        // Process chunks
        while audio_buffer.len() >= chunk_size * 2 {
            let chunk: Vec<u8> = audio_buffer.drain(..chunk_size * 2).collect();
            let is_awake = awake_clone.load(Ordering::Relaxed);

            // Always check for wake word
            if let Ok(mut detector) = wake_detector_clone.try_lock() {
                if let Some(detection) = detector.process_bytes(&chunk) {
                    state_clone.add_detection("wake".to_string(), detection.score);
                    handle_wake_detection(detection, &state_clone);
                }
            }

            // Check for stop word if awake and detector exists
            if is_awake {
                if let Ok(stop_opt) = stop_detector_clone.try_lock() {
                    if let Some(ref mut stop_det) = *stop_opt {
                        if let Some(detection) = stop_det.process_bytes(&chunk) {
                            state_clone.add_detection("stop".to_string(), detection.score);
                            handle_stop_detection(detection, &state_clone);
                        }
                    }
                }
            }
        }
    };

    let error_callback = |err| {
        eprintln!("Audio error: {}", err);
    };

    let stream = device.build_input_stream(&stream_config, data_callback, error_callback, None)?;

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
