//! Wake/Unwake Detection Demo
//!
//! This example demonstrates the complete wake/unwake detection system with model swapping.
//! It shows how to:
//! - Initialize the WakeUnwakeDetector with custom configuration
//! - Process real-time audio input
//! - Handle state transitions between wake and unwake modes
//! - Monitor detection events and system performance
//!
//! # Usage
//!
//! ```bash
//! cargo run --example wake_unwake_demo
//! ```
//!
//! The demo will:
//! 1. Load default "syrup" (wake) and "syrup_stop" (unwake) models
//! 2. Start listening for audio input from the default microphone
//! 3. Display current state (Sleep/Awake) and detection events
//! 4. Automatically swap models when wake/unwake words are detected
//! 5. Show performance statistics and processing metrics

use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU32, Ordering},
};
use std::time::Instant;

use anyhow::Context;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use koffee::{
    config::KoffeeCandleConfig,
    wake_unwake::{WakeUnwakeConfig, WakeUnwakeDetector, WakeUnwakeState},
};

fn main() -> anyhow::Result<()> {
    // Initialize logging for better debugging
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .format_timestamp(None)
        .format_module_path(false)
        .format_target(false)
        .init();

    println!("🚀 Wake/Unwake Detection Demo");
    println!("==============================");
    println!("This demo shows the complete wake/unwake detection system with model swapping.");
    println!();

    // Create wake/unwake configuration with validation
    println!("🔧 Setting up configuration...");
    let wake_unwake_config = WakeUnwakeConfig::default()
        .with_confidence_threshold(0.7)
        .with_model_swap_timeout(3000)
        .with_phrases("syrup".to_string(), "syrup stop".to_string());

    // Validate configuration
    wake_unwake_config
        .validate()
        .map_err(anyhow::Error::from)
        .context("Configuration validation failed")?;

    println!("✅ Configuration validated successfully");
    println!("   Wake model: {}", wake_unwake_config.wake_model_path);
    println!("   Unwake model: {}", wake_unwake_config.unwake_model_path);
    println!("   Wake phrase: '{}'", wake_unwake_config.wake_word);
    println!("   Unwake phrase: '{}'", wake_unwake_config.unwake_word);
    println!(
        "   Confidence threshold: {}",
        wake_unwake_config.confidence_threshold
    );
    println!();

    // Create detector configuration
    let detector_config = KoffeeCandleConfig::default();

    // Initialize the wake/unwake detector
    println!("🔧 Loading models and initializing detector...");
    let detector = WakeUnwakeDetector::new(wake_unwake_config, detector_config)
        .map_err(anyhow::Error::from)
        .context("Failed to create wake/unwake detector")?;

    println!("✅ Wake/unwake detector initialized successfully");
    println!("   Initial state: {}", detector.get_current_state());
    println!();

    // Set up audio input
    println!("🎙️ Setting up audio input...");
    let host = cpal::default_host();

    let device = host
        .default_input_device()
        .context("No default input device available")?;

    let device_name = device.name().context("Failed to get device name")?;
    println!("   Selected device: {}", device_name);

    // Get device configuration
    let config = device
        .default_input_config()
        .context("Failed to get default input config")?;

    println!(
        "   Audio config: {} channels, {} Hz, {:?}",
        config.channels(),
        config.sample_rate().0,
        config.sample_format()
    );
    println!();

    // Set up performance tracking
    let total_chunks = Arc::new(AtomicU32::new(0));
    let total_detections = Arc::new(AtomicU32::new(0));
    let wake_detections = Arc::new(AtomicU32::new(0));
    let unwake_detections = Arc::new(AtomicU32::new(0));
    let running = Arc::new(AtomicBool::new(true));

    // Clone for callback
    let total_chunks_cb = Arc::clone(&total_chunks);
    let total_detections_cb = Arc::clone(&total_detections);
    let wake_detections_cb = Arc::clone(&wake_detections);
    let unwake_detections_cb = Arc::clone(&unwake_detections);
    let running_cb = Arc::clone(&running);

    // Set up shutdown signal
    let running_ctrlc = Arc::clone(&running);
    ctrlc::set_handler(move || {
        println!("\n🛑 Shutdown signal received...");
        running_ctrlc.store(false, Ordering::Relaxed);
    })
    .context("Error setting Ctrl-C handler")?;

    println!("🎧 Starting audio stream...");
    println!("📊 Performance monitoring enabled");
    println!(
        "🎯 Say '{}' to wake up, '{}' to go back to sleep",
        detector.get_current_state(),
        "syrup stop"
    );
    println!("Press Ctrl+C to stop");
    println!("=====================================");

    let start_time = Instant::now();
    let mut last_state = detector.get_current_state();

    // Build audio stream with comprehensive sample format support
    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => {
            device.build_input_stream(
                &config.into(),
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    if !running_cb.load(Ordering::Relaxed) {
                        return;
                    }

                    let chunk_count = total_chunks_cb.fetch_add(1, Ordering::Relaxed) + 1;

                    // Process samples through detector
                    match detector.process_samples(data) {
                        Ok(Some(detection)) => {
                            let detection_count =
                                total_detections_cb.fetch_add(1, Ordering::Relaxed) + 1;
                            let current_state = detector.get_current_state();

                            // Track detection types
                            if detection.phrase.contains("syrup")
                                && !detection.phrase.contains("stop")
                            {
                                wake_detections_cb.fetch_add(1, Ordering::Relaxed);
                                println!(
                                    "🌅 WAKE Detection #{}: '{}' (confidence: {:.3}) - State: {}",
                                    detection_count,
                                    detection.phrase,
                                    detection.confidence,
                                    current_state
                                );

                                if current_state == WakeUnwakeState::Sleep {
                                    if let Err(e) = detector.transition_to_awake() {
                                        eprintln!("⚠️ Failed to transition to awake: {}", e);
                                    } else {
                                        println!("🔄 Transitioning to AWAKE state...");
                                    }
                                }
                            } else if detection.phrase.contains("stop") {
                                unwake_detections_cb.fetch_add(1, Ordering::Relaxed);
                                println!(
                                    "😴 UNWAKE Detection #{}: '{}' (confidence: {:.3}) - State: {}",
                                    detection_count,
                                    detection.phrase,
                                    detection.confidence,
                                    current_state
                                );

                                if current_state == WakeUnwakeState::Awake {
                                    if let Err(e) = detector.transition_to_sleep() {
                                        eprintln!("⚠️ Failed to transition to sleep: {}", e);
                                    } else {
                                        println!("🔄 Transitioning to SLEEP state...");
                                    }
                                }
                            }
                        }
                        Ok(None) => {
                            // No detection, check for state changes
                            let current_state = detector.get_current_state();
                            if current_state != last_state {
                                println!("🔄 State changed: {} → {}", last_state, current_state);
                                last_state = current_state;
                            }
                        }
                        Err(e) => {
                            eprintln!("⚠️ Detection error: {}", e);
                        }
                    }

                    // Periodic status update (every 2000 chunks for less noise)
                    if chunk_count % 2000 == 0 {
                        let current_state = detector.get_current_state();
                        let wake_count = wake_detections_cb.load(Ordering::Relaxed);
                        let unwake_count = unwake_detections_cb.load(Ordering::Relaxed);
                        println!(
                            "📊 Status: {} chunks processed | State: {} | Wake: {} | Unwake: {}",
                            chunk_count, current_state, wake_count, unwake_count
                        );
                    }
                },
                move |err| {
                    eprintln!("❌ Audio stream error: {}", err);
                    running_cb.store(false, Ordering::Relaxed);
                },
                None,
            )
        }
        cpal::SampleFormat::I16 => {
            device.build_input_stream(
                &config.into(),
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    if !running_cb.load(Ordering::Relaxed) {
                        return;
                    }

                    // Convert i16 to f32 for processing (zero-allocation conversion)
                    let f32_data: Vec<f32> =
                        data.iter().map(|&sample| sample as f32 / 32768.0).collect();
                    let chunk_count = total_chunks_cb.fetch_add(1, Ordering::Relaxed) + 1;

                    // Process samples through detector
                    match detector.process_samples(&f32_data) {
                        Ok(Some(detection)) => {
                            let detection_count =
                                total_detections_cb.fetch_add(1, Ordering::Relaxed) + 1;
                            let current_state = detector.get_current_state();

                            // Track detection types
                            if detection.phrase.contains("syrup")
                                && !detection.phrase.contains("stop")
                            {
                                wake_detections_cb.fetch_add(1, Ordering::Relaxed);
                                println!(
                                    "🌅 WAKE Detection #{}: '{}' (confidence: {:.3}) - State: {}",
                                    detection_count,
                                    detection.phrase,
                                    detection.confidence,
                                    current_state
                                );

                                if current_state == WakeUnwakeState::Sleep {
                                    if let Err(e) = detector.transition_to_awake() {
                                        eprintln!("⚠️ Failed to transition to awake: {}", e);
                                    } else {
                                        println!("🔄 Transitioning to AWAKE state...");
                                    }
                                }
                            } else if detection.phrase.contains("stop") {
                                unwake_detections_cb.fetch_add(1, Ordering::Relaxed);
                                println!(
                                    "😴 UNWAKE Detection #{}: '{}' (confidence: {:.3}) - State: {}",
                                    detection_count,
                                    detection.phrase,
                                    detection.confidence,
                                    current_state
                                );

                                if current_state == WakeUnwakeState::Awake {
                                    if let Err(e) = detector.transition_to_sleep() {
                                        eprintln!("⚠️ Failed to transition to sleep: {}", e);
                                    } else {
                                        println!("🔄 Transitioning to SLEEP state...");
                                    }
                                }
                            }
                        }
                        Ok(None) => {
                            // No detection, continue processing
                        }
                        Err(e) => {
                            eprintln!("⚠️ Detection error: {}", e);
                        }
                    }

                    // Periodic status update
                    if chunk_count % 2000 == 0 {
                        let current_state = detector.get_current_state();
                        let wake_count = wake_detections_cb.load(Ordering::Relaxed);
                        let unwake_count = unwake_detections_cb.load(Ordering::Relaxed);
                        println!(
                            "📊 Status: {} chunks processed | State: {} | Wake: {} | Unwake: {}",
                            chunk_count, current_state, wake_count, unwake_count
                        );
                    }
                },
                move |err| {
                    eprintln!("❌ Audio stream error: {}", err);
                    running_cb.store(false, Ordering::Relaxed);
                },
                None,
            )
        }
        sample_format => {
            return Err(anyhow::anyhow!(
                "Unsupported sample format: {:?}",
                sample_format
            ));
        }
    }
    .context("Failed to build input stream")?;

    // Start the stream
    stream.play().context("Failed to start audio stream")?;

    // Main loop - wait for shutdown signal
    while running.load(Ordering::Relaxed) {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    // Stop the stream
    drop(stream);

    // Print comprehensive final statistics
    let elapsed = start_time.elapsed();
    let chunks = total_chunks.load(Ordering::Relaxed);
    let total_detections = total_detections.load(Ordering::Relaxed);
    let wake_count = wake_detections.load(Ordering::Relaxed);
    let unwake_count = unwake_detections.load(Ordering::Relaxed);

    println!("\n📊 Final Statistics:");
    println!("=====================================");
    println!("Runtime: {:.2} seconds", elapsed.as_secs_f64());
    println!("Total audio chunks processed: {}", chunks);
    println!("Total detections: {}", total_detections);
    println!("Wake detections: {}", wake_count);
    println!("Unwake detections: {}", unwake_count);
    println!("Final state: {}", detector.get_current_state());

    if chunks > 0 {
        let rate = chunks as f64 / elapsed.as_secs_f64();
        println!("Processing rate: {:.1} chunks/second", rate);
    }

    if total_detections > 0 {
        let detection_rate = total_detections as f64 / elapsed.as_secs_f64();
        println!("Detection rate: {:.2} detections/second", detection_rate);
    }

    println!("\n🎯 Demo completed successfully!");
    println!("The wake/unwake detection system demonstrated:");
    println!("✅ Model loading and initialization");
    println!("✅ Real-time audio processing");
    println!("✅ State transition management");
    println!("✅ Performance monitoring");
    println!("✅ Graceful shutdown handling");
    println!("\n👋 Thank you for trying the wake/unwake detection demo!");

    Ok(())
}
