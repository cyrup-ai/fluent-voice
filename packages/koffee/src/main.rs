//! Koffee-Candle command-line interface
//!
//! $ kfc listen --port 13345
//! $ kfc train  --in data/ --out model.kc
//! $ kfc detect --model model.kc --device "Built-in Microphone" --threshold 0.5 --sample-rate 16000

use anyhow::Context;
use clap::{Parser, Subcommand};
use cpal::traits::{DeviceTrait, HostTrait};
use ctrlc;
use koffee::{
    Endianness, Kfc, KoffeeCandle, SampleFormat,
    config::{AudioFmt, DetectorConfig, FiltersConfig, KoffeeCandleConfig, ScoreMode, VADMode},
    wake_unwake::{WakeUnwakeDetector, WakeUnwakeConfig, WakeUnwakeState},
    wakewords::wakeword_model::ModelType,
    wakewords::{WakewordLoad, WakewordModel},
};
use std::fs;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Parser)]
#[command(author, version, about)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// List available audio input devices
    ListDevices {
        /// Show detailed device information
        #[arg(short, long)]
        verbose: bool,
    },
    /// Run the wake-word detector (TCP stream of scores on --port)
    Listen {
        #[arg(long, default_value_t = 13345)]
        port: u16,
        /// Path to a compiled *.kc* model
        #[arg(long)]
        model: String,
    },
    /// Train a new model from a directory of wav files
    Train {
        /// Directory with label-encoded wavs
        #[arg(long)]
        input: String,
        /// Output path for *.kc* model
        #[arg(long)]
        output: String,
        #[arg(long, default_value_t = ModelType::Small)]
        model_type: ModelType,
    },
    /// List available audio input devices
    Devices {
        /// Show detailed information about each device
        #[arg(short, long)]
        verbose: bool,
    },
    /// Run real-time wake word detection
    Detect {
        /// Path to the wake word model file (default: syrup.rpw)
        #[arg(short, long, default_value = "syrup.rpw")]
        model: String,

        /// Path to the stop word model file (optional)
        #[arg(long)]
        stop_model: Option<String>,

        /// Audio device to use (use 'devices' command to list available devices)
        #[arg(short, long)]
        device: Option<String>,

        /// Detection threshold (0.0 to 1.0)
        #[arg(short, long, default_value_t = 0.5)]
        threshold: f32,

        /// Sample rate in Hz
        #[arg(long, default_value_t = 16000)]
        sample_rate: u32,
    },
    /// Run real-time wake/unwake detection with model swapping
    DetectWakeUnwake {
        /// Path to the wake word model file (default: syrup.rpw)
        #[arg(long, default_value = "syrup.rpw")]
        wake_model: String,

        /// Path to the unwake word model file (default: syrup_stop.rpw)
        #[arg(long, default_value = "syrup_stop.rpw")]
        unwake_model: String,

        /// Audio device to use (use 'devices' command to list available devices)
        #[arg(short, long)]
        device: Option<String>,

        /// Detection threshold (0.0 to 1.0)
        #[arg(short, long, default_value_t = 0.5)]
        threshold: f32,

        /// Sample rate in Hz
        #[arg(long, default_value_t = 16000)]
        sample_rate: u32,
    },
}

fn main() -> anyhow::Result<()> {
    // Set up a panic hook to capture and log panics
    std::panic::set_hook(Box::new(|panic_info| {
        let backtrace = std::backtrace::Backtrace::force_capture();
        log::error!(
            "Thread panicked: {}\nBacktrace:\n{:?}",
            panic_info,
            backtrace
        );
        eprintln!("\n❌ Thread panicked: {}", panic_info);
        eprintln!("\n💡 Check the logs for more details (set RUST_LOG=debug for verbose output)");
    }));

    // Initialize logging
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .format_timestamp(None)
        .format_module_path(false)
        .format_target(false)
        .init();

    log::info!("Starting koffee CLI");
    log::debug!(
        "Command line arguments: {:?}",
        std::env::args().collect::<Vec<_>>()
    );

    // Parse command line arguments
    let cli = Cli::parse();

    // Run the appropriate command with error handling
    log::debug!("Running command: {:?}", cli.cmd);

    // Execute the command and handle any panics
    let result = std::panic::catch_unwind(|| -> anyhow::Result<()> {
        match cli.cmd {
            Cmd::Listen { port, model } => {
                log::info!(
                    "Starting TCP wake word server on port {} with model: {}",
                    port,
                    model
                );
                // TODO: Implement TCP streaming functionality for wake word detection
                // This should start a TCP server that streams detection scores
                anyhow::bail!(
                    "Listen command (TCP streaming) not yet implemented - use 'detect' command for real-time detection"
                )
            }

            Cmd::Train {
                input,
                output,
                model_type,
            } => {
                log::info!("Training model from directory: {}", input);
                koffee::trainer::train_dir(&input, &output, model_type)
                    .map_err(|e| anyhow::anyhow!("Training failed: {}", e))
            }
            Cmd::Devices { verbose } => {
                log::info!("Listing audio devices");
                list_audio_devices(verbose)
            }
            Cmd::Detect {
                model,
                stop_model,
                device,
                threshold,
                sample_rate,
            } => {
                log::info!("Starting wake word detection with model: {}", model);
                if let Some(ref stop_model) = stop_model {
                    log::info!("Using stop model: {}", stop_model);
                }
                run_detection(model, stop_model, device.as_deref(), threshold, sample_rate)
            }
            Cmd::DetectWakeUnwake {
                wake_model,
                unwake_model,
                device,
                threshold,
                sample_rate,
            } => {
                log::info!("Starting wake/unwake detection with wake model: {} and unwake model: {}", wake_model, unwake_model);
                run_wake_unwake_detection(wake_model, unwake_model, device.as_deref(), threshold, sample_rate)
            }

            Cmd::ListDevices { verbose } => {
                log::info!("Listing audio devices");
                list_audio_devices(verbose)
            }
        }
    });

    // Handle panic results
    match result {
        Ok(Ok(())) => {
            log::info!("Command completed successfully");
            Ok(())
        }
        Ok(Err(e)) => {
            log::error!("Command failed: {}", e);
            Err(e)
        }
        Err(panic) => {
            let panic_msg = if let Some(s) = panic.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = panic.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "Unknown panic occurred".to_string()
            };
            log::error!("Panic: {}", panic_msg);
            eprintln!("\n❌ Panic: {}", panic_msg);
            Err(anyhow::anyhow!("Panic: {}", panic_msg))
        }
    }
}

/// Run real-time wake word detection
fn run_detection(
    model_path: String,
    stop_model_path: Option<String>,
    device_name: Option<&str>,
    threshold: f32,
    sample_rate: u32,
) -> anyhow::Result<()> {
    println!("🔍 Loading wake word model from: {}", model_path);

    // Load stop word model if provided
    let stop_model = if let Some(path) = stop_model_path {
        println!("🎤 Loading stop word model from: {}", path);
        Some(
            WakewordModel::load_from_buffer(
                &std::fs::read(&path)
                    .with_context(|| format!("Failed to read stop model file: {}", path))?,
            )
            .with_context(|| format!("Failed to load stop word model: {}", path))?,
        )
    } else {
        println!("ℹ️  No stop word model provided - using double-tap to stop");
        None
    };

    // Configure the detector
    let cfg = KoffeeCandleConfig {
        detector: DetectorConfig {
            avg_threshold: threshold * 0.5, // Slightly lower than main threshold
            threshold,
            min_scores: 1,
            eager: false,
            score_ref: 0.5, // Default value, can be adjusted
            band_size: 5,
            score_mode: ScoreMode::Max,
            vad_mode: Some(VADMode::Medium), // Use Medium VAD mode
            #[cfg(feature = "record")]
            record_path: None,
        },
        filters: FiltersConfig::default(),
        fmt: AudioFmt {
            sample_rate: sample_rate as usize, // Convert u32 to usize
            channels: 1,
            sample_format: SampleFormat::F32, // Use koffee's SampleFormat
            endianness: Endianness::Little,   // Use koffee's Endianness
        },

    };

    // Create the detector
    println!("🔧 Creating detector with config: {:#?}", cfg);
    let mut detector =
        Kfc::new(&cfg).map_err(|e| anyhow::anyhow!("Failed to create detector: {}", e))?;

    // Add wake word model
    println!("🔍 Loading wake word model from: {}", model_path);
    let model_data =
        fs::read(&model_path).map_err(|e| anyhow::anyhow!("Failed to read model file: {}", e))?;
    let wake_model = WakewordModel::load_from_buffer(&model_data)
        .map_err(|e| anyhow::anyhow!("Failed to load wake word model: {}", e))?;

    println!("✅ Successfully loaded wake word model");

    detector
        .add_wakeword_model(wake_model)
        .map_err(|e| anyhow::anyhow!("Failed to add wake word model: {}", e))?;

    // Add stop word model if provided
    if let Some(stop_model) = stop_model {
        detector
            .add_wakeword_model(stop_model)
            .map_err(|e| anyhow::anyhow!("Failed to add stop word model: {}", e))?;
    }

    println!("🎤 Starting wake word detection (press Ctrl+C to exit)...");
    println!("🔊 Listening for wake word...");

    // Get the audio host and device
    let host = cpal::default_host();
    println!("🔊 Enumerating audio devices...");
    let devices: Vec<_> = host.input_devices()?.collect();
    println!("   Found {} audio input devices", devices.len());
    for (i, dev) in devices.iter().enumerate() {
        println!(
            "   {}. {}",
            i + 1,
            dev.name().unwrap_or_else(|_| "<unknown>".to_string())
        );
    }

    let device = match device_name {
        Some(name) => {
            println!("🔍 Looking for device containing: {}", name);
            host.input_devices()?
                .find(|d| {
                    let dev_name = d.name().unwrap_or_default();
                    println!("   Checking device: {}", dev_name);
                    dev_name.to_lowercase().contains(&name.to_lowercase())
                })
                .ok_or_else(|| {
                    eprintln!("❌ No input device found containing '{}'", name);
                    anyhow::anyhow!("No input device found containing '{}'", name)
                })?
        }
        None => {
            println!("🔍 Using default input device");
            host.default_input_device().ok_or_else(|| {
                eprintln!("❌ No default input device found");
                anyhow::anyhow!("No default input device found")
            })?
        }
    };

    println!("✅ Selected audio device: {}", device.name()?);

    // Configure the audio stream with detailed error handling
    let config = cpal::StreamConfig {
        channels: 1,
        sample_rate: cpal::SampleRate(sample_rate),
        buffer_size: cpal::BufferSize::Default,
    };

    // Create thread-safe data structures
    let running = Arc::new(AtomicBool::new(true));
    let processed_chunks = Arc::new(AtomicU32::new(0));
    let detections = Arc::new(AtomicU32::new(0));
    let start_time = Instant::now();

    // Handle Ctrl+C to gracefully exit
    ctrlc::set_handler({
        let running_handler = running.clone();
        move || {
            println!("\\n🛑 Stopping detection...");
            running_handler.store(false, Ordering::SeqCst);
        }
    })
    .map_err(|e| anyhow::anyhow!("Failed to set Ctrl+C handler: {}", e))?;

    // Create a thread-safe reference to the detector
    let detector: Arc<Mutex<KoffeeCandle>> = Arc::new(Mutex::new(detector));

    // Set up the data callback
    let _stream = {
        let detector_clone = Arc::clone(&detector);
        let processed_chunks_clone = Arc::clone(&processed_chunks);
        let detections_clone = Arc::clone(&detections);

        let data_fn = move |data: &[f32], _info: &cpal::InputCallbackInfo| {
            // Process the audio data through the detector
            if let Ok(mut detector_lock) = detector_clone.try_lock() {
                if let Some(detection) = detector_lock.process_samples(data) {
                    processed_chunks_clone.fetch_add(1, Ordering::Relaxed);
                    detections_clone.fetch_add(1, Ordering::Relaxed);
                    println!("🎯 Wake word detected! Score: {:.2}", detection.score);
                }
            }
        };

        let err_fn = |err: cpal::StreamError| {
            eprintln!("🔴 Audio stream error: {}", err);
        };

        device.build_input_stream(&config, data_fn, err_fn, None)?
    };

    println!("🎧 Audio stream started successfully. Speak your wake word...");
    println!("    (Press Ctrl+C to stop)");

    // Main loop
    while running.load(Ordering::SeqCst) {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    // Calculate final stats
    let elapsed = start_time.elapsed();
    let chunks = processed_chunks.load(Ordering::Relaxed);
    let total_detections = detections.load(Ordering::Relaxed);

    println!("\\n📊 Final Statistics:");
    println!("=================");
    println!("Total runtime: {:.1} seconds", elapsed.as_secs_f64());
    println!("Audio chunks processed: {}", chunks);
    println!("Total detections: {}", total_detections);

    if chunks > 0 {
        let rate = chunks as f64 / elapsed.as_secs_f64();
        println!("Processing rate: {:.1} chunks/second", rate);
    }

    println!("\\n👋 Exiting...");
    Ok(())
}

/// List available audio input devices
fn list_audio_devices(verbose: bool) -> anyhow::Result<()> {
    println!("🎙️ Available audio input devices:");
    println!("==================================");

    let host = cpal::default_host();

    // List all available input devices
    let devices = host
        .input_devices()
        .map_err(|e| anyhow::anyhow!("Failed to get input devices: {}", e))?;

    for (i, device) in devices.enumerate() {
        let name = device
            .name()
            .map_err(|e| anyhow::anyhow!("Failed to get device name: {}", e))?;

        let default = match host.default_input_device() {
            Some(default_device) => {
                if device.name().unwrap_or_default() == default_device.name().unwrap_or_default() {
                    " (default)"
                } else {
                    ""
                }
            }
            None => "",
        };

        println!("{}. {}{}", i + 1, name, default);

        if verbose {
            // List supported input configurations
            if let Ok(configs) = device.supported_input_configs() {
                println!("   Supported configurations:");
                for (j, config) in configs.enumerate() {
                    println!(
                        "   {}.{: <3} Sample rate: {}-{} Hz, Channels: {}, Format: {:?}",
                        i + 1,
                        j + 1,
                        config.min_sample_rate().0,
                        config.max_sample_rate().0,
                        config.channels(),
                        config.sample_format()
                    );
                }
            } else {
                println!("   No supported input configurations found");
            }
            println!();
        }
    }

    Ok(())
}

/// Run wake/unwake detection with model swapping
fn run_wake_unwake_detection(
    wake_model_path: String,
    unwake_model_path: String,
    device_name: Option<&str>,
    threshold: f32,
    sample_rate: u32,
) -> anyhow::Result<()> {
    println!("🚀 Initializing wake/unwake detection system...");
    println!("Wake model: {}", wake_model_path);
    println!("Unwake model: {}", unwake_model_path);
    println!("Threshold: {}", threshold);
    println!("Sample rate: {} Hz", sample_rate);

    // Create wake/unwake configuration
    let wake_unwake_config = WakeUnwakeConfig {
        wake_model_path: wake_model_path.clone(),
        unwake_model_path: unwake_model_path.clone(),
        wake_phrase: "syrup".to_string(),
        unwake_phrase: "syrup stop".to_string(),
        confidence_threshold: threshold,
        state_timeout_ms: 30000, // 30 seconds timeout
    };

    // Create detector configuration
    let mut detector_config = KoffeeCandleConfig::default();
    detector_config.fmt.sample_rate = sample_rate as usize;
    detector_config.detector.threshold = threshold;

    // Initialize the wake/unwake detector
    println!("🔧 Loading models and initializing detector...");
    let detector = WakeUnwakeDetector::new(wake_unwake_config, detector_config)
        .context("Failed to create wake/unwake detector")?;

    println!("✅ Wake/unwake detector initialized successfully");

    // Set up audio input
    let host = cpal::default_host();
    
    // Select audio device
    let device = if let Some(device_name) = device_name {
        println!("🎙️ Looking for audio device: {}", device_name);
        host.input_devices()
            .context("Failed to enumerate input devices")?
            .find(|d| d.name().unwrap_or_default().contains(device_name))
            .with_context(|| format!("Audio device '{}' not found", device_name))?
    } else {
        println!("🎙️ Using default audio input device");
        host.default_input_device()
            .context("No default input device available")?
    };

    let device_name = device.name().context("Failed to get device name")?;
    println!("🎙️ Selected audio device: {}", device_name);

    // Get device configuration
    let config = device
        .default_input_config()
        .context("Failed to get default input config")?;

    println!("🔧 Audio config: {} channels, {} Hz, {:?}", 
        config.channels(), config.sample_rate().0, config.sample_format());

    // Set up atomic counters for statistics
    let total_chunks = Arc::new(AtomicU32::new(0));
    let total_detections = Arc::new(AtomicU32::new(0));
    let running = Arc::new(AtomicBool::new(true));

    // Clone for callback
    let total_chunks_cb = Arc::clone(&total_chunks);
    let total_detections_cb = Arc::clone(&total_detections);
    let running_cb = Arc::clone(&running);

    // Set up shutdown signal
    let running_ctrlc = Arc::clone(&running);
    ctrlc::set_handler(move || {
        println!("\n🛑 Shutdown signal received...");
        running_ctrlc.store(false, Ordering::Relaxed);
    }).context("Error setting Ctrl-C handler")?;

    println!("🎧 Starting audio stream...");
    println!("Current state: {}", detector.get_current_state());
    println!("Press Ctrl+C to stop");
    println!("=====================================");

    let start_time = Instant::now();

    // Build audio stream with zero-allocation callback
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
                            let detection_count = total_detections_cb.fetch_add(1, Ordering::Relaxed) + 1;
                            let current_state = detector.get_current_state();
                            
                            println!("🎯 Detection #{}: {} (confidence: {:.3}) - State: {}", 
                                detection_count, detection.phrase, detection.confidence, current_state);

                            // Handle state transitions
                            if current_state == WakeUnwakeState::Sleep && detection.phrase.contains("syrup") {
                                if let Err(e) = detector.transition_to_awake() {
                                    eprintln!("⚠️ Failed to transition to awake: {}", e);
                                }
                            } else if current_state == WakeUnwakeState::Awake && detection.phrase.contains("stop") {
                                if let Err(e) = detector.transition_to_sleep() {
                                    eprintln!("⚠️ Failed to transition to sleep: {}", e);
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

                    // Periodic status update (every 1000 chunks)
                    if chunk_count % 1000 == 0 {
                        let current_state = detector.get_current_state();
                        println!("📊 Processed {} chunks - Current state: {}", chunk_count, current_state);
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

                    // Convert i16 to f32 for processing
                    let f32_data: Vec<f32> = data.iter().map(|&sample| sample as f32 / 32768.0).collect();
                    let chunk_count = total_chunks_cb.fetch_add(1, Ordering::Relaxed) + 1;

                    // Process samples through detector
                    match detector.process_samples(&f32_data) {
                        Ok(Some(detection)) => {
                            let detection_count = total_detections_cb.fetch_add(1, Ordering::Relaxed) + 1;
                            let current_state = detector.get_current_state();
                            
                            println!("🎯 Detection #{}: {} (confidence: {:.3}) - State: {}", 
                                detection_count, detection.phrase, detection.confidence, current_state);

                            // Handle state transitions
                            if current_state == WakeUnwakeState::Sleep && detection.phrase.contains("syrup") {
                                if let Err(e) = detector.transition_to_awake() {
                                    eprintln!("⚠️ Failed to transition to awake: {}", e);
                                }
                            } else if current_state == WakeUnwakeState::Awake && detection.phrase.contains("stop") {
                                if let Err(e) = detector.transition_to_sleep() {
                                    eprintln!("⚠️ Failed to transition to sleep: {}", e);
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

                    // Periodic status update (every 1000 chunks)
                    if chunk_count % 1000 == 0 {
                        let current_state = detector.get_current_state();
                        println!("📊 Processed {} chunks - Current state: {}", chunk_count, current_state);
                    }
                },
                move |err| {
                    eprintln!("❌ Audio stream error: {}", err);
                    running_cb.store(false, Ordering::Relaxed);
                },
                None,
            )
        }
        cpal::SampleFormat::U16 => {
            device.build_input_stream(
                &config.into(),
                move |data: &[u16], _: &cpal::InputCallbackInfo| {
                    if !running_cb.load(Ordering::Relaxed) {
                        return;
                    }

                    // Convert u16 to f32 for processing
                    let f32_data: Vec<f32> = data.iter().map(|&sample| (sample as f32 - 32768.0) / 32768.0).collect();
                    let chunk_count = total_chunks_cb.fetch_add(1, Ordering::Relaxed) + 1;

                    // Process samples through detector
                    match detector.process_samples(&f32_data) {
                        Ok(Some(detection)) => {
                            let detection_count = total_detections_cb.fetch_add(1, Ordering::Relaxed) + 1;
                            let current_state = detector.get_current_state();
                            
                            println!("🎯 Detection #{}: {} (confidence: {:.3}) - State: {}", 
                                detection_count, detection.phrase, detection.confidence, current_state);

                            // Handle state transitions
                            if current_state == WakeUnwakeState::Sleep && detection.phrase.contains("syrup") {
                                if let Err(e) = detector.transition_to_awake() {
                                    eprintln!("⚠️ Failed to transition to awake: {}", e);
                                }
                            } else if current_state == WakeUnwakeState::Awake && detection.phrase.contains("stop") {
                                if let Err(e) = detector.transition_to_sleep() {
                                    eprintln!("⚠️ Failed to transition to sleep: {}", e);
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

                    // Periodic status update (every 1000 chunks)
                    if chunk_count % 1000 == 0 {
                        let current_state = detector.get_current_state();
                        println!("📊 Processed {} chunks - Current state: {}", chunk_count, current_state);
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
            return Err(anyhow::anyhow!("Unsupported sample format: {:?}", sample_format));
        }
    }.context("Failed to build input stream")?;

    // Start the stream
    stream.play().context("Failed to start audio stream")?;

    // Main loop - wait for shutdown signal
    while running.load(Ordering::Relaxed) {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    // Stop the stream
    drop(stream);

    // Print final statistics
    let elapsed = start_time.elapsed();
    let chunks = total_chunks.load(Ordering::Relaxed);
    let total_detections = total_detections.load(Ordering::Relaxed);

    println!("\n📊 Final Statistics:");
    println!("=====================================");
    println!("Runtime: {:.2} seconds", elapsed.as_secs_f64());
    println!("Total audio chunks processed: {}", chunks);
    println!("Total detections: {}", total_detections);

    if chunks > 0 {
        let rate = chunks as f64 / elapsed.as_secs_f64();
        println!("Processing rate: {:.1} chunks/second", rate);
    }

    println!("\n👋 Exiting wake/unwake detection...");
    Ok(())
}
