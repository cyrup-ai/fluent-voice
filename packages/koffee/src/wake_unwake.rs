//! Wake/Unwake State Machine Module
//!
//! This module provides a production-quality wake/unwake detection system that manages
//! state transitions between sleep and awake modes with dynamic model swapping.
//!
//! # Architecture
//!
//! The system uses a lock-free architecture with crossbeam channels for communication
//! between threads. Model swapping happens asynchronously to avoid audio dropouts.
//!
//! # Example
//!
//! ```rust
//! use koffee::wake_unwake::{WakeUnwakeDetector, WakeUnwakeConfig};
//! use koffee::config::KoffeeCandleConfig;
//!
//! let config = WakeUnwakeConfig::default();
//! let detector_config = KoffeeCandleConfig::default();
//! let detector = WakeUnwakeDetector::new(config, detector_config)?;
//! ```

use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender, TryRecvError, bounded};
use tracing::{debug, error, info, instrument, warn};

use crate::KoffeeCandle;
use crate::config::KoffeeCandleConfig;
use crate::wakewords::{WakewordLoad, WakewordModel};

/// Wake/Unwake state enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WakeUnwakeState {
    /// System is in sleep mode, listening for wake words
    Sleep,
    /// System is awake, listening for unwake words
    Awake,
}

impl Default for WakeUnwakeState {
    fn default() -> Self {
        Self::Sleep
    }
}

impl std::fmt::Display for WakeUnwakeState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sleep => write!(f, "Sleep"),
            Self::Awake => write!(f, "Awake"),
        }
    }
}

/// Configuration for wake/unwake detection system
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WakeUnwakeConfig {
    /// Path to the wake word model file
    pub wake_model_path: String,
    /// Path to the unwake word model file
    pub unwake_model_path: String,
    /// Wake word phrase
    pub wake_word: String,
    /// Unwake word phrase
    pub unwake_word: String,
    /// Detection confidence threshold (0.0 to 1.0)
    pub confidence_threshold: f32,
    /// Model swap timeout in milliseconds
    pub model_swap_timeout_ms: u64,
}

impl Default for WakeUnwakeConfig {
    fn default() -> Self {
        Self {
            wake_model_path: "syrup.rpw".to_string(),
            unwake_model_path: "syrup_stop.rpw".to_string(),
            wake_word: "syrup".to_string(),
            unwake_word: "syrup stop".to_string(),
            confidence_threshold: 0.7,
            model_swap_timeout_ms: 5000,
        }
    }
}

impl WakeUnwakeConfig {
    /// Validate the configuration and check that model files exist
    pub fn validate(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Validate confidence threshold
        if !(0.0..=1.0).contains(&self.confidence_threshold) {
            return Err(format!(
                "Confidence threshold must be between 0.0 and 1.0, got: {}",
                self.confidence_threshold
            )
            .into());
        }

        // Validate model swap timeout
        if self.model_swap_timeout_ms == 0 {
            return Err("Model swap timeout must be greater than 0".into());
        }

        // Check that wake model file exists and is readable
        if !std::path::Path::new(&self.wake_model_path).exists() {
            return Err(format!("Wake model file does not exist: {}", self.wake_model_path).into());
        }

        // Check that unwake model file exists and is readable
        if !std::path::Path::new(&self.unwake_model_path).exists() {
            return Err(format!(
                "Unwake model file does not exist: {}",
                self.unwake_model_path
            )
            .into());
        }

        // Validate that model files are readable
        std::fs::File::open(&self.wake_model_path).map_err(|e| {
            format!(
                "Cannot read wake model file '{}': {}",
                self.wake_model_path, e
            )
        })?;

        std::fs::File::open(&self.unwake_model_path).map_err(|e| {
            format!(
                "Cannot read unwake model file '{}': {}",
                self.unwake_model_path, e
            )
        })?;

        // Validate that wake and unwake phrases are not empty
        if self.wake_word.trim().is_empty() {
            return Err("Wake word phrase cannot be empty".into());
        }

        if self.unwake_word.trim().is_empty() {
            return Err("Unwake word phrase cannot be empty".into());
        }

        Ok(())
    }

    /// Create a new configuration with custom model paths
    pub fn with_models(wake_model_path: String, unwake_model_path: String) -> Self {
        Self {
            wake_model_path,
            unwake_model_path,
            ..Default::default()
        }
    }

    /// Set the confidence threshold
    pub fn with_confidence_threshold(mut self, threshold: f32) -> Self {
        self.confidence_threshold = threshold;
        self
    }

    /// Set the model swap timeout
    pub fn with_model_swap_timeout(mut self, timeout_ms: u64) -> Self {
        self.model_swap_timeout_ms = timeout_ms;
        self
    }

    /// Set the wake and unwake phrases
    pub fn with_phrases(mut self, wake_word: String, unwake_word: String) -> Self {
        self.wake_word = wake_word;
        self.unwake_word = unwake_word;
        self
    }
}

/// Commands sent to the detector thread
#[derive(Debug)]
enum DetectorCommand {
    /// Process audio samples
    ProcessSamples(Vec<f32>),
    /// Swap to wake model
    SwapToWakeModel,
    /// Swap to unwake model
    SwapToUnwakeModel,
    /// Shutdown the detector
    Shutdown,
}

/// Events sent from the detector thread
#[derive(Debug, Clone)]
pub enum DetectorEvent {
    /// Wake word detected
    WakeDetected { confidence: f32, timestamp: Instant },
    /// Unwake word detected
    UnwakeDetected { confidence: f32, timestamp: Instant },
    /// Model swap completed successfully
    ModelSwapCompleted {
        new_state: WakeUnwakeState,
        swap_duration: Duration,
    },
    /// Error occurred during processing
    Error { message: String, timestamp: Instant },
}

/// Performance metrics for the wake/unwake detector
#[derive(Debug, Default)]
pub struct DetectorMetrics {
    /// Total number of audio samples processed
    pub samples_processed: AtomicU64,
    /// Total number of wake detections
    pub wake_detections: AtomicU64,
    /// Total number of unwake detections
    pub unwake_detections: AtomicU64,
    /// Total number of model swaps
    pub model_swaps: AtomicU64,
    /// Average processing time per audio buffer (microseconds)
    pub avg_processing_time_us: AtomicU64,
    /// Total processing time (microseconds)
    pub total_processing_time_us: AtomicU64,
    /// Number of processing iterations
    pub processing_iterations: AtomicU64,
}

impl DetectorMetrics {
    /// Update processing time metrics
    #[inline]
    pub fn update_processing_time(&self, duration: Duration) {
        let duration_us = duration.as_micros() as u64;
        self.total_processing_time_us
            .fetch_add(duration_us, Ordering::Relaxed);
        let iterations = self.processing_iterations.fetch_add(1, Ordering::Relaxed) + 1;
        let total_time = self.total_processing_time_us.load(Ordering::Relaxed);
        self.avg_processing_time_us
            .store(total_time / iterations, Ordering::Relaxed);
    }

    /// Increment wake detection counter
    #[inline]
    pub fn increment_wake_detections(&self) {
        self.wake_detections.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment unwake detection counter
    #[inline]
    pub fn increment_unwake_detections(&self) {
        self.unwake_detections.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment model swap counter
    #[inline]
    pub fn increment_model_swaps(&self) {
        self.model_swaps.fetch_add(1, Ordering::Relaxed);
    }

    /// Add to samples processed counter
    #[inline]
    pub fn add_samples_processed(&self, count: u64) {
        self.samples_processed.fetch_add(count, Ordering::Relaxed);
    }
}

/// Wake/Unwake detector with lock-free architecture
pub struct WakeUnwakeDetector {
    /// Current state of the detector
    current_state: Arc<std::sync::atomic::AtomicU8>,
    /// Configuration
    config: WakeUnwakeConfig,
    /// Command sender to detector thread
    command_sender: Sender<DetectorCommand>,
    /// Event receiver from detector thread
    event_receiver: Receiver<DetectorEvent>,
    /// Performance metrics
    metrics: Arc<DetectorMetrics>,
    /// Shutdown flag
    shutdown: Arc<AtomicBool>,
}

impl WakeUnwakeDetector {
    /// Create a new wake/unwake detector
    #[instrument(skip(detector_config))]
    pub fn new(
        config: WakeUnwakeConfig,
        detector_config: KoffeeCandleConfig,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        info!("Initializing WakeUnwakeDetector with config: {:?}", config);

        // Create channels for communication
        let (command_sender, command_receiver) = bounded(1000);
        let (event_sender, event_receiver) = bounded(1000);

        // Shared state and metrics
        let current_state = Arc::new(std::sync::atomic::AtomicU8::new(
            WakeUnwakeState::Sleep as u8,
        ));
        let metrics = Arc::new(DetectorMetrics::default());
        let shutdown = Arc::new(AtomicBool::new(false));

        // Clone for the detector thread
        let thread_config = config.clone();
        let thread_state = Arc::clone(&current_state);
        let thread_metrics = Arc::clone(&metrics);
        let thread_shutdown = Arc::clone(&shutdown);

        // Spawn the detector thread
        std::thread::Builder::new()
            .name("wake-unwake-detector".to_string())
            .spawn(move || {
                Self::detector_thread(
                    thread_config,
                    detector_config,
                    command_receiver,
                    event_sender,
                    thread_state,
                    thread_metrics,
                    thread_shutdown,
                )
            })
            .map_err(|e| format!("Failed to spawn detector thread: {e}"))?;

        Ok(Self {
            current_state,
            config,
            command_sender,
            event_receiver,
            metrics,
            shutdown,
        })
    }

    /// Get the current state
    #[inline]
    pub fn current_state(&self) -> WakeUnwakeState {
        match self.current_state.load(Ordering::Acquire) {
            0 => WakeUnwakeState::Sleep,
            1 => WakeUnwakeState::Awake,
            _ => WakeUnwakeState::Sleep, // Default fallback
        }
    }

    /// Process audio samples
    #[instrument(skip(self, samples))]
    pub fn process_samples(
        &self,
        samples: Vec<f32>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.shutdown.load(Ordering::Acquire) {
            return Err("Detector is shutting down".into());
        }

        self.command_sender
            .try_send(DetectorCommand::ProcessSamples(samples))
            .map_err(|e| format!("Failed to send samples to detector: {e}"))?;

        Ok(())
    }

    /// Get the next event from the detector
    #[inline]
    pub fn try_recv_event(&self) -> Result<DetectorEvent, TryRecvError> {
        self.event_receiver.try_recv()
    }

    /// Wait for the next event with timeout
    #[instrument(skip(self))]
    pub fn recv_event_timeout(
        &self,
        timeout: Duration,
    ) -> Result<DetectorEvent, crossbeam_channel::RecvTimeoutError> {
        self.event_receiver.recv_timeout(timeout)
    }

    /// Get performance metrics
    #[inline]
    pub fn metrics(&self) -> &DetectorMetrics {
        &self.metrics
    }

    /// Get configuration
    #[inline]
    pub fn config(&self) -> &WakeUnwakeConfig {
        &self.config
    }

    /// Shutdown the detector
    #[instrument(skip(self))]
    pub fn shutdown(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("Shutting down WakeUnwakeDetector");

        self.shutdown.store(true, Ordering::Release);

        self.command_sender
            .try_send(DetectorCommand::Shutdown)
            .map_err(|e| format!("Failed to send shutdown command: {e}"))?;

        Ok(())
    }

    /// Detector thread main loop
    #[instrument(skip_all)]
    fn detector_thread(
        config: WakeUnwakeConfig,
        detector_config: KoffeeCandleConfig,
        command_receiver: Receiver<DetectorCommand>,
        event_sender: Sender<DetectorEvent>,
        current_state: Arc<std::sync::atomic::AtomicU8>,
        metrics: Arc<DetectorMetrics>,
        shutdown: Arc<AtomicBool>,
    ) {
        info!("Starting detector thread");

        // Load initial wake model
        let mut current_detector = match Self::load_wake_model(&config, &detector_config) {
            Ok(detector) => detector,
            Err(e) => {
                error!("Failed to load initial wake model: {}", e);
                let _ = event_sender.try_send(DetectorEvent::Error {
                    message: format!("Failed to load initial wake model: {e}"),
                    timestamp: Instant::now(),
                });
                return;
            }
        };

        info!("Initial wake model loaded successfully");

        // Main processing loop
        while !shutdown.load(Ordering::Acquire) {
            match command_receiver.recv_timeout(Duration::from_millis(100)) {
                Ok(command) => {
                    let start_time = Instant::now();

                    match command {
                        DetectorCommand::ProcessSamples(samples) => {
                            Self::process_samples_internal(
                                &mut current_detector,
                                &samples,
                                &config,
                                &current_state,
                                &event_sender,
                                &metrics,
                            );
                        }
                        DetectorCommand::SwapToWakeModel => {
                            match Self::load_wake_model(&config, &detector_config) {
                                Ok(new_detector) => {
                                    current_detector = new_detector;
                                    current_state
                                        .store(WakeUnwakeState::Sleep as u8, Ordering::Release);
                                    metrics.increment_model_swaps();
                                    let swap_duration = start_time.elapsed();

                                    let _ =
                                        event_sender.try_send(DetectorEvent::ModelSwapCompleted {
                                            new_state: WakeUnwakeState::Sleep,
                                            swap_duration,
                                        });

                                    info!("Swapped to wake model in {:?}", swap_duration);
                                }
                                Err(e) => {
                                    error!("Failed to swap to wake model: {}", e);
                                    let _ = event_sender.try_send(DetectorEvent::Error {
                                        message: format!("Failed to swap to wake model: {e}"),
                                        timestamp: Instant::now(),
                                    });
                                }
                            }
                        }
                        DetectorCommand::SwapToUnwakeModel => {
                            match Self::load_unwake_model(&config, &detector_config) {
                                Ok(new_detector) => {
                                    current_detector = new_detector;
                                    current_state
                                        .store(WakeUnwakeState::Awake as u8, Ordering::Release);
                                    metrics.increment_model_swaps();
                                    let swap_duration = start_time.elapsed();

                                    let _ =
                                        event_sender.try_send(DetectorEvent::ModelSwapCompleted {
                                            new_state: WakeUnwakeState::Awake,
                                            swap_duration,
                                        });

                                    info!("Swapped to unwake model in {:?}", swap_duration);
                                }
                                Err(e) => {
                                    error!("Failed to swap to unwake model: {}", e);
                                    let _ = event_sender.try_send(DetectorEvent::Error {
                                        message: format!("Failed to swap to unwake model: {e}"),
                                        timestamp: Instant::now(),
                                    });
                                }
                            }
                        }
                        DetectorCommand::Shutdown => {
                            info!("Detector thread received shutdown command");
                            break;
                        }
                    }

                    metrics.update_processing_time(start_time.elapsed());
                }
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                    // Normal timeout, continue loop
                    continue;
                }
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                    warn!("Command channel disconnected, shutting down detector thread");
                    break;
                }
            }
        }

        info!("Detector thread shutting down");
    }

    /// Process audio samples internally
    #[inline]
    fn process_samples_internal(
        detector: &mut KoffeeCandle,
        samples: &[f32],
        config: &WakeUnwakeConfig,
        current_state: &Arc<std::sync::atomic::AtomicU8>,
        event_sender: &Sender<DetectorEvent>,
        metrics: &Arc<DetectorMetrics>,
    ) {
        metrics.add_samples_processed(samples.len() as u64);

        if let Some(detection) = detector.process_samples(samples)
            && detection.score >= config.confidence_threshold
        {
            let current_state_val = current_state.load(Ordering::Acquire);
            let timestamp = Instant::now();

            match current_state_val {
                0 => {
                    // Sleep state - wake word detected
                    metrics.increment_wake_detections();
                    debug!("Wake word detected with confidence: {}", detection.score);

                    let _ = event_sender.try_send(DetectorEvent::WakeDetected {
                        confidence: detection.score,
                        timestamp,
                    });
                }
                1 => {
                    // Awake state - unwake word detected
                    metrics.increment_unwake_detections();
                    debug!("Unwake word detected with confidence: {}", detection.score);

                    let _ = event_sender.try_send(DetectorEvent::UnwakeDetected {
                        confidence: detection.score,
                        timestamp,
                    });
                }
                _ => {
                    warn!("Invalid state value: {}", current_state_val);
                }
            }
        }
    }

    /// Load wake word model
    #[instrument(skip(detector_config))]
    fn load_wake_model(
        config: &WakeUnwakeConfig,
        detector_config: &KoffeeCandleConfig,
    ) -> Result<KoffeeCandle, Box<dyn std::error::Error + Send + Sync>> {
        debug!("Loading wake model from: {}", config.wake_model_path);

        let model_data = std::fs::read(&config.wake_model_path).map_err(|e| {
            format!(
                "Failed to read wake model file '{}': {}",
                config.wake_model_path, e
            )
        })?;

        let model = WakewordModel::load_from_buffer(&model_data)
            .map_err(|e| format!("Failed to load wake model: {e}"))?;

        let mut detector = KoffeeCandle::new(detector_config)
            .map_err(|e| format!("Failed to create wake detector: {e}"))?;

        detector
            .add_wakeword_model(model)
            .map_err(|e| format!("Failed to add wake model: {e}"))?;

        Ok(detector)
    }

    /// Load unwake word model
    #[instrument(skip(detector_config))]
    fn load_unwake_model(
        config: &WakeUnwakeConfig,
        detector_config: &KoffeeCandleConfig,
    ) -> Result<KoffeeCandle, Box<dyn std::error::Error + Send + Sync>> {
        debug!("Loading unwake model from: {}", config.unwake_model_path);

        let model_data = std::fs::read(&config.unwake_model_path).map_err(|e| {
            format!(
                "Failed to read unwake model file '{}': {}",
                config.unwake_model_path, e
            )
        })?;

        let model = WakewordModel::load_from_buffer(&model_data)
            .map_err(|e| format!("Failed to load unwake model: {e}"))?;

        let mut detector = KoffeeCandle::new(detector_config)
            .map_err(|e| format!("Failed to create unwake detector: {e}"))?;

        detector
            .add_wakeword_model(model)
            .map_err(|e| format!("Failed to add unwake model: {e}"))?;

        Ok(detector)
    }

    /// Trigger state transition to awake (swap to unwake model)
    #[instrument(skip(self))]
    pub fn transition_to_awake(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("Transitioning to awake state");

        self.command_sender
            .try_send(DetectorCommand::SwapToUnwakeModel)
            .map_err(|e| format!("Failed to send swap to unwake model command: {e}"))?;

        Ok(())
    }

    /// Trigger state transition to sleep (swap to wake model)
    #[instrument(skip(self))]
    pub fn transition_to_sleep(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("Transitioning to sleep state");

        self.command_sender
            .try_send(DetectorCommand::SwapToWakeModel)
            .map_err(|e| format!("Failed to send swap to wake model command: {e}"))?;

        Ok(())
    }
}

impl Drop for WakeUnwakeDetector {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}
