//! Koffee-Candle – public crate root
//! =================================
//! Cross-platform **wake–word** detector (KFC front-end + Candle back-end).
//!
//! * **Desktop** builds use SIMD + Rayon (default feature set).
//! * **WASM** builds leave SIMD/Rayon off and expose a tiny JS binding layer.
//!
//! The library is **self-contained**: you embed it, feed PCM frames (bytes or
//! `f32` samples) and receive [`KoffeeCandleDetection`] whenever any loaded
//! wake-word fires.
//
//  ───────────────────────────────────────────────────────────────────────────
//  lints: keep the public surface clean while allowing “private dead code” in
//  helper modules that are only used by the demo CLIs / integration tests.
//  ───────────────────────────────────────────────────────────────────────────
//! # Koffee Wake Word Detection Library
//! 
//! Koffee is a sophisticated cross-platform wake word detection system built using the Candle ML framework.
//! It provides real-time wake word detection with configurable thresholds, multiple model sizes,
//! advanced audio processing capabilities, and comprehensive training tools.
//! 
//! ## Features
//! 
//! - **Real-time Wake Word Detection**: High-performance audio processing with configurable detection thresholds
//! - **Multiple Model Sizes**: Support for Tiny, Small, Medium, and Large models to balance accuracy vs performance
//! - **Advanced Audio Processing**: Built-in resampling, filtering, and feature extraction
//! - **Training Pipeline**: Complete training system with TTS-based synthetic sample generation
//! - **TCP Server**: Network-based detection service with connection management and rate limiting
//! - **Cross-platform Audio**: Support for multiple audio devices and formats using CPAL
//! - **Production Ready**: Comprehensive error handling, logging, and monitoring capabilities
//! 
//! ## Quick Start
//! 
//! ### Basic Wake Word Detection
//! 
//! ```rust
//! use koffee::{KoffeeCandle, KoffeeCandleConfig};
//! use koffee::wakewords::WakewordModel;
//! 
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Initialize the wake word detector
//! let config = KoffeeCandleConfig::default();
//! let mut detector = KoffeeCandle::new(&config)?;
//! 
//! // Load a pre-trained wake word model
//! let model = WakewordModel::load_from_file("wake_word.rpw")?;
//! detector.add_wakeword_model(model)?;
//! 
//! // Process audio samples (16kHz, mono, f32)
//! let audio_samples: Vec<f32> = vec![0.0; 16000]; // 1 second of silence
//! if let Some(detection) = detector.process_samples(&audio_samples) {
//!     println!("Wake word detected: {} (confidence: {:.3})", 
//!              detection.name, detection.score);
//! }
//! # Ok(())
//! # }
//! ```
//! 
//! ### Training a Custom Wake Word Model
//! 
//! ```rust
//! use koffee::trainer;
//! use koffee::ModelType;
//! use std::path::Path;
//! 
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Train a model from a directory of WAV files
//! let input_dir = Path::new("training_data/");
//! let output_path = Path::new("my_wake_word.rpw");
//! 
//! trainer::train_dir(input_dir, output_path, ModelType::Small)?;
//! println!("Model trained successfully!");
//! # Ok(())
//! # }
//! ```
//! 
//! ### TCP Server for Network Detection
//! 
//! ```rust,no_run
//! use koffee::{KoffeeCandle, KoffeeCandleConfig};
//! use koffee::server::run_tcp_server;
//! use std::sync::{Arc, Mutex};
//! 
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Initialize detector
//! let config = KoffeeCandleConfig::default();
//! let detector = Arc::new(Mutex::new(KoffeeCandle::new(&config)?));
//! 
//! // Start TCP server on port 8080
//! run_tcp_server(detector, 8080).await?;
//! # Ok(())
//! # }
//! ```
//! 
//! ## Architecture
//! 
//! The library consists of several key components:
//! 
//! ### Core Detection Engine
//! - **[`KoffeeCandle`]**: Main detection engine with configurable parameters
//! - **[`KoffeeCandleConfig`]**: Configuration system for detection thresholds and behavior
//! - **[`Kfc`]**: Low-level feature extraction and audio processing
//! 
//! ### Model Management
//! - **[`wakewords::WakewordModel`]**: Wake word model loading and management
//! - **[`wakewords::WakewordLoad`]**: Trait for loading models from various sources
//! - **[`wakewords::WakewordSave`]**: Trait for saving trained models
//! 
//! ### Training System
//! - **[`trainer`]**: Complete training pipeline with data splitting and validation
//! - **TTS Integration**: Synthetic sample generation using dia voice synthesis
//! - **Audio Processing**: Advanced preprocessing and augmentation
//! 
//! ### Network Services
//! - **[`server`]**: TCP server with connection management and rate limiting
//! - **Binary Protocol**: Efficient audio streaming and detection result protocol
//! - **Multi-client Support**: Concurrent connection handling with resource management
//! 
//! ### Audio Processing
//! - **[`audio`]**: Comprehensive audio processing pipeline
//! - **Device Management**: Cross-platform audio device enumeration and selection
//! - **Format Support**: Multiple audio formats and sample rates
//! 
//! ## Configuration
//! 
//! ### Detection Configuration
//! 
//! ```rust
//! use koffee::{KoffeeCandleConfig, ModelType};
//! 
//! let config = KoffeeCandleConfig {
//!     detection_threshold: 0.7,        // Higher threshold = fewer false positives
//!     model_type: ModelType::Medium,   // Balance between accuracy and speed
//!     sample_rate: 16000,              // Audio sample rate in Hz
//!     frame_size: 512,                 // Audio frame size for processing
//!     overlap: 0.5,                    // Frame overlap ratio
//!     ..Default::default()
//! };
//! ```
//! 
//! ### Training Configuration
//! 
//! Training supports various model sizes and configurations:
//! 
//! - **Tiny**: Fastest inference, lowest accuracy (~1MB model size)
//! - **Small**: Good balance for mobile/edge devices (~5MB model size)  
//! - **Medium**: Higher accuracy for desktop applications (~20MB model size)
//! - **Large**: Maximum accuracy for server deployments (~80MB model size)
//! 
//! ## Performance Considerations
//! 
//! ### Model Size vs Accuracy Trade-offs
//! 
//! | Model Type | Size | Inference Time | Accuracy | Use Case |
//! |------------|------|----------------|----------|----------|
//! | Tiny       | ~1MB | <1ms          | Good     | IoT/Embedded |
//! | Small      | ~5MB | ~2ms          | Better   | Mobile/Edge |
//! | Medium     | ~20MB| ~5ms          | High     | Desktop |
//! | Large      | ~80MB| ~10ms         | Highest  | Server |
//! 
//! ### Audio Processing Optimization
//! 
//! - Use 16kHz sample rate for optimal balance of quality and performance
//! - Process audio in 512-sample frames with 50% overlap
//! - Enable hardware acceleration when available
//! - Use appropriate buffer sizes for your latency requirements
//! 
//! ## Error Handling
//! 
//! All functions return [`Result`] types with descriptive error messages:
//! 
//! ```rust
//! use koffee::{KoffeeCandle, KoffeeCandleConfig};
//! 
//! match KoffeeCandle::new(&KoffeeCandleConfig::default()) {
//!     Ok(detector) => {
//!         // Use detector
//!     }
//!     Err(e) => {
//!         eprintln!("Failed to initialize detector: {}", e);
//!         // Handle error appropriately
//!     }
//! }
//! ```
//! 
//! ## Thread Safety
//! 
//! The library is designed for multi-threaded usage:
//! 
//! - [`KoffeeCandle`] can be safely shared between threads when wrapped in `Arc<Mutex<>>`
//! - Audio processing is thread-safe and can be parallelized
//! - TCP server supports multiple concurrent connections
//! 
//! ## Examples
//! 
//! See the `examples/` directory for complete usage examples:
//! 
//! - `basic_detection.rs`: Simple wake word detection
//! - `training_pipeline.rs`: Complete model training workflow
//! - `tcp_server.rs`: Network-based detection service
//! - `audio_recording.rs`: Real-time audio capture and processing

#![deny(unsafe_code)] // Allow override for specific performance-critical modules
#![warn(missing_docs)] // Comprehensive API documentation provided

/* ────────────────────────  sub-modules  ─────────────────────────────── */

/// Audio processing and device management
pub mod audio;

/// Builder patterns for configuration objects
pub mod builder;

/// Configuration structures and enums
pub mod config;

/// Library constants and default values
pub mod constants;

/// Koffee Feature Computation - core audio feature extraction
pub mod kfc;

/// TCP server for network-based wake word detection
pub mod server;

/// Model training pipeline and utilities
pub mod trainer;

/// Wake/unwake state detection and management
pub mod wake_unwake;

/// Wake word model definitions and loading/saving
pub mod wakewords;

/* ────────── public façade & re-exports (backward-compat) ─────────────── */
pub use audio::{Endianness, Sample, SampleFormat};
pub use config::{
    AudioFmt, BandPassConfig, DetectorConfig, FiltersConfig, GainNormalizationConfig,
    KoffeeCandleConfig, ScoreMode, VADMode,
};
pub use constants::*;
pub use wake_unwake::{WakeUnwakeConfig, WakeUnwakeDetector, WakeUnwakeState};

pub use wakewords::wakeword_model::{ModelType, ModelWeights};

/* ───────────────────────── crate imports ─────────────────────────────── */
use std::collections::HashMap;

use audio::{AudioEncoder, BandPassFilter, GainNormalizerFilter};
use kfc::{KfcExtractor, VadDetector};
use wakewords::{WakewordDetector, WakewordFile, WakewordLoad, WakewordModel, WakewordRef};

/* ────────────────────────── type helpers ─────────────────────────────── */
/// Original VAD constants: 50 ms window / 500 ms hang-over.
type DefaultVad = VadDetector<50, 500>;

/// Result alias used across the public API.
///
/// All public functions return this type for consistent error handling.
pub type Result<T> = std::result::Result<T, String>;

/* ───────────────────────── main detector ─────────────────────────────── */

/// **KoffeeCandle** – instant-use wake-word detector.
///
/// Build with [`Kfc::new`], feed it PCM (bytes or `f32`) through
/// [`process_bytes`] / [`process_samples`], pick up detections.
pub struct KoffeeCandle {
    /* ---------- config (immutable after ctor) ---------- */
    avg_threshold: f32,
    threshold: f32,
    min_scores: usize,
    eager: bool,
    score_mode: ScoreMode,
    score_ref: f32,
    band_size: u16,

    /* ----------------- front-end helpers ---------------- */
    wav_encoder: AudioEncoder,
    kfc_extractor: KfcExtractor,
    band_pass_filter: Option<BandPassFilter>,
    gain_normalizer_filter: Option<GainNormalizerFilter>,
    vad_detector: Option<DefaultVad>,

    /* ----------------- runtime state -------------------- */
    wakewords: HashMap<String, Box<dyn WakewordDetector>>,
    #[allow(dead_code)]
    rms_level: f32,

    #[cfg(feature = "record")]
    record_path: Option<String>,
    #[cfg(feature = "record")]
    audio_window: Vec<f32>,
    #[cfg(feature = "record")]
    max_audio_samples: usize,
}

/// Historic API compatibility alias for [`KoffeeCandle`].
/// 
/// This type alias is maintained for backward compatibility with existing code
/// that uses the shorter `Kfc` name. New code should prefer using [`KoffeeCandle`] directly.
pub type Kfc = KoffeeCandle;

/* ───────────────── constructor & config ───────────────── */

impl Kfc {
    /// Build a new detector from a [`KoffeeCandleConfig`].
    pub fn new(cfg: &KoffeeCandleConfig) -> Result<Self> {
        /* 1.  re-encode / resample front-end */
        let reenc = AudioEncoder::new(
            &cfg.fmt,
            KFCS_EXTRACTOR_FRAME_LENGTH_MS,
            DETECTOR_INTERNAL_SAMPLE_RATE,
        )
        .map_err(|e| e.to_string())?;

        /* 2.  KFC feature extractor */
        let samples_per_frame = reenc.output_samples();
        let samples_per_shift = (samples_per_frame as f32
            / (KFCS_EXTRACTOR_FRAME_LENGTH_MS as f32 / KFCS_EXTRACTOR_FRAME_SHIFT_MS as f32))
            as usize;

        let kfc_extractor = KfcExtractor::new(
            DETECTOR_INTERNAL_SAMPLE_RATE,
            samples_per_frame,
            samples_per_shift,
            40,
        )
        .map_err(|e| e.to_string())?;

        /* 3.  optional VAD */
        let vad_detector = cfg.detector.vad_mode.map(DefaultVad::new);

        /* 4.  assemble */
        Ok(Self {
            /* constants / thresholds */
            avg_threshold: cfg.detector.avg_threshold,
            threshold: cfg.detector.threshold,
            min_scores: cfg.detector.min_scores,
            eager: cfg.detector.eager,
            score_mode: cfg.detector.score_mode,
            score_ref: cfg.detector.score_ref,
            band_size: cfg.detector.band_size,

            /* helpers */
            wav_encoder: reenc,
            kfc_extractor,
            band_pass_filter: (&cfg.filters.band_pass).into(),
            gain_normalizer_filter: (&cfg.filters.gain_normalizer).into(),
            vad_detector,

            /* state */
            wakewords: HashMap::new(),
            rms_level: 0.0,

            #[cfg(feature = "record")]
            record_path: cfg.detector.record_path.clone(),
            #[cfg(feature = "record")]
            audio_window: Vec::new(),
            #[cfg(feature = "record")]
            max_audio_samples: 0,
        })
    }

    // ------------------------------------------------------------------
    //  Wake-word management
    // ------------------------------------------------------------------

    /// Load / replace a wake-word model.
    ///
    /// * The `name` inside `model.labels` is used as key.
    pub fn add_wakeword_model(&mut self, model: WakewordModel) -> Result<()> {
        let key = model
            .labels
            .iter()
            .find(|s| s.as_str() != NN_NONE_LABEL)
            .cloned()
            .ok_or("model has no non-none label")?;

        let det = model.get_detector(self.score_ref, self.band_size, self.score_mode);
        self.wakewords.insert(key, det);
        Ok(())
    }

    /// Convenience wrapper: load from bytes (CBOR payload).
    pub fn add_wakeword_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        let model = WakewordModel::load_from_buffer(bytes).map_err(|e| e.to_string())?;
        self.add_wakeword_model(model)
    }

    // ------------------------------------------------------------------
    //  Processing API
    // ------------------------------------------------------------------

    /// Push **interleaved signed-16-bit little-endian** PCM bytes.
    pub fn process_bytes(&mut self, pcm: &[u8]) -> Option<KoffeeCandleDetection> {
        let mono = self.wav_encoder.encode_and_resample(pcm).ok()?;
        self.process_samples(&mono)
    }

    /// Push **mono f32** samples.
    pub fn process_samples(&mut self, samples: &[f32]) -> Option<KoffeeCandleDetection> {
        /* optional VAD (skip speech-free windows) */
        if let Some(vad) = &mut self.vad_detector
            && !vad.is_voice(samples)
        {
            return None;
        }

        /* optional filters */
        let filtered: Vec<f32> = if let Some(bp) = &mut self.band_pass_filter {
            bp.process_block(samples)
        } else {
            samples.to_vec()
        };
        let norm: Vec<f32> = if let Some(gn) = &mut self.gain_normalizer_filter {
            gn.process_block(&filtered)
        } else {
            filtered
        };

        /* feature extraction */
        let frames: Vec<Vec<f32>> = self.kfc_extractor.compute(&norm).collect();

        /* per-model detection */
        let mut best: Option<KoffeeCandleDetection> = None;
        for (name, det) in &self.wakewords {
            if let Some(mut hit) = det.run_detection(
                frames.clone(), // clone: cheap (Vec<Vec<f32>>)
                self.avg_threshold,
                self.threshold,
            ) {
                hit.name = name.clone();
                if best.as_ref().is_none_or(|b| hit.score > b.score) {
                    best = Some(hit);
                }
            }
        }
        best
    }

    /// Update runtime thresholds / config without rebuilding the struct.
    pub fn update_config(&mut self, cfg: &DetectorConfig) {
        self.avg_threshold = cfg.avg_threshold;
        self.threshold = cfg.threshold;
        self.min_scores = cfg.min_scores;
        self.eager = cfg.eager;
        self.score_mode = cfg.score_mode;
        self.score_ref = cfg.score_ref;
        self.band_size = cfg.band_size;

        for det in self.wakewords.values_mut() {
            det.update_config(cfg.score_ref, cfg.band_size, cfg.score_mode);
        }
    }
}

/* ─────────────────────── detection struct ───────────────────────────── */

/// A successful wake-word hit returned by [`Kfc::process_*`].
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct KoffeeCandleDetection {
    /// Wake-word name (`labels[..]` in the model).
    pub name: String,
    /// Similarity vs. “next best” label (0-1).
    pub avg_score: f32,
    /// Similarity vs. “none” label (0-1).
    pub score: f32,
    /// Per-label raw probabilities.
    pub scores: HashMap<String, f32>,
    /// Optional counter for downstream aggregation.
    pub counter: usize,
    /// Post-gain level suggested by AGC.
    pub gain: f32,
}

/* ─────────────────────────── helpers ────────────────────────────────── */

#[allow(dead_code)]
#[inline(always)]
const fn out_shifts() -> usize {
    KFCS_EXTRACTOR_OUT_SHIFTS
}
