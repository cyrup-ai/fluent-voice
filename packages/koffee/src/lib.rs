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
#![deny(unsafe_code)] // Allow override for specific performance-critical modules
#![allow(missing_docs)] // TODO: Add comprehensive docs after compilation fixes

/* ────────────────────────  sub-modules  ─────────────────────────────── */
pub mod audio;
pub mod builder;
pub mod config;
pub mod constants;
pub mod kfc;
pub mod server;

pub mod trainer;
pub mod wake_unwake;
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

/* public alias kept from historic API */
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
