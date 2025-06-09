//! GainNormalizerFilter
//!
//! Purpose
//! -------
//! Dynamically scale the incoming audio so that its RMS power tends toward a
//! reference level.  This implementation uses a **fixed-length ring-buffer**
//! (no `Vec::drain` cost) and keeps a running sum so the per-frame gain
//! calculation is *O(1)* instead of *O(N)* over the window.
//
//! Public API – unchanged
//! ----------------------
//! * `new`
//! * `filter`
//! * `set_rms_level_ref`
//! * `get_rms_level_ref`
//! * `get_rms_level`

use crate::config::GainNormalizationConfig;

/// RMS-based automatic gain control.
pub struct GainNormalizerFilter {
    /* immutable settings */
    min_gain: f32,
    max_gain: f32,
    fixed_rms_level: bool,

    /* reference level (mutable when not fixed) */
    rms_level_ref: f32,
    rms_level_sqrt: f32,

    /* sliding-window state */
    window_size: usize,
    window: Vec<f32>,
    write_pos: usize,
    running_sum: f32,
}

impl GainNormalizerFilter {
    /// Create a new filter.
    pub fn new(min_gain: f32, max_gain: f32, fixed_rms_level: Option<f32>) -> Self {
        // Use NaN as the default when no fixed RMS level is provided
        let rms_ref = fixed_rms_level.unwrap_or(f32::NAN);
        let rms_sqrt = fixed_rms_level.map_or(f32::NAN, f32::sqrt);

        Self {
            min_gain,
            max_gain,
            fixed_rms_level: fixed_rms_level.is_some(),
            rms_level_ref: rms_ref,
            rms_level_sqrt: rms_sqrt,
            window_size: 1,    // updated later by set_rms_level_ref()
            window: vec![0.0], // dummy
            write_pos: 0,
            running_sum: 0.0,
        }
    }

    /* ─────────── public helpers ─────────── */

    #[inline]
    pub fn get_rms_level_ref(&self) -> f32 {
        self.rms_level_ref
    }

    /// Update the target RMS level **and** analysis-window length.
    pub fn set_rms_level_ref(&mut self, rms_level: f32, window_size: usize) {
        if !self.fixed_rms_level {
            self.rms_level_ref = rms_level;
            self.rms_level_sqrt = rms_level.sqrt();
        }
        let w = window_size.max(1);
        if w != self.window_size {
            self.window_size = w;
            self.window = vec![0.0; w];
            self.write_pos = 0;
            self.running_sum = 0.0;
        }
    }

    /// Per-frame gain application.  Returns the *applied* gain (==1.0 if none).
    pub fn filter(&mut self, signal: &mut [f32], frame_rms: f32) -> f32 {
        if self.rms_level_ref.is_nan() || frame_rms == 0.0 {
            return 1.0;
        }

        /* ---- update sliding window ---- */
        let old = self.window[self.write_pos];
        self.window[self.write_pos] = frame_rms;
        self.write_pos = (self.write_pos + 1) % self.window_size;
        self.running_sum += frame_rms - old;

        let mean_rms = self.running_sum / self.window_size as f32;

        /* ---- compute & clamp gain ---- */
        let mut gain = self.rms_level_sqrt / mean_rms.sqrt();
        gain = ((gain * 10.0).round() / 10.0).clamp(self.min_gain, self.max_gain);

        /* ---- apply if meaningful ---- */
        if (gain - 1.0).abs() > f32::EPSILON {
            for sample in signal {
                *sample = (*sample * gain).clamp(-1.0, 1.0);
            }
        }
        gain
    }

    /* ---- static helper ---- */

    /// Plain RMS of an f32 slice.
    #[inline]
    pub fn get_rms_level(signal: &[f32]) -> f32 {
        let sum_sq: f32 = signal.iter().map(|s| s * s).sum();
        (sum_sq / signal.len() as f32).sqrt()
    }
}

/* ───────── trait adapter ───────── */

impl From<&GainNormalizationConfig> for Option<GainNormalizerFilter> {
    fn from(cfg: &GainNormalizationConfig) -> Self {
        if cfg.enabled {
            Some(GainNormalizerFilter::new(cfg.min_gain, cfg.max_gain, cfg.gain_ref))
        } else {
            None
        }
    }
}

/* ───────── tests (existing integration tests still compile) ───────── */
