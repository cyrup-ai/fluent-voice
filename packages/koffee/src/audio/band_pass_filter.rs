//! Simple one-pole band-pass IIR used by the gain/VAD front-end.

use crate::{BandPassConfig, constants::DETECTOR_INTERNAL_SAMPLE_RATE};
use core::f32::consts::PI;

/// Second-order biquad configured as **band-pass** (Skirt-Gain = 0 dB).
pub struct BandPassFilter {
    /* coefficients */
    a0: f32,
    a1: f32,
    a2: f32,
    b1: f32,
    b2: f32,
    /* 2-sample delay-line (x[n-k], y[n-k]) */
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl BandPassFilter {
    /* ─────────────────── public API ─────────────────── */

    /// Process **in-place** (mono) buffer.
    #[inline]
    pub fn filter(&mut self, signal: &mut [f32]) {
        let Self {
            a0,
            a1,
            a2,
            b1,
            b2,
            x1,
            x2,
            y1,
            y2,
        } = self;

        // Use local mutable copies of state variables
        let a0_val = *a0;
        let a1_val = *a1;
        let a2_val = *a2;
        let b1_val = *b1;
        let b2_val = *b2;
        let mut x1_val = *x1;
        let mut x2_val = *x2;
        let mut y1_val = *y1;
        let mut y2_val = *y2;

        for s in signal {
            let x0 = *s;
            let y0 = a0_val * x0 + a1_val * x1_val + a2_val * x2_val /* feed-forward */
                     - b1_val * y1_val - b2_val * y2_val; /* feed-back */
            x2_val = x1_val;
            x1_val = x0;
            y2_val = y1_val;
            y1_val = y0;
            *s = y0;
        }

        // Update the original struct state at the end
        *x1 = x1_val;
        *x2 = x2_val;
        *y1 = y1_val;
        *y2 = y2_val;
    }

    /// Create a new filter with the given low / high cut-offs (Hz).
    pub fn new(sample_rate: f32, low_cut: f32, high_cut: f32) -> Self {
        Self::design(sample_rate, low_cut, high_cut)
    }

    /// Process a block of samples in-place
    pub fn process(&mut self, samples: &mut [f32]) {
        for sample in samples.iter_mut() {
            *sample = self.process_sample(*sample);
        }
    }

    /// Re-compute coefficients (keeps delay-line).
    pub fn update(&mut self, sample_rate: f32, low_cut: f32, high_cut: f32) {
        *self = Self {
            ..Self::design(sample_rate, low_cut, high_cut)
        };
    }

    /// Zero internal delay-line.
    #[inline]
    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }

    /* ────────────────── helpers ────────────────── */

    fn design(sr: f32, low: f32, high: f32) -> Self {
        // Cook-book RBJ band-pass (constant skirt gain, peak gain = Q)
        let w0l = 2.0 * PI * low / sr;
        let w0h = 2.0 * PI * high / sr;

        let (sin_l, cos_l) = w0l.sin_cos();
        let (sin_h, cos_h) = w0h.sin_cos();

        // Quality factors picked so the two single-pole HP/LP cascades
        // meet at –3 dB ≈ sqrt(½), giving good speech pass-band (≈ 300-3 kHz
        // when using the defaults 80-400 Hz *upstream*).
        let alpha_l = sin_l / 2.0;
        let alpha_h = sin_h / 2.0;

        let a0_inv = 1.0 / (1.0 + alpha_h - alpha_l);
        let a0 = a0_inv;
        let a1 = -2.0 * cos_l * a0_inv;
        let a2 = (1.0 - alpha_h - alpha_l) * a0_inv;
        let b1 = -2.0 * cos_h * a0_inv;
        let b2 = (1.0 - alpha_h + alpha_l) * a0_inv;

        Self {
            a0,
            a1,
            a2,
            b1,
            b2,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    /// Process a block of samples and return filtered results
    pub fn process_block(&mut self, samples: &[f32]) -> Vec<f32> {
        samples
            .iter()
            .map(|&sample| self.process_sample(sample))
            .collect()
    }

    /// Process a single sample through the filter
    pub fn process_sample(&mut self, input: f32) -> f32 {
        let output = self.a0 * input + self.a1 * self.x1 + self.a2 * self.x2
            - self.b1 * self.y1
            - self.b2 * self.y2;

        // Update state variables
        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = output;

        output
    }
}

/* ───────── automatic From<BandPassConfig> wiring ───────── */

impl From<&BandPassConfig> for Option<BandPassFilter> {
    fn from(cfg: &BandPassConfig) -> Self {
        cfg.enabled.then(|| {
            BandPassFilter::new(
                DETECTOR_INTERNAL_SAMPLE_RATE as f32,
                cfg.low_cutoff,
                cfg.high_cutoff,
            )
        })
    }
}
