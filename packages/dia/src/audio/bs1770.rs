//! A **tiny, no-std-friendly** ITU-R BS.1770-4 loudness meter (mono only).
//!
//! This implementation follows the reference K-weighting filter and momentary/
//! integrated loudness specification but is deliberately lightweight: no
//! dynamic allocations inside the real-time path and no dependencies outside
//! `core`/`alloc`.

use core::f64::consts::PI;

/// Length of the *momentary* window in seconds (400 ms, ITU spec).
const MOMENTARY_S: f64 = 0.400;
/// Length of the *short-term* window in seconds (3 s, ITU spec – we track it
/// for completeness even though the project only needs integrated &
/// momentary metrics).
const SHORTTERM_S: f64 = 3.0;

// ─────────────────────────────────────── Biquad ────────────────────────────

/// Advanced direct-form I biquad filter.
#[derive(Clone, Copy)]
struct Biquad {
    // feed-forward
    b0: f64,
    b1: f64,
    b2: f64,
    // feed-back (a0 is assumed 1.0)
    a1: f64,
    a2: f64,
    // state (DF-I)
    z1: f64,
    z2: f64,
}

impl Biquad {
    /// Design the cascaded K-weighting filter (high-pass + high-shelf) for the
    /// given sample-rate using a bilinear transform.  The coefficients come
    /// from ITU-R BS.1770-4 Annex 2.
    fn k_weighting(fs: u32) -> (Self, Self) {
        let fs = fs as f64;

        // --- High-pass (first stage) --------------------------------------
        let f0 = 38.135_470_876_024_44; // Hz
        let q = 0.500_327_05;
        let k = (PI * f0 / fs).tan();
        let norm = 1.0 / (1.0 + k / q + k * k);

        let hp = Self {
            b0: 1.0 * norm,
            b1: -2.0 * norm,
            b2: 1.0 * norm,
            a1: 2.0 * (k * k - 1.0) * norm,
            a2: (1.0 - k / q + k * k) * norm,
            z1: 0.0,
            z2: 0.0,
        };

        // --- High-shelf (second stage) ------------------------------------
        let f0 = 1_681.974_450_955_533; // Hz (grouped for readability)
        let gain_db = 3.999_843_853_97; // dB
        let q = 0.707_175_236; // 1/√2

        let k = (PI * f0 / fs).tan();
        let v0 = 10.0_f64.powf(gain_db / 20.0);
        let root = v0.sqrt();
        let norm = 1.0 / (1.0 + k / q + k * k);

        let shelf = Self {
            b0: v0 + root * k / q + k * k,
            b1: 2.0 * (k * k - v0),
            b2: v0 - root * k / q + k * k,
            a1: 2.0 * (k * k - 1.0),
            a2: 1.0 - k / q + k * k,
            z1: 0.0,
            z2: 0.0,
        };

        // Normalize by the common denominator (a0).
        let a0 = 1.0 + k / q + k * k;
        let hp = hp.scale(norm);
        let shelf = shelf.scale(norm / a0);

        (hp, shelf)
    }

    #[inline(always)]
    fn process(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }

    fn scale(mut self, s: f64) -> Self {
        self.b0 *= s;
        self.b1 *= s;
        self.b2 *= s;
        self.a1 *= s;
        self.a2 *= s;
        self
    }
}

// ───────────────────────────── Sliding-window RMS helper ───────────────────

struct RmsWindow {
    buf: alloc::vec::Vec<f64>,
    sumsq: f64,
    idx: usize,
    filled: bool,
}

impl RmsWindow {
    fn new(len: usize) -> Self {
        Self {
            buf: alloc::vec![0.0; len],
            sumsq: 0.0,
            idx: 0,
            filled: false,
        }
    }

    #[inline(always)]
    fn push(&mut self, x: f64) {
        let prev = self.buf[self.idx];
        self.buf[self.idx] = x;
        self.sumsq += x * x - prev * prev;

        self.idx += 1;
        if self.idx == self.buf.len() {
            self.idx = 0;
            self.filled = true;
        }
    }

    #[inline(always)]
    fn rms(&self) -> f64 {
        let n = if self.filled {
            self.buf.len()
        } else {
            self.idx
        };
        if n == 0 {
            0.0
        } else {
            (self.sumsq / n as f64).sqrt()
        }
    }
}

// ──────────────────────────────── Meter struct ─────────────────────────────

/// Real-time ITU-R BS.1770-4 loudness meter (mono only).
pub struct Bs1770Meter {
    #[allow(dead_code)]
    fs: u32,
    hp: Biquad,
    shelf: Biquad,

    momentary: RmsWindow,
    shortterm: RmsWindow,

    // For (simplified) integrated loudness – we accumulate *after* the
    // K-weighting filter.  ITU gating is not fully replicated as it is not
    // critical for interactive TTS preview; silence is handled by the caller
    // with a cheap RMS gate.
    int_sum_sq: f64,
    int_samples: u64,
}

impl Bs1770Meter {
    /// Create a new meter for the given sample-rate.
    pub fn new(fs: u32) -> Self {
        let (hp, shelf) = Biquad::k_weighting(fs);
        let mom = (fs as f64 * MOMENTARY_S).ceil() as usize;
        let st = (fs as f64 * SHORTTERM_S).ceil() as usize;

        Self {
            fs,
            hp,
            shelf,
            momentary: RmsWindow::new(mom),
            shortterm: RmsWindow::new(st),
            int_sum_sq: 0.0,
            int_samples: 0,
        }
    }

    /// Feed an arbitrary-length block of *mono* samples (-1.0…1.0).
    pub fn push_block(&mut self, block: &[f32]) {
        for &s in block {
            let s = s as f64;
            let w = {
                let w1 = self.hp.process(s);
                self.shelf.process(w1)
            };

            self.momentary.push(w);
            self.shortterm.push(w);

            self.int_sum_sq += w * w;
            self.int_samples += 1;
        }
    }

    /// Convert RMS value to LUFS (offset −0.691 dB per spec).
    #[inline(always)]
    fn rms_to_lufs(rms: f64) -> f64 {
        -0.691 + 10.0 * (rms.max(1e-12)).log10()
    }

    /// 400 ms window loudness.
    pub fn momentary_lufs(&self) -> f64 {
        Self::rms_to_lufs(self.momentary.rms())
    }

    /// 3 s window loudness.
    pub fn short_term_lufs(&self) -> f64 {
        Self::rms_to_lufs(self.shortterm.rms())
    }

    /// Very advanced integrated measurement – average over the entire signal.
    /// This is *not* fully ITU-compliant (no gating) but adequate for
    /// real-time normalization where long stretches of silence are excluded
    /// by an upfront RMS gate.
    pub fn integrated_lufs(&self) -> Option<f64> {
        if self.int_samples == 0 {
            return None;
        }
        let rms = (self.int_sum_sq / self.int_samples as f64).sqrt();
        Some(Self::rms_to_lufs(rms))
    }
}

// We rely on `alloc` for the sliding-window buffer – required when targeting
// `no_std` (WASM).  On std targets the global allocator is already available.
extern crate alloc;
