#![allow(clippy::needless_range_loop)]
//! Lightweight log-mel feature extractor for 16-kHz speech.
//!
//! The design goal is **speed <5 ms / 10 ms frame** and no heap allocations in
//! the hot path once the ring-buffer is initialised.
//!
//! * 16-kHz, mono PCM `f32` samples.
//! * 10-ms hop (160 samples) ⇒ `FRAME`.
//! * 25-ms analysis window (400 samples, Hann).
//! * 49 frames (~0.5 s) per inference window ⇒ 40 × 49 = `FEATURE_DIM`.
//! * 40 equal-width pseudo-Mel bands formed by pooling the power spectrum.
//! * Output: log-energy per band in row-major `[time, mel]` order.
//!
//! **Dependencies**
//! ```toml
//! [dependencies]
//! rustfft = "0.10"
//! once_cell = "1"
//! ```

use once_cell::sync::Lazy;
use rustfft::{Fft, FftPlanner, num_complex::Complex32};

/// 16-kHz ⇒ 10-ms hop = 160 samples.
pub const FRAME: usize = 160;
/// 25-ms analysis window.
pub const WINDOW: usize = 400;
/// Number of frames that make up one inference slice (≈ 0.49 s).
pub const N_FRAMES: usize = 49;
/// Number of coarse Mel bands.
pub const N_MELS: usize = 40;
/// Feature vector length (`N_MELS × N_FRAMES`).
pub const FEATURE_DIM: usize = N_MELS * N_FRAMES;
/// Size of ring buffer used by the detector (`2 ×` inference slice for overlap).
pub const RING_SIZE: usize = FRAME * N_FRAMES * 2;

/// Pre-computed Hann window for `WINDOW` samples.
static HANN: Lazy<Vec<f32>> = Lazy::new(|| {
    (0..WINDOW)
        .map(|i| {
            let phase = (i as f32) * core::f32::consts::PI * 2.0 / (WINDOW as f32);
            0.5 * (1.0 - phase.cos())
        })
        .collect()
});

/// Extract a **[N_FRAMES × N_MELS]** log-Mel patch ending at `pos` in `ring`.
///
/// * `ring` — circular audio buffer of length `RING_SIZE`.
/// * `pos`  — write-cursor position _after_ the most-recent frame.
///
/// Returns a fixed-size array suitable for a fully-connected wake-word model.
pub fn extract(ring: &[f32], pos: usize) -> [f32; FEATURE_DIM] {
    debug_assert_eq!(ring.len(), RING_SIZE);
    debug_assert!(pos < RING_SIZE);

    // FFT setup (lazy-initialised, shared across calls).
    static PLANNER: Lazy<FftPlanner<f32>> = Lazy::new(|| FftPlanner::<f32>::new());
    static FFT: Lazy<std::sync::Arc<dyn Fft<f32> + Sync + Send>> =
        Lazy::new(|| PLANNER.plan_fft_forward(WINDOW));

    let mut scratch = vec![Complex32::default(); WINDOW];
    let mut out = [0f32; FEATURE_DIM];

    // Each frame is aligned on 10-ms (FRAME) boundaries counting *backwards*
    // from `pos`.
    for t in 0..N_FRAMES {
        // Index of the *first* sample in this analysis window.
        let mut idx = pos as isize - ((N_FRAMES - t) * FRAME) as isize - WINDOW as isize;
        // Fold negative indices around the ring.
        idx = (idx % RING_SIZE as isize + RING_SIZE as isize) % RING_SIZE as isize;

        // Copy + window.
        for i in 0..WINDOW {
            let sample = ring[(idx as usize + i) % RING_SIZE];
            scratch[i].re = sample * HANN[i];
            scratch[i].im = 0.0;
        }

        // FFT → power spectrum.
        FFT.process(&mut scratch);
        let half = WINDOW / 2 + 1;
        let mut powers = [0f32; N_MELS];
        let bin_size = half / N_MELS;
        for (bin, p) in scratch.iter().take(half).enumerate() {
            let mel = core::cmp::min(bin / bin_size, N_MELS - 1);
            powers[mel] += p.norm_sqr();
        }

        // Log compression + write to output row-major (time fastest).
        for (m, &p) in powers.iter().enumerate() {
            out[t * N_MELS + m] = (p + 1e-6).ln();
        }
    }

    out
}
