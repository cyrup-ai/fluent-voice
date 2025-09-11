//! `audio/mod.rs` – public façade for all low-level audio helpers
//!
//! After `use crate::audio::*` you get:
//!   * PCM decoding  → `pcm_decode()`
//!   * Generic mono resampling  → `resample_mono()` / `to_24k_mono()`
//!   * BS.1770 loudness utils  → `normalize_loudness()`,
//!     `EnhancedNormalizer`, `LoudnessPreset`
//!   * Tiny real-time loudness meter  → `Bs1770Meter`
//!   * WAV export with built-in BS.1770 normalisation  → `write_pcm_as_wav()`
//!

use crate::{DType, Tensor};

// **There is exactly one implementation of every helper**; if you add a new
// resampler or meter place it in this directory and re-export it here.

use anyhow::Result;

// sub-modules --------------------------------------------------------------

pub mod bs1770;
pub mod channel_delay;
pub mod enhanced_normalizer;
pub mod pcm;
pub mod play;
pub mod resample;
pub mod wav;

// public re-exports --------------------------------------------------------

// NOTE: Avoid blanket wildcard exports; only expose the high-level helpers that
// are part of the public API surface.  Low-level implementation details stay
// inside their respective sub-modules.

pub use bs1770::Bs1770Meter;
pub use channel_delay::{DELAY_PATTERN, delayed_view, undelayed_view};
pub use enhanced_normalizer::{EnhancedNormalizer, LoudnessPreset};
pub use play::play_pcm;
pub use wav::write_pcm_as_wav;

/// Fixed internal working rate for Dia-Voice (24 kHz mono).
pub const SAMPLE_RATE: usize = 24_000;

// -------------------------------------------------------------------------
// Thin convenience wrappers
// -------------------------------------------------------------------------

/// Decode **any** audio file supported by Symphonia to _mono_ `Vec<f32>` plus
/// its original sample-rate.
#[inline]
pub fn pcm_decode(path: &str) -> Result<(Vec<f32>, u32)> {
    pcm::load(path)
}

/// Resample an arbitrary mono buffer to a new sample-rate.
#[cfg(any(
    feature = "microphone",
    feature = "encodec",
    feature = "mimi",
    feature = "snac"
))]
#[inline]
pub fn resample_mono(input: &[f32], sr_in: u32, sr_out: u32) -> Result<Vec<f32>> {
    resample::resample_mono(input, sr_in, sr_out)
}

#[cfg(not(any(
    feature = "microphone",
    feature = "encodec",
    feature = "mimi",
    feature = "snac"
)))]
#[inline]
pub fn resample_mono(_input: &[f32], _sr_in: u32, _sr_out: u32) -> Result<Vec<f32>> {
    Err(anyhow::anyhow!("resample_mono requires audio features"))
}

/// Convenience: down-mix (if necessary) **and** resample to 24 kHz mono.
#[cfg(any(
    feature = "microphone",
    feature = "encodec",
    feature = "mimi",
    feature = "snac"
))]
#[inline]
pub fn to_24k_mono(pcm: Vec<f32>, sr_in: u32, channels: usize) -> Result<Vec<f32>> {
    resample::to_24k_mono(pcm, sr_in, channels)
}

#[cfg(not(any(
    feature = "microphone",
    feature = "encodec",
    feature = "mimi",
    feature = "snac"
)))]
#[inline]
pub fn to_24k_mono(_pcm: Vec<f32>, _sr_in: u32, _channels: usize) -> Result<Vec<f32>> {
    Err(anyhow::anyhow!("to_24k_mono requires audio features"))
}

/// One-shot BS.1770 loudness normalisation helper – converts a `Tensor`
/// (any length) in-place and returns it as a fresh tensor.  Used by the CLI
/// exporter; real-time paths embed an `EnhancedNormalizer` instance instead.
pub fn normalize_loudness(wav: &Tensor, sr: u32, use_compression: bool) -> Result<Tensor> {
    // flatten to Vec<f32>
    let mut pcm = wav.to_vec1::<f32>()?;

    // process block-wise
    let mut norm = EnhancedNormalizer::new(sr as usize);
    norm.set_preset(LoudnessPreset::Voice);
    norm.set_compression(use_compression);

    const BLOCK: usize = 2048;
    // Use explicit slice operations to avoid sized_chunks trait conflicts
    let mut offset = 0;
    while offset < pcm.len() {
        let end = (offset + BLOCK).min(pcm.len());
        norm.process(&mut pcm[offset..end]);
        offset = end;
    }

    // rebuild tensor with the same shape / device / dtype
    let tensor = Tensor::from_vec(pcm, wav.dims(), wav.device())?.to_dtype(match wav.dtype() {
        DType::F16 | DType::BF16 | DType::F32 => wav.dtype(),
        _ => DType::F32,
    })?;

    Ok(tensor)
}
