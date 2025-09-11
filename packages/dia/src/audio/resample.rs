//! src/audio/resample.rs
//!
//! Single-point audio resampler used by Dia-Voice.  Host-side only
//! (rubato is pure-CPU SIMD); callers feed/consume plain `Vec<f32>`.
//
//! Maintainers: there must be **no other resample helpers** in the tree.

use anyhow::Result;

#[cfg(not(any(
    feature = "microphone",
    feature = "encodec",
    feature = "mimi",
    feature = "snac"
)))]
pub fn to_24k_mono(_pcm: Vec<f32>, _sr_in: u32, _channels: usize) -> Result<Vec<f32>> {
    Err(anyhow::anyhow!("to_24k_mono requires audio features"))
}

#[cfg(not(any(
    feature = "microphone",
    feature = "encodec",
    feature = "mimi",
    feature = "snac"
)))]
pub fn resample_mono(_input: &[f32], _sr_in: u32, _sr_out: u32) -> Result<Vec<f32>> {
    Err(anyhow::anyhow!("resample_mono requires audio features"))
}
#[cfg(any(
    feature = "microphone",
    feature = "encodec",
    feature = "mimi",
    feature = "snac"
))]
use rubato::{FftFixedIn, Resampler};

/// Resample arbitrary PCM to **24 kHz mono**.
///
/// * `pcm`        – interleaved samples (length = frames × channels)  
/// * `sr_in`      – original sample-rate (Hz)  
/// * `channels`   – number of interleaved channels in `pcm`  
///
/// Returns a `Vec<f32>` containing mono PCM @ 24 000 Hz.
#[cfg(any(
    feature = "microphone",
    feature = "encodec",
    feature = "mimi",
    feature = "snac"
))]
pub fn to_24k_mono(mut pcm: Vec<f32>, sr_in: u32, channels: usize) -> Result<Vec<f32>> {
    // ────────────────────────────────────────────────────────────────────────
    // 1. Channel down-mix (advanced L+R average – good enough for TTS prompts)
    // ────────────────────────────────────────────────────────────────────────
    if channels == 2 {
        let mut mono = Vec::with_capacity(pcm.len() / 2);
        for frame in pcm.chunks_exact(2) {
            mono.push((frame[0] + frame[1]) * 0.5);
        }
        pcm = mono;
    } else if channels != 1 {
        anyhow::bail!("unsupported channel count: {channels}");
    }

    // ────────────────────────────────────────────────────────────────────────
    // 2. Early-out when nothing to do
    // ────────────────────────────────────────────────────────────────────────
    if sr_in == 24_000 {
        return Ok(pcm); // already 24 kHz mono
    }

    // ────────────────────────────────────────────────────────────────────────
    // 3. Chunked FFT resampling with FftFixedIn for flexibility
    // ────────────────────────────────────────────────────────────────────────
    // Using FftFixedIn allows variable output sizes, avoiding buffer size issues
    const CHUNK: usize = 1024; // Larger chunk for better compatibility
    const SUB_CHUNKS: usize = 2; // Number of sub-chunks for processing
    let mut resampler = FftFixedIn::<f32>::new(sr_in as usize, 24_000, CHUNK, SUB_CHUNKS, 1)?;

    // Calculate expected output capacity
    let expected_len = (pcm.len() as f64 * 24_000.0 / sr_in as f64).ceil() as usize;
    let mut out = Vec::with_capacity(expected_len + CHUNK);

    // Process in chunks
    let mut pos = 0;
    while pos < pcm.len() {
        let end = (pos + CHUNK).min(pcm.len());
        let chunk_len = end - pos;

        // Create input buffer
        let mut input_chunk = vec![0.0; CHUNK];
        input_chunk[..chunk_len].copy_from_slice(&pcm[pos..end]);

        // Process this chunk
        let block = vec![input_chunk];
        let frames = resampler.process(&block, None)?;
        out.extend_from_slice(&frames[0]);

        pos += chunk_len;

        // For the last partial chunk, we're done
        if chunk_len < CHUNK {
            break;
        }
    }

    Ok(out)
}

/// Low-level helper used by unit-tests or by callers that need an
/// *arbitrary* output rate (still mono).
#[cfg(any(
    feature = "microphone",
    feature = "encodec",
    feature = "mimi",
    feature = "snac"
))]
pub fn resample_mono(input: &[f32], sr_in: u32, sr_out: u32) -> Result<Vec<f32>> {
    if sr_in == sr_out {
        return Ok(input.to_vec());
    }

    // Use FftFixedIn for flexibility
    const CHUNK: usize = 1024;
    const SUB_CHUNKS: usize = 2; // Number of sub-chunks for processing
    let mut resampler =
        FftFixedIn::<f32>::new(sr_in as usize, sr_out as usize, CHUNK, SUB_CHUNKS, 1)?;

    // Calculate expected output capacity
    let expected_len = (input.len() as f64 * sr_out as f64 / sr_in as f64).ceil() as usize;
    let mut out = Vec::with_capacity(expected_len + CHUNK);

    // Process in chunks
    let mut pos = 0;
    while pos < input.len() {
        let end = (pos + CHUNK).min(input.len());
        let chunk_len = end - pos;

        // Create input buffer
        let mut input_chunk = vec![0.0; CHUNK];
        input_chunk[..chunk_len].copy_from_slice(&input[pos..end]);

        // Process this chunk
        let block = vec![input_chunk];
        let frames = resampler.process(&block, None)?;
        out.extend_from_slice(&frames[0]);

        pos += chunk_len;

        // For the last partial chunk, we're done
        if chunk_len < CHUNK {
            break;
        }
    }

    Ok(out)
}
