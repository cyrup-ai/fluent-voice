//! Audio helpers: loudness normalisation + zero-copy WAV writer.
//! Only std + bytemuck – no allocator in the inner loops.

use candle_core::{Result as CandleResult, Tensor};

/// Reference: https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/data/audio_utils.py
pub fn normalize_loudness(
    wav: &Tensor,
    sample_rate: u32,
    loudness_compressor: bool,
) -> CandleResult<Tensor> {
    // rms in one fused kernel
    let energy = wav.sqr()?.mean_all()?.sqrt()?.to_vec0::<f32>()?;
    if energy < 2e-3 {
        return Ok(wav.clone());
    }

    // Delegate to shared helper.
    crate::audio::normalize_loudness(wav, sample_rate, loudness_compressor)
        .map_err(|e| candle_core::Error::Msg(format!("Loudness normalization failed: {e}")))
}

use std::io::{BufWriter, Write};

/// Write mono 16-bit PCM as WAVE. Supports any sample-rate, no heap copies.
pub fn write_pcm_as_wav<W: Write>(mut w: W, pcm: &[i16], sample_rate: u32) -> std::io::Result<()> {
    const RIFF_HEADER: &[u8; 4] = b"RIFF";
    const WAVE_HEADER: &[u8; 4] = b"WAVE";
    const FMT_HEADER: &[u8; 4] = b"fmt ";
    const DATA_HEADER: &[u8; 4] = b"data";

    let bytes_len = (pcm.len() * 2) as u32;
    // ---------- RIFF ----------
    w.write_all(RIFF_HEADER)?;
    w.write_all(&(36 + bytes_len).to_le_bytes())?;
    w.write_all(WAVE_HEADER)?;
    // ---------- fmt  ----------
    w.write_all(FMT_HEADER)?;
    w.write_all(&16u32.to_le_bytes())?; // block size
    w.write_all(&1u16.to_le_bytes())?; // PCM
    w.write_all(&1u16.to_le_bytes())?; // mono
    w.write_all(&sample_rate.to_le_bytes())?;
    w.write_all(&(sample_rate * 2).to_le_bytes())?; // byte-rate
    w.write_all(&2u16.to_le_bytes())?; // frame-align
    w.write_all(&16u16.to_le_bytes())?; // bits/sample
    // ---------- data ----------
    w.write_all(DATA_HEADER)?;
    w.write_all(&bytes_len.to_le_bytes())?;
    // payload – 0-copy cast
    w.write_all(bytemuck::cast_slice(pcm))?;
    Ok(())
}

/// Convenience – always uses a `BufWriter` around a `File`.
pub fn write_wav_file(path: &std::path::Path, samples: &[i16], sr: u32) -> std::io::Result<()> {
    let f = std::fs::File::create(path)?;
    let mut bw = BufWriter::with_capacity(1 << 16, f);
    write_pcm_as_wav(&mut bw, samples, sr)?;
    bw.flush()
}
