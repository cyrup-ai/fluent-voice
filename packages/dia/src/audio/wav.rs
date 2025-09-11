//! WAV I/O helpers – always pass audio through the BS.1770-4 normaliser
//! before the final file is written.

use std::io::{Seek, SeekFrom, Write};

use anyhow::{Context, Result};
use byteorder::{LittleEndian, WriteBytesExt};

use crate::audio::enhanced_normalizer::{EnhancedNormalizer, LoudnessPreset};

/// Write `pcm` (32-bit float, -1.0‥+1.0) to `w` as a 16-bit-PCM WAV, **after**
/// loudness-normalising to the requested preset (defaults to “Voice chat”).
///
/// * `pcm` is **mono** – resample beforehand if you need stereo/48 kHz, etc.
/// * `sample_rate_hz` is the *current* rate of `pcm`, not the target rate.
///   The normaliser runs at that native rate; the file is written unchanged.
pub fn write_pcm_as_wav<W: Write + Seek>(
    mut w: W,
    mut pcm: Vec<f32>,
    sample_rate_hz: u32,
    preset: Option<LoudnessPreset>,
) -> Result<()> {
    //----------------------------------------------------------------------
    // 1.  Loudness-normalise in-place  (BS.1770-4, true-peak limiter)
    //----------------------------------------------------------------------
    let preset = preset.unwrap_or(LoudnessPreset::Voice);
    let mut norm = EnhancedNormalizer::new(sample_rate_hz as usize);
    norm.set_preset(preset);
    norm.set_compression(true); // engage peak-limiter

    // Block size: 2048 is plenty for offline export.
    const BLOCK: usize = 2048;
    for chunk in pcm.chunks_mut(BLOCK) {
        norm.process(chunk);
    }

    //----------------------------------------------------------------------
    // 2.  Convert f32 → i16  (clip -1.0‥+1.0, scale to ±32767)
    //----------------------------------------------------------------------
    let mut pcm_i16 = Vec::<i16>::with_capacity(pcm.len());
    for &s in &pcm {
        let clamped = s.clamp(-1.0, 1.0);
        pcm_i16.push((clamped * 32767.0) as i16);
    }

    //----------------------------------------------------------------------
    // 3.  Write RIFF/WAVE container
    //----------------------------------------------------------------------
    // RIFF header - calculate total file size upfront for proper header
    w.write_all(b"RIFF")?;
    let total_samples = pcm_i16.len() as u32;
    let data_size = total_samples * 2; // 16-bit samples = 2 bytes each
    let chunk_size = 36 + data_size; // Header size (44 bytes) - 8 bytes for RIFF header
    w.write_u32::<LittleEndian>(chunk_size)?;
    w.write_all(b"WAVE")?;

    // fmt  sub-chunk -------------------------------------------------------
    w.write_all(b"fmt ")?;
    w.write_u32::<LittleEndian>(16)?; // PCM header size
    w.write_u16::<LittleEndian>(1)?; // PCM = 1
    w.write_u16::<LittleEndian>(1)?; // mono
    w.write_u32::<LittleEndian>(sample_rate_hz)?; // sample rate
    let byte_rate = sample_rate_hz * 2; // samp/sec * bytes/frame
    w.write_u32::<LittleEndian>(byte_rate)?;
    let block_align = 2; // 1 ch * 16 bit / 8
    w.write_u16::<LittleEndian>(block_align)?;
    w.write_u16::<LittleEndian>(16)?; // bits per sample

    // data sub-chunk -------------------------------------------------------
    w.write_all(b"data")?;
    w.write_u32::<LittleEndian>((pcm_i16.len() * 2) as u32)?;
    for sample in pcm_i16 {
        w.write_i16::<LittleEndian>(sample)?;
    }

    //----------------------------------------------------------------------
    // 4.  Go back & fill in RIFF chunk size
    //----------------------------------------------------------------------
    let file_len = w.seek(SeekFrom::End(0))?;
    let riff_size = (file_len - 8) as u32;
    w.seek(SeekFrom::Start(4))?;
    w.write_u32::<LittleEndian>(riff_size)?;
    w.flush().context("flush wav")?;

    Ok(())
}
