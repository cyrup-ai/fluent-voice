// src/wav.rs

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::error::{MoshiError, Result};

/// Writes PCM samples as a WAV file.
///
/// This function writes the provided PCM samples to a WAV file with the specified sample rate and number of channels.
/// The samples are expected to be in f32 format and will be converted to i16 for writing.
///
/// # Arguments
///
/// * `path` - The path to the output WAV file.
/// * `samples` - A slice of f32 PCM samples.
/// * `sample_rate` - The sample rate in Hz.
/// * `channels` - The number of audio channels.
///
/// # Returns
///
/// * `Result<()>` - Ok if the file was written successfully, otherwise an error.
pub fn write_pcm_as_wav<P: AsRef<Path>, S: AsRef<[f32]>>(
    path: P,
    samples: S,
    sample_rate: u32,
    channels: u16,
) -> Result<()> {
    let samples = samples.as_ref();
    let mut file =
        BufWriter::new(File::create(path.as_ref()).map_err(|e| MoshiError::Io(e.into()))?);

    let num_samples = samples.len() as u32;
    let bytes_per_sample = 2u32; // i16
    let block_align = channels as u32 * bytes_per_sample;
    let avg_bytes_per_sec = sample_rate * block_align;
    let subchunk2_size = num_samples * block_align;
    let chunk_size = 36 + subchunk2_size;

    // Write RIFF header
    file.write_all(b"RIFF")
        .map_err(|e| MoshiError::Io(e.into()))?;
    file.write_u32::<LittleEndian>(chunk_size)
        .map_err(|e| MoshiError::Io(e.into()))?;
    file.write_all(b"WAVE")
        .map_err(|e| MoshiError::Io(e.into()))?;

    // Write fmt subchunk
    file.write_all(b"fmt ")
        .map_err(|e| MoshiError::Io(e.into()))?;
    file.write_u32::<LittleEndian>(16)
        .map_err(|e| MoshiError::Io(e.into()))?; // Subchunk1Size for PCM
    file.write_u16::<LittleEndian>(1)
        .map_err(|e| MoshiError::Io(e.into()))?; // AudioFormat: PCM
    file.write_u16::<LittleEndian>(channels)
        .map_err(|e| MoshiError::Io(e.into()))?;
    file.write_u32::<LittleEndian>(sample_rate)
        .map_err(|e| MoshiError::Io(e.into()))?;
    file.write_u32::<LittleEndian>(avg_bytes_per_sec)
        .map_err(|e| MoshiError::Io(e.into()))?;
    file.write_u16::<LittleEndian>(block_align as u16)
        .map_err(|e| MoshiError::Io(e.into()))?;
    file.write_u16::<LittleEndian>(16)
        .map_err(|e| MoshiError::Io(e.into()))?; // BitsPerSample

    // Write data subchunk
    file.write_all(b"data")
        .map_err(|e| MoshiError::Io(e.into()))?;
    file.write_u32::<LittleEndian>(subchunk2_size)
        .map_err(|e| MoshiError::Io(e.into()))?;

    // Write samples as i16
    for &sample in samples {
        let i16_sample = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
        file.write_i16::<LittleEndian>(i16_sample)
            .map_err(|e| MoshiError::Io(e.into()))?;
    }

    Ok(())
}

/// Reads a WAV file and returns the PCM samples as Vec<f32>.
///
/// This function reads a WAV file and extracts the PCM samples, converting them to f32 format.
///
/// # Arguments
///
/// * `path` - The path to the input WAV file.
///
/// # Returns
///
/// * `Result<(Vec<f32>, u32, u16)>` - A tuple containing the PCM samples, sample rate, and number of channels.
pub fn read_wav<P: AsRef<Path>>(path: P) -> Result<(Vec<f32>, u32, u16)> {
    let mut file = BufReader::new(File::open(path.as_ref()).map_err(|e| MoshiError::Io(e.into()))?);

    // Read RIFF header
    let mut riff = [0u8; 4];
    file.read_exact(&mut riff)
        .map_err(|e| MoshiError::Io(e.into()))?;
    if riff != *b"RIFF" {
        return Err(MoshiError::Custom("Invalid RIFF header".into()));
    }

    let chunk_size = file
        .read_u32::<LittleEndian>()
        .map_err(|e| MoshiError::Io(e.into()))?;

    // Validate chunk size is reasonable (WAV files should be at least 44 bytes)
    if chunk_size < 36 {
        return Err(MoshiError::Custom(
            "Invalid chunk size in RIFF header".into(),
        ));
    }

    let mut wave = [0u8; 4];
    file.read_exact(&mut wave)
        .map_err(|e| MoshiError::Io(e.into()))?;
    if wave != *b"WAVE" {
        return Err(MoshiError::Custom("Invalid WAVE format".into()));
    }

    // Read fmt subchunk
    let mut fmt = [0u8; 4];
    file.read_exact(&mut fmt)
        .map_err(|e| MoshiError::Io(e.into()))?;
    if fmt != *b"fmt " {
        return Err(MoshiError::Custom("Invalid fmt subchunk".into()));
    }

    let subchunk1_size = file
        .read_u32::<LittleEndian>()
        .map_err(|e| MoshiError::Io(e.into()))?;
    if subchunk1_size < 16 {
        return Err(MoshiError::Custom("Invalid fmt subchunk size".into()));
    }

    let audio_format = file
        .read_u16::<LittleEndian>()
        .map_err(|e| MoshiError::Io(e.into()))?;
    if audio_format != 1 {
        return Err(MoshiError::Custom("Only PCM format is supported".into()));
    }

    let channels = file
        .read_u16::<LittleEndian>()
        .map_err(|e| MoshiError::Io(e.into()))?;

    let sample_rate = file
        .read_u32::<LittleEndian>()
        .map_err(|e| MoshiError::Io(e.into()))?;

    let _byte_rate = file
        .read_u32::<LittleEndian>()
        .map_err(|e| MoshiError::Io(e.into()))?;

    let _block_align = file
        .read_u16::<LittleEndian>()
        .map_err(|e| MoshiError::Io(e.into()))?;

    let bits_per_sample = file
        .read_u16::<LittleEndian>()
        .map_err(|e| MoshiError::Io(e.into()))?;
    if bits_per_sample != 16 {
        return Err(MoshiError::Custom(
            "Only 16-bit samples are supported".into(),
        ));
    }

    // Skip extra fmt bytes if any
    if subchunk1_size > 16 {
        let mut extra = vec![0u8; (subchunk1_size - 16) as usize];
        file.read_exact(&mut extra)
            .map_err(|e| MoshiError::Io(e.into()))?;
    }

    // Read data subchunk
    loop {
        let mut subchunk_id = [0u8; 4];
        if file.read_exact(&mut subchunk_id).is_err() {
            return Err(MoshiError::Custom("No data subchunk found".into()));
        }
        if subchunk_id == *b"data" {
            break;
        }
        let subchunk_size = file
            .read_u32::<LittleEndian>()
            .map_err(|e| MoshiError::Io(e.into()))?;
        let mut skip = vec![0u8; subchunk_size as usize];
        file.read_exact(&mut skip)
            .map_err(|e| MoshiError::Io(e.into()))?;
    }

    let subchunk2_size = file
        .read_u32::<LittleEndian>()
        .map_err(|e| MoshiError::Io(e.into()))?;

    let num_samples = (subchunk2_size / (channels as u32 * 2)) as usize;
    let mut samples = Vec::with_capacity(num_samples);

    for _ in 0..num_samples {
        let sample = file
            .read_i16::<LittleEndian>()
            .map_err(|e| MoshiError::Io(e.into()))?;
        samples.push(sample as f32 / 32768.0);
    }

    Ok((samples, sample_rate, channels))
}
