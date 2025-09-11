//! Audio file metadata extraction and validation.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{Result, SampleError};

/// Metadata for an audio sample
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SampleMetadata {
    /// Path to the audio file
    pub path: PathBuf,
    /// Duration in seconds
    pub duration_secs: f32,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of audio channels
    pub channels: u16,
    /// Audio format (e.g., "wav", "mp3", "flac")
    pub format: String,
    /// Number of samples
    pub num_samples: u64,
    /// File size in bytes
    pub file_size: u64,
    /// Optional speaker ID if available
    pub speaker_id: Option<String>,
    /// Optional language code if available (ISO 639-1)
    pub language: Option<String>,
    /// Optional tags for categorization
    pub tags: Vec<String>,
}

impl SampleMetadata {
    /// Creates a new `SampleMetadata` by analyzing an audio file
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        use symphonia::core::{
            formats::FormatOptions,
            io::{MediaSourceStream, MediaSourceStreamOptions},
            meta::MetadataOptions,
            probe::Hint,
        };

        let path = path.as_ref();
        let file = std::fs::File::open(path).map_err(|e| SampleError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;

        let mss = MediaSourceStream::new(Box::new(file), MediaSourceStreamOptions::default());
        let mut hint = Hint::new();
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            hint.with_extension(ext);
        }

        let format_opts = FormatOptions {
            enable_gapless: false,
            ..Default::default()
        };

        let metadata_opts = MetadataOptions {
            limit_metadata_bytes: symphonia::core::meta::Limit::Maximum(1024 * 1024), // 1MB
            ..Default::default()
        };

        let probed = symphonia::default::get_probe()
            .format(&hint, mss, &format_opts, &metadata_opts)
            .map_err(|e| SampleError::InvalidAudioFile {
                path: path.to_path_buf(),
                source: Box::new(e),
            })?;

        let track = probed
            .format
            .default_track()
            .ok_or_else(|| SampleError::InvalidAudioFile {
                path: path.to_path_buf(),
                source: "No audio tracks found".into(),
            })?;

        let params = &track.codec_params;
        let duration = if let (Some(time_base), Some(n_frames)) =
            (params.time_base, params.n_frames)
        {
            // Calculate duration using time_base and n_frames
            // time_base is a rational number (numer/denom) representing seconds per frame
            let seconds_per_frame = f64::from(time_base.numer) / f64::from(time_base.denom);
            (n_frames as f64 * seconds_per_frame) as f32
        } else if let (Some(sample_rate), Some(n_frames)) = (params.sample_rate, params.n_frames) {
            // Fallback: estimate duration from sample rate and frame count
            n_frames as f32 / sample_rate as f32
        } else {
            // Default to 0 if duration cannot be determined
            0.0
        };

        let file_size = std::fs::metadata(path)
            .map_err(|e| SampleError::Io {
                path: path.to_path_buf(),
                source: e,
            })?
            .len();

        Ok(Self {
            path: path.to_path_buf(),
            duration_secs: duration,
            sample_rate: params.sample_rate.unwrap_or(0),
            channels: params.channels.map_or(0, |c| c.count() as u16),
            format: "audio".to_string(),
            num_samples: params.n_frames.unwrap_or(0),
            file_size,
            speaker_id: None,
            language: None,
            tags: Vec::new(),
        })
    }

    /// Validates the metadata against quality criteria
    pub fn validate(&self) -> Result<()> {
        if self.sample_rate < 8000 || self.sample_rate > 48000 {
            return Err(SampleError::UnsupportedSampleRate(self.sample_rate));
        }

        if self.channels == 0 {
            return Err(SampleError::InvalidAudioFile {
                path: self.path.clone(),
                source: "No audio channels".into(),
            });
        }

        if self.duration_secs < 0.5 {
            return Err(SampleError::InvalidSampleDuration(
                self.duration_secs,
                0.5,
                10.0,
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    /// Generate test WAV data - a simple 1-second sine wave
    fn generate_test_wav_data() -> Vec<u8> {
        let sample_rate = 44100u32;
        let duration_samples = sample_rate;
        let frequency = 440.0; // A4 note
        let pcm_samples: Vec<i16> = (0..duration_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                let sample = (2.0 * std::f32::consts::PI * frequency * t).sin();
                (sample * 16384.0) as i16 // Moderate volume
            })
            .collect();

        let mut wav_data = Vec::new();
        let bytes_len = (pcm_samples.len() * 2) as u32;

        // Write WAV header
        wav_data.extend_from_slice(b"RIFF");
        wav_data.extend_from_slice(&(36 + bytes_len).to_le_bytes());
        wav_data.extend_from_slice(b"WAVE");
        wav_data.extend_from_slice(b"fmt ");
        wav_data.extend_from_slice(&16u32.to_le_bytes());
        wav_data.extend_from_slice(&1u16.to_le_bytes());
        wav_data.extend_from_slice(&1u16.to_le_bytes());
        wav_data.extend_from_slice(&sample_rate.to_le_bytes());
        wav_data.extend_from_slice(&(sample_rate * 2).to_le_bytes());
        wav_data.extend_from_slice(&2u16.to_le_bytes());
        wav_data.extend_from_slice(&16u16.to_le_bytes());
        wav_data.extend_from_slice(b"data");
        wav_data.extend_from_slice(&bytes_len.to_le_bytes());

        // Write PCM data
        for sample in pcm_samples {
            wav_data.extend_from_slice(&sample.to_le_bytes());
        }

        wav_data
    }

    #[test]
    fn test_metadata_validation() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test.wav");

        // Create a minimal WAV file for testing
        let mut file = File::create(&file_path).unwrap();
        file.write_all(&generate_test_wav_data()).unwrap();

        let metadata = SampleMetadata::from_file(&file_path).unwrap();
        assert!(metadata.validate().is_ok());
    }
}
