//! Audio sample processing and management.
//!
//! This module provides functionality for working with audio samples, including
//! loading, processing, and validating audio data.

use std::path::{Path, PathBuf};

use crate::SampleMetadata;
use crate::error::{Result, SampleError};

/// Represents an audio sample with its associated metadata.
#[derive(Debug, Clone)]
pub struct AudioSample {
    /// Path to the audio file
    pub path: PathBuf,
    /// Metadata about the audio sample
    pub metadata: SampleMetadata,
    /// Optional raw audio data (if loaded)
    pub data: Option<Vec<f32>>,
}

impl AudioSample {
    /// Creates a new `AudioSample` from a file path.
    ///
    /// # Arguments
    /// * `path` - Path to the audio file
    ///
    /// # Returns
    /// A new `AudioSample` with metadata loaded from the file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let metadata = SampleMetadata::from_file(path)?;

        Ok(Self {
            path: path.to_path_buf(),
            metadata,
            data: None,
        })
    }

    /// Loads the audio data from the file into memory.
    ///
    /// # Returns
    /// A reference to the loaded audio data.
    ///
    /// # Errors
    /// Returns an error if the audio file cannot be read or decoded.
    pub fn load_audio_data(&mut self) -> Result<&[f32]> {
        if self.data.is_none() {
            let audio_data = self.decode_audio_file()?;
            self.data = Some(audio_data);
        }

        self.data
            .as_deref()
            .ok_or_else(|| SampleError::AudioDataNotLoaded {
                path: self.path.clone(),
                reason: "Audio data has not been loaded yet".into(),
            })
    }

    /// Decodes an audio file to f32 samples using symphonia.
    ///
    /// # Returns
    /// A vector of f32 audio samples.
    ///
    /// # Errors
    /// Returns an error if the file cannot be decoded.
    fn decode_audio_file(&self) -> Result<Vec<f32>> {
        use std::fs::File;
        use symphonia::core::{
            audio::{AudioBufferRef, Signal},
            codecs::{CODEC_TYPE_NULL, DecoderOptions},
            formats::FormatOptions,
            io::MediaSourceStream,
            meta::MetadataOptions,
            probe::Hint,
        };

        // Open the audio file
        let file = File::open(&self.path).map_err(|e| SampleError::io_error(&self.path, e))?;
        let mss = MediaSourceStream::new(Box::new(file), Default::default());

        // Create a hint based on the file extension
        let mut hint = Hint::new();
        if let Some(extension) = self.path.extension().and_then(|ext| ext.to_str()) {
            hint.with_extension(extension);
        }

        // Probe the media source for format and metadata
        let mut format = symphonia::default::get_probe()
            .format(
                &hint,
                mss,
                &FormatOptions::default(),
                &MetadataOptions::default(),
            )
            .map_err(|e| SampleError::InvalidAudioFile {
                path: self.path.clone(),
                source: Box::new(e),
            })?
            .format;

        // Find the default audio track
        let track = format
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
            .ok_or_else(|| SampleError::UnsupportedAudioFormat("No audio track found".into()))?;

        // Create a decoder for the track
        let mut decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &DecoderOptions::default())
            .map_err(|e| SampleError::InvalidAudioFile {
                path: self.path.clone(),
                source: Box::new(e),
            })?;

        // Decode the audio packets and collect samples
        let mut audio_samples = Vec::new();
        let track_id = track.id;

        loop {
            // Get the next packet from the format reader
            let packet = match format.next_packet() {
                Ok(packet) => packet,
                Err(symphonia::core::errors::Error::ResetRequired) => {
                    // The decoder needs to be reset
                    decoder.reset();
                    continue;
                }
                Err(symphonia::core::errors::Error::IoError(ref e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    // End of stream
                    break;
                }
                Err(e) => {
                    return Err(SampleError::InvalidAudioFile {
                        path: self.path.clone(),
                        source: Box::new(e),
                    });
                }
            };

            // Skip packets for other tracks
            if packet.track_id() != track_id {
                continue;
            }

            // Decode the packet
            match decoder.decode(&packet) {
                Ok(audio_buf) => {
                    // Convert the audio buffer to f32 samples
                    match audio_buf {
                        AudioBufferRef::F32(buf) => {
                            audio_samples.extend_from_slice(buf.chan(0));
                        }
                        AudioBufferRef::F64(buf) => {
                            audio_samples.extend(buf.chan(0).iter().map(|&s| s as f32));
                        }
                        AudioBufferRef::S32(buf) => {
                            const SCALE: f32 = 1.0 / (i32::MAX as f32);
                            audio_samples.extend(buf.chan(0).iter().map(|&s| s as f32 * SCALE));
                        }
                        AudioBufferRef::S16(buf) => {
                            const SCALE: f32 = 1.0 / (i16::MAX as f32);
                            audio_samples.extend(buf.chan(0).iter().map(|&s| f32::from(s) * SCALE));
                        }
                        AudioBufferRef::U32(buf) => {
                            const OFFSET: f32 = (u32::MAX / 2) as f32;
                            const SCALE: f32 = 1.0 / OFFSET;
                            audio_samples
                                .extend(buf.chan(0).iter().map(|&s| (s as f32 - OFFSET) * SCALE));
                        }
                        AudioBufferRef::U16(buf) => {
                            const OFFSET: f32 = (u16::MAX / 2) as f32;
                            const SCALE: f32 = 1.0 / OFFSET;
                            audio_samples.extend(
                                buf.chan(0).iter().map(|&s| (f32::from(s) - OFFSET) * SCALE),
                            );
                        }
                        AudioBufferRef::U8(buf) => {
                            const OFFSET: f32 = (u8::MAX / 2) as f32;
                            const SCALE: f32 = 1.0 / OFFSET;
                            audio_samples.extend(
                                buf.chan(0).iter().map(|&s| (f32::from(s) - OFFSET) * SCALE),
                            );
                        }
                        AudioBufferRef::S24(buf) => {
                            const SCALE: f32 = 1.0 / 8_388_607.0; // 2^23 - 1
                            audio_samples
                                .extend(buf.chan(0).iter().map(|&s| s.inner() as f32 * SCALE));
                        }
                        AudioBufferRef::U24(buf) => {
                            const OFFSET: f32 = 8_388_608.0; // 2^23
                            const SCALE: f32 = 1.0 / 8_388_607.0; // 2^23 - 1
                            audio_samples.extend(
                                buf.chan(0)
                                    .iter()
                                    .map(|&s| (s.inner() as f32 - OFFSET) * SCALE),
                            );
                        }
                        AudioBufferRef::S8(buf) => {
                            const SCALE: f32 = 1.0 / (i8::MAX as f32);
                            audio_samples.extend(buf.chan(0).iter().map(|&s| f32::from(s) * SCALE));
                        }
                    }
                }
                Err(symphonia::core::errors::Error::DecodeError(_)) => {
                    // Decode error, skip this packet
                    continue;
                }
                Err(e) => {
                    return Err(SampleError::InvalidAudioFile {
                        path: self.path.clone(),
                        source: Box::new(e),
                    });
                }
            }
        }

        if audio_samples.is_empty() {
            return Err(SampleError::InvalidAudioFile {
                path: self.path.clone(),
                source: "No audio data found in file".into(),
            });
        }

        Ok(audio_samples)
    }

    /// Validates the audio sample against quality criteria.
    ///
    /// # Returns
    /// `Ok(())` if the sample is valid, or an error describing the issue.
    pub fn validate(&self) -> Result<()> {
        self.metadata.validate()?;

        // Additional validation can be added here

        Ok(())
    }

    /// Gets the duration of the audio sample in seconds.
    #[must_use]
    pub const fn duration_secs(&self) -> f32 {
        self.metadata.duration_secs
    }

    /// Gets the sample rate of the audio in Hz.
    #[must_use]
    pub const fn sample_rate(&self) -> u32 {
        self.metadata.sample_rate
    }

    /// Gets the number of channels in the audio.
    #[must_use]
    pub const fn channels(&self) -> u16 {
        self.metadata.channels
    }
}

/// Processes a batch of audio samples in parallel.
///
/// # Arguments
/// * `samples` - A slice of `AudioSample`s to process
/// * `processor` - A function that processes each sample
///
/// # Returns
/// A vector of results from processing each sample.
pub fn process_samples_batch<F, T>(samples: &[AudioSample], processor: F) -> Vec<Result<T>>
where
    F: Fn(&AudioSample) -> Result<T> + Send + Sync,
    T: Send,
{
    use rayon::prelude::*;

    samples.par_iter().map(processor).collect()
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
                #[allow(clippy::cast_precision_loss)]
                let t = i as f32 / sample_rate as f32;
                let sample = (2.0 * std::f32::consts::PI * frequency * t).sin();
                #[allow(clippy::cast_possible_truncation)]
                let sample_i16 = (sample * 16384.0) as i16; // Moderate volume
                sample_i16
            })
            .collect();

        let mut wav_data = Vec::new();
        #[allow(clippy::cast_possible_truncation)]
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
    fn test_audio_sample_creation() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test.wav");

        // Create a minimal WAV file for testing
        let mut file = File::create(&file_path).unwrap();
        file.write_all(&generate_test_wav_data()).unwrap();

        let sample = AudioSample::from_file(&file_path).unwrap();
        assert!(sample.validate().is_ok());
        assert_eq!(sample.path, file_path);
        assert!(sample.metadata.duration_secs > 0.0);
    }

    #[test]
    fn test_process_samples_batch() {
        let temp_dir = tempdir().unwrap();
        let file_path1 = temp_dir.path().join("test1.wav");
        let file_path2 = temp_dir.path().join("test2.wav");

        // Create test WAV files
        File::create(&file_path1)
            .unwrap()
            .write_all(&generate_test_wav_data())
            .unwrap();
        File::create(&file_path2)
            .unwrap()
            .write_all(&generate_test_wav_data())
            .unwrap();

        let first_sample = AudioSample::from_file(&file_path1).unwrap();
        let second_sample = AudioSample::from_file(&file_path2).unwrap();
        let samples = vec![first_sample, second_sample];

        let results = process_samples_batch(&samples, |sample| Ok(sample.duration_secs()));

        assert_eq!(results.len(), 2);
        for result in results {
            assert!(result.is_ok());
            assert!(result.unwrap() > 0.0);
        }
    }
}
