//! Audio chunk types for streaming TTS synthesis operations
//!
//! These types represent partial audio data that flows through AsyncStream<T>
//! and are designed to work with the cyrup_sugars NotResult constraint.

use fluent_voice_domain::{AudioFormat, VoiceError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Chunk of synthesized audio for streaming TTS operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioChunk {
    /// Raw audio data (typically PCM samples)
    pub audio_data: Vec<u8>,
    
    /// Audio format specification
    pub format: AudioFormat,
    
    /// Duration of this chunk in milliseconds
    pub duration_ms: Option<u64>,
    
    /// Sample rate in Hz
    pub sample_rate: Option<u32>,
    
    /// Whether this is the final chunk in the synthesis
    pub is_final: bool,
    
    /// Sequence number for chunk ordering
    pub sequence: Option<u64>,
    
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Synthesis chunk that can contain either audio data or an error
///
/// This type implements the cyrup_sugars pattern for handling Results
/// in streaming operations while maintaining the NotResult constraint.
#[derive(Debug, Clone)]
pub struct SynthesisChunk {
    inner: Result<AudioChunk, VoiceError>,
}

impl AudioChunk {
    /// Create a new audio chunk with the specified data and format
    pub fn new(audio_data: Vec<u8>, format: AudioFormat) -> Self {
        Self {
            audio_data,
            format,
            duration_ms: None,
            sample_rate: None,
            is_final: false,
            sequence: None,
            metadata: HashMap::new(),
        }
    }
    
    /// Create an empty final chunk to signal end of synthesis
    pub fn final_chunk() -> Self {
        Self {
            audio_data: Vec::new(),
            format: AudioFormat::Mp3_44100_192,
            duration_ms: Some(0),
            sample_rate: None,
            is_final: true,
            sequence: None,
            metadata: HashMap::new(),
        }
    }
    
    /// Create an error chunk containing error information
    pub fn error(error: VoiceError) -> SynthesisChunk {
        SynthesisChunk {
            inner: Err(error),
        }
    }
    
    /// Set the duration of this chunk
    pub fn with_duration(mut self, duration_ms: u64) -> Self {
        self.duration_ms = Some(duration_ms);
        self
    }
    
    /// Set the sample rate of this chunk
    pub fn with_sample_rate(mut self, sample_rate: u32) -> Self {
        self.sample_rate = Some(sample_rate);
        self
    }
    
    /// Mark this chunk as the final chunk
    pub fn with_final(mut self) -> Self {
        self.is_final = true;
        self
    }
    
    /// Set the sequence number for ordering
    pub fn with_sequence(mut self, sequence: u64) -> Self {
        self.sequence = Some(sequence);
        self
    }
    
    /// Add metadata to this chunk
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
    
    /// Get the size of the audio data in bytes
    pub fn size_bytes(&self) -> usize {
        self.audio_data.len()
    }
    
    /// Check if this chunk contains audio data
    pub fn has_audio(&self) -> bool {
        !self.audio_data.is_empty()
    }
    
    /// Convert to PCM i16 samples if the format supports it
    pub fn to_pcm_samples(&self) -> Option<Vec<i16>> {
        match self.format {
            AudioFormat::Mp3_22050_64 | AudioFormat::Mp3_44100_64 | 
            AudioFormat::Mp3_44100_96 | AudioFormat::Mp3_44100_128 | 
            AudioFormat::Mp3_44100_192 => {
                // For MP3, we'd need to decode - this is a placeholder
                None
            }
            AudioFormat::PcmI16Le22050 | AudioFormat::PcmI16Le44100 => {
                // Convert raw bytes to i16 samples
                if self.audio_data.len() % 2 != 0 {
                    return None;
                }
                
                let samples: Vec<i16> = self.audio_data
                    .chunks_exact(2)
                    .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();
                Some(samples)
            }
            AudioFormat::UlawMonoI16Le8000 => {
                // μ-law decoding would be needed here
                None
            }
        }
    }
}

impl SynthesisChunk {
    /// Create a successful synthesis chunk
    pub fn ok(chunk: AudioChunk) -> Self {
        Self {
            inner: Ok(chunk),
        }
    }
    
    /// Create an error synthesis chunk
    pub fn err(error: VoiceError) -> Self {
        Self {
            inner: Err(error),
        }
    }
    
    /// Consume this chunk and return the inner Result
    pub fn into_inner(self) -> Result<AudioChunk, VoiceError> {
        self.inner
    }
    
    /// Get a reference to the inner Result
    pub fn as_ref(&self) -> Result<&AudioChunk, &VoiceError> {
        self.inner.as_ref()
    }
    
    /// Check if this chunk contains a successful audio chunk
    pub fn is_ok(&self) -> bool {
        self.inner.is_ok()
    }
    
    /// Check if this chunk contains an error
    pub fn is_err(&self) -> bool {
        self.inner.is_err()
    }
    
    /// Convert into an AudioChunk, following the cyrup_sugars pattern
    /// This enables the `synthesis_chunk.into()` syntax
    pub fn into(self) -> AudioChunk {
        match self.inner {
            Ok(chunk) => chunk,
            Err(e) => AudioChunk::new(Vec::new(), AudioFormat::Mp3_44100_192)
                .with_metadata("error", serde_json::Value::String(e.to_string())),
        }
    }
}

// Implement From conversions for fluent usage
impl From<AudioChunk> for SynthesisChunk {
    fn from(chunk: AudioChunk) -> Self {
        Self::ok(chunk)
    }
}

impl From<VoiceError> for SynthesisChunk {
    fn from(error: VoiceError) -> Self {
        Self::err(error)
    }
}

impl From<Result<AudioChunk, VoiceError>> for SynthesisChunk {
    fn from(result: Result<AudioChunk, VoiceError>) -> Self {
        Self { inner: result }
    }
}

// Implement cyrup_sugars NotResult pattern
// Both AudioChunk and SynthesisChunk can be used in AsyncStream/AsyncTask
unsafe impl cyrup_sugars::async_task::NotResult for AudioChunk {}
unsafe impl cyrup_sugars::async_task::NotResult for SynthesisChunk {}