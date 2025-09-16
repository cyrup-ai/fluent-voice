//! Audio chunk types for TTS streaming
//!
//! This module defines the structured audio chunk types that are used
//! in TTS streaming operations, providing rich metadata and audio data.

use crate::timestamps::TimestampMetadata;
use crate::voice_error::VoiceError;
use std::collections::HashMap;

use cyrup_sugars::prelude::MessageChunk;

/// Represents a chunk of synthesized audio with metadata
///
/// This type provides structured access to audio data along with
/// timing information, speaker details, and other metadata that
/// enables rich streaming experiences.
#[derive(Debug, Clone)]
pub struct AudioChunk {
    /// The raw audio data as bytes
    pub data: Vec<u8>,
    /// Duration of this chunk in milliseconds
    pub duration_ms: u64,
    /// Cumulative start time in milliseconds
    pub start_ms: u64,
    /// Speaker identifier for this chunk
    pub speaker_id: Option<String>,
    /// Text that generated this audio chunk
    pub text: Option<String>,
    /// Audio format metadata
    pub format: Option<crate::audio_format::AudioFormat>,
    /// Sample rate in Hz (for streaming compatibility)
    pub sample_rate: Option<u32>,
    /// Whether this is the final chunk in the synthesis (for streaming)
    pub is_final: bool,
    /// Sequence number for chunk ordering (for streaming)
    pub sequence: Option<u64>,
    /// Additional metadata key-value pairs
    pub metadata: HashMap<String, serde_json::Value>,
    /// Error information if this chunk represents an error
    error: Option<String>,
}

impl AudioChunk {
    /// Create a new audio chunk with the given data
    pub fn new(data: Vec<u8>) -> Self {
        Self {
            data,
            duration_ms: 0,
            start_ms: 0,
            speaker_id: None,
            text: None,
            format: None,
            sample_rate: None,
            is_final: false,
            sequence: None,
            metadata: HashMap::new(),
            error: None,
        }
    }

    /// Create a new audio chunk with full metadata
    pub fn with_metadata(
        data: Vec<u8>,
        duration_ms: u64,
        start_ms: u64,
        speaker_id: Option<String>,
        text: Option<String>,
        format: Option<crate::audio_format::AudioFormat>,
    ) -> Self {
        Self {
            data,
            duration_ms,
            start_ms,
            speaker_id,
            text,
            format,
            sample_rate: None,
            is_final: false,
            sequence: None,
            metadata: HashMap::new(),
            error: None,
        }
    }

    /// Get the audio data as bytes
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Get the duration in milliseconds
    pub fn duration_ms(&self) -> u64 {
        self.duration_ms
    }

    /// Get the start time in milliseconds
    pub fn start_ms(&self) -> u64 {
        self.start_ms
    }

    /// Get the speaker identifier
    pub fn speaker_id(&self) -> Option<&str> {
        self.speaker_id.as_deref()
    }

    /// Get the source text
    pub fn text(&self) -> Option<&str> {
        self.text.as_deref()
    }

    /// Get the audio format
    pub fn format(&self) -> Option<&crate::audio_format::AudioFormat> {
        self.format.as_ref()
    }

    /// Convert this audio chunk into raw bytes
    pub fn into_bytes(self) -> Vec<u8> {
        self.data
    }

    /// Get the size of the audio data in bytes
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the audio chunk is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Attach timestamp metadata to this chunk
    pub fn with_timestamp_metadata(mut self, metadata: TimestampMetadata) -> Self {
        self.metadata.insert(
            "timestamp_metadata".to_string(),
            serde_json::to_value(&metadata).unwrap_or(serde_json::Value::Null),
        );
        self
    }

    /// Add custom metadata to this chunk
    pub fn add_metadata<V: serde::Serialize>(mut self, key: &str, value: V) -> Self {
        self.metadata.insert(
            key.to_string(),
            serde_json::to_value(&value).unwrap_or(serde_json::Value::Null),
        );
        self
    }

    /// Retrieve timestamp metadata from this chunk
    pub fn timestamp_metadata(&self) -> Option<TimestampMetadata> {
        self.metadata
            .get("timestamp_metadata")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
    }

    /// Check if this chunk contains timestamp information
    pub fn has_timestamps(&self) -> bool {
        self.timestamp_metadata().is_some()
    }

    /// Get custom metadata value
    pub fn get_metadata(&self, key: &str) -> Option<&serde_json::Value> {
        self.metadata.get(key)
    }

    /// Create a new audio chunk with the specified data and format (fluent-voice compatibility)
    pub fn new_with_format(data: Vec<u8>, format: crate::audio_format::AudioFormat) -> Self {
        Self {
            data,
            duration_ms: 0,
            start_ms: 0,
            speaker_id: None,
            text: None,
            format: Some(format),
            sample_rate: None,
            is_final: false,
            sequence: None,
            metadata: HashMap::new(),
            error: None,
        }
    }

    /// Create an empty final chunk to signal end of synthesis
    pub fn final_chunk() -> Self {
        Self {
            data: Vec::new(),
            duration_ms: 0,
            start_ms: 0,
            speaker_id: None,
            text: None,
            format: Some(crate::audio_format::AudioFormat::Mp3Khz44_192),
            sample_rate: None,
            is_final: true,
            sequence: None,
            metadata: HashMap::new(),
            error: None,
        }
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

    /// Set the duration of this chunk (fluent-voice compatibility)
    pub fn with_duration(mut self, duration_ms: u64) -> Self {
        self.duration_ms = duration_ms;
        self
    }

    /// Get the size of the audio data in bytes (fluent-voice compatibility)
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }

    /// Check if this chunk contains audio data (fluent-voice compatibility)
    pub fn has_audio(&self) -> bool {
        !self.data.is_empty()
    }

    /// Convert to PCM i16 samples if the format supports it
    pub fn to_pcm_samples(&self) -> Option<Vec<i16>> {
        let format = self.format.as_ref()?;
        match format {
            crate::audio_format::AudioFormat::Mp3Khz22_32
            | crate::audio_format::AudioFormat::Mp3Khz44_32
            | crate::audio_format::AudioFormat::Mp3Khz44_64
            | crate::audio_format::AudioFormat::Mp3Khz44_96
            | crate::audio_format::AudioFormat::Mp3Khz44_128
            | crate::audio_format::AudioFormat::Mp3Khz44_192
            | crate::audio_format::AudioFormat::OggOpusKhz48 => {
                // For MP3 and Ogg, we'd need to decode - this is a placeholder
                None
            }
            crate::audio_format::AudioFormat::Pcm16Khz
            | crate::audio_format::AudioFormat::Pcm22Khz
            | crate::audio_format::AudioFormat::Pcm24Khz
            | crate::audio_format::AudioFormat::Pcm48Khz => {
                // Convert raw bytes to i16 samples
                if self.data.len() % 2 != 0 {
                    return None;
                }

                let samples: Vec<i16> = self
                    .data
                    .chunks_exact(2)
                    .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();
                Some(samples)
            }
            crate::audio_format::AudioFormat::ULaw8Khz => {
                // μ-law decoding would be needed here
                None
            }
        }
    }

    /// Create an AudioChunk from i16 PCM samples
    pub fn from_pcm_samples(samples: &[i16], format: crate::audio_format::AudioFormat) -> Self {
        let data: Vec<u8> = samples
            .iter()
            .flat_map(|&sample| sample.to_le_bytes())
            .collect();

        Self::new_with_format(data, format)
    }

    /// Check if this chunk contains an error (fluent-voice compatibility)
    pub fn is_error_chunk(&self) -> bool {
        self.error.is_some() || self.metadata.contains_key("error")
    }

    /// Get error message from metadata if present (fluent-voice compatibility)
    pub fn error_message(&self) -> Option<&str> {
        self.error
            .as_deref()
            .or_else(|| self.metadata.get("error").and_then(|v| v.as_str()))
    }

    /// Export timestamps as SRT format
    pub fn export_srt(&self) -> Option<Result<String, VoiceError>> {
        self.timestamp_metadata().map(|metadata| {
            metadata
                .to_srt()
                .map_err(|e| VoiceError::Configuration(e.to_string()))
        })
    }

    /// Export timestamps as WebVTT format  
    pub fn export_vtt(&self) -> Option<Result<String, VoiceError>> {
        self.timestamp_metadata().map(|metadata| {
            metadata
                .to_vtt()
                .map_err(|e| VoiceError::Configuration(e.to_string()))
        })
    }
}

impl From<Vec<u8>> for AudioChunk {
    fn from(data: Vec<u8>) -> Self {
        Self::new(data)
    }
}

impl From<AudioChunk> for Vec<u8> {
    fn from(chunk: AudioChunk) -> Self {
        chunk.data
    }
}

/// Result type for audio chunk operations
pub type AudioChunkResult = Result<AudioChunk, VoiceError>;

// Implement cyrup_sugars NotResult pattern for AudioChunk
impl cyrup_sugars::NotResult for AudioChunk {}

impl MessageChunk for AudioChunk {
    fn bad_chunk(error: String) -> Self {
        AudioChunk {
            data: Vec::new(),
            duration_ms: 0,
            start_ms: 0,
            speaker_id: None,
            text: Some(format!("[ERROR] {}", error)),
            format: None,
            sample_rate: None,
            is_final: false,
            sequence: None,
            metadata: HashMap::new(),
            error: Some(error),
        }
    }

    fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }
}
