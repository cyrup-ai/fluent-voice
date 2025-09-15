//! Audio chunk types for streaming TTS synthesis operations
//!
//! These types represent partial audio data that flows through AsyncStream<T>
//! and are designed to work with the cyrup_sugars NotResult constraint.

use cyrup_sugars::prelude::MessageChunk;
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

/// Synthesis chunk that contains audio data (errors are handled separately in stream)
///
/// This type implements the cyrup_sugars pattern for streaming operations
/// while maintaining the NotResult constraint by not containing Result types.
#[derive(Debug, Clone)]
pub struct SynthesisChunk {
    /// The audio chunk data
    pub chunk: AudioChunk,
    /// Whether this represents a successful synthesis
    pub success: bool,
    /// Optional error message if synthesis failed
    pub error_message: Option<String>,
}

/// Error audio stream segment for default error handling
///
/// This type is returned by default error handling when synthesis fails.
/// It logs the error using env_logger and provides a placeholder AudioChunk.
#[derive(Debug, Clone)]
pub struct BadAudioStreamSegment {
    /// The error that occurred
    pub error: VoiceError,
    /// Placeholder audio chunk for the error
    pub placeholder_chunk: AudioChunk,
}

impl BadAudioStreamSegment {
    /// Create a new BadAudioStreamSegment with the given error
    pub fn new(error: VoiceError) -> AudioChunk {
        // Log the error using env_logger
        log::error!("Audio stream error: {}", error);

        // Return a placeholder AudioChunk
        AudioChunk::new(Vec::new(), AudioFormat::Mp3Khz44_192)
            .with_metadata("error", serde_json::json!(error.to_string()))
    }
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
            format: AudioFormat::Mp3Khz44_192,
            duration_ms: Some(0),
            sample_rate: None,
            is_final: true,
            sequence: None,
            metadata: HashMap::new(),
        }
    }

    /// Create an error chunk containing error information
    pub fn error(error: VoiceError) -> SynthesisChunk {
        SynthesisChunk::err(error)
    }

    /// Create a bad chunk for MessageChunk trait
    pub fn bad_chunk(error: String) -> Self {
        Self {
            audio_data: Vec::new(),
            format: AudioFormat::Mp3Khz44_192,
            duration_ms: Some(0),
            sample_rate: None,
            is_final: false,
            sequence: None,
            metadata: {
                let mut map = HashMap::new();
                map.insert("error".to_string(), serde_json::json!(error));
                map
            },
        }
    }

    /// Check if this chunk contains an error
    pub fn is_error_chunk(&self) -> bool {
        self.metadata.contains_key("error")
    }

    /// Get error message from metadata if present
    pub fn error_message(&self) -> Option<&str> {
        self.metadata.get("error").and_then(|v| v.as_str())
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
            AudioFormat::Mp3Khz22_32
            | AudioFormat::Mp3Khz44_32
            | AudioFormat::Mp3Khz44_64
            | AudioFormat::Mp3Khz44_96
            | AudioFormat::Mp3Khz44_128
            | AudioFormat::Mp3Khz44_192
            | AudioFormat::OggOpusKhz48 => {
                // For MP3 and Ogg, we'd need to decode - this is a placeholder
                None
            }
            AudioFormat::Pcm16Khz
            | AudioFormat::Pcm22Khz
            | AudioFormat::Pcm24Khz
            | AudioFormat::Pcm48Khz => {
                // Convert raw bytes to i16 samples
                if self.audio_data.len() % 2 != 0 {
                    return None;
                }

                let samples: Vec<i16> = self
                    .audio_data
                    .chunks_exact(2)
                    .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();
                Some(samples)
            }
            AudioFormat::ULaw8Khz => {
                // μ-law decoding would be needed here
                None
            }
        }
    }

    /// Create an AudioChunk from i16 PCM samples
    pub fn from_pcm_samples(samples: &[i16], format: AudioFormat) -> Self {
        let audio_data: Vec<u8> = samples
            .iter()
            .flat_map(|&sample| sample.to_le_bytes())
            .collect();

        Self::new(audio_data, format)
    }

    /// Attach timestamp metadata to this chunk
    pub fn with_timestamp_metadata(
        mut self,
        metadata: fluent_voice_elevenlabs::TimestampMetadata,
    ) -> Self {
        self.metadata.insert(
            "timestamp_metadata".to_string(),
            serde_json::to_value(&metadata).unwrap_or(serde_json::Value::Null),
        );
        self
    }

    /// Retrieve timestamp metadata from this chunk
    pub fn timestamp_metadata(&self) -> Option<fluent_voice_elevenlabs::TimestampMetadata> {
        self.metadata
            .get("timestamp_metadata")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
    }

    /// Check if this chunk contains timestamp information
    pub fn has_timestamps(&self) -> bool {
        self.metadata.contains_key("timestamp_metadata")
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

impl SynthesisChunk {
    /// Create a successful synthesis chunk
    pub fn ok(chunk: AudioChunk) -> Self {
        Self {
            chunk,
            success: true,
            error_message: None,
        }
    }

    /// Create an error synthesis chunk
    pub fn err(error: VoiceError) -> Self {
        Self {
            chunk: AudioChunk::new(Vec::new(), AudioFormat::Mp3Khz44_192),
            success: false,
            error_message: Some(error.to_string()),
        }
    }

    /// Check if this chunk contains a successful audio chunk
    pub fn is_ok(&self) -> bool {
        self.success
    }

    /// Check if this chunk contains an error
    pub fn is_err(&self) -> bool {
        !self.success
    }

    /// Get the audio chunk (always available)
    pub fn chunk(&self) -> &AudioChunk {
        &self.chunk
    }

    /// Get the error message if any
    pub fn error(&self) -> Option<&str> {
        self.error_message.as_deref()
    }

    /// Convert into an AudioChunk, following the cyrup_sugars pattern
    /// This enables the `synthesis_chunk.into()` syntax
    pub fn into_chunk(self) -> AudioChunk {
        self.chunk
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
        match result {
            Ok(chunk) => Self::ok(chunk),
            Err(error) => Self::err(error),
        }
    }
}

// Implement cyrup_sugars NotResult pattern
// Both AudioChunk and SynthesisChunk can be used in AsyncStream/AsyncTask
impl cyrup_sugars::NotResult for AudioChunk {}
impl cyrup_sugars::NotResult for SynthesisChunk {}

// Implement MessageChunk trait for AudioChunk
impl MessageChunk for AudioChunk {
    fn bad_chunk(error: String) -> Self {
        Self::bad_chunk(error)
    }

    fn error(&self) -> Option<&str> {
        self.error_message()
    }

    fn is_error(&self) -> bool {
        self.is_error_chunk()
    }
}

// Implement MessageChunk trait for SynthesisChunk
impl MessageChunk for SynthesisChunk {
    fn bad_chunk(error: String) -> Self {
        Self {
            chunk: AudioChunk::bad_chunk(error.clone()),
            success: false,
            error_message: Some(error),
        }
    }

    fn error(&self) -> Option<&str> {
        self.error_message.as_deref()
    }

    fn is_error(&self) -> bool {
        !self.success
    }
}

// Remove MessageChunk implementation for String - trait not defined in current crate
// impl MessageChunk for String {
//     fn bad_chunk(error: String) -> Self {
//         format!("[ERROR: {}]", error)
//     }
//
//     fn error(&self) -> Option<&str> {
//         if self.starts_with("[ERROR:") && self.ends_with("]") {
//             Some(&self[8..self.len() - 1])
//         } else {
//             None
//         }
//     }
//
//     fn is_error(&self) -> bool {
//         self.starts_with("[ERROR:") && self.ends_with("]")
//     }
// }

/// Helper function to convert Stream<i16> to Stream<Vec<u8>>
///
/// This function batches i16 samples into Vec<u8> segments for streaming synthesis.
/// It chunks the samples into reasonable-sized audio segments.
pub fn i16_stream_to_bytes_stream<S>(
    stream: S,
    chunk_size: usize,
) -> impl futures_core::Stream<Item = Vec<u8>> + Send + Unpin
where
    S: futures_core::Stream<Item = i16> + Send + Unpin + 'static,
{
    use futures_util::StreamExt;

    stream.chunks(chunk_size).map(|samples| {
        samples
            .into_iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect()
    })
}

/// Helper function to convert Stream<String> to Stream<Vec<u8>>
///
/// This function converts string streams to byte streams for TTS examples.
/// Used when the conversation stream returns strings instead of raw audio samples.
pub fn string_stream_to_bytes_stream<S>(
    stream: S,
    _chunk_size: usize,
) -> impl futures_core::Stream<Item = Vec<u8>> + Send + Unpin
where
    S: futures_core::Stream<Item = String> + Send + Unpin + 'static,
{
    use futures_util::StreamExt;
    stream.map(|text| text.into_bytes())
}

/// Error transcription segment for default error handling
#[derive(Debug, Clone)]
pub struct BadTranscriptionSegment {
    error_message: String,
}

impl fluent_voice_domain::transcription::TranscriptionSegment for BadTranscriptionSegment {
    fn start_ms(&self) -> u32 {
        0
    }
    fn end_ms(&self) -> u32 {
        0
    }
    fn text(&self) -> &str {
        &self.error_message
    }
    fn speaker_id(&self) -> Option<&str> {
        None
    }
}

impl BadTranscriptionSegment {
    /// Create a bad transcription segment from an error
    pub fn from_err(error: fluent_voice_domain::VoiceError) -> Self {
        BadTranscriptionSegment {
            error_message: format!("[ERROR: {}]", error),
        }
    }
}

/// Helper function to convert Stream<i16> to Stream<AudioChunk>
///
/// This function batches i16 samples into AudioChunk segments for streaming synthesis.
/// It chunks the samples into reasonable-sized audio segments.
pub fn i16_stream_to_audio_chunk_stream<S>(
    stream: S,
    format: AudioFormat,
    chunk_size: usize,
) -> impl futures_core::Stream<Item = AudioChunk> + Send + Unpin
where
    S: futures_core::Stream<Item = i16> + Send + Unpin + 'static,
{
    use futures_util::StreamExt;

    stream
        .chunks(chunk_size)
        .map(move |samples| AudioChunk::from_pcm_samples(&samples, format))
}

/// Helper function to convert TranscriptStream to Stream<String>
///
/// This function converts a transcript stream (which yields Result<TranscriptionSegment, VoiceError>)
/// to a stream of strings for direct consumption.
pub fn transcript_stream_to_string_stream<S, T>(
    stream: S,
) -> impl futures_core::Stream<Item = String> + Send + Unpin
where
    S: futures_core::Stream<Item = Result<T, VoiceError>> + Send + Unpin + 'static,
    T: fluent_voice_domain::transcription::TranscriptionSegment,
{
    use futures_util::StreamExt;

    stream.map(|result| match result {
        Ok(segment) => segment.text().to_string(),
        Err(e) => {
            log::error!("Transcript error: {}", e);
            String::new()
        }
    })
}
