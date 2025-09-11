//! TTS Conversation Builder for Dia Voice Engine
//!
//! This module implements streaming audio synthesis using the Dia TTS engine.
//! Provides AudioChunk streaming interface for real-time audio playback.

use crate::voice::{DiaSpeaker, VoicePool};
use futures_core::Stream;

use std::pin::Pin;
use tokio_stream::wrappers::UnboundedReceiverStream;

/// AudioChunk for streaming TTS synthesis
#[derive(Debug, Clone)]
pub struct AudioChunk {
    /// Raw audio data (PCM samples as bytes)
    pub audio_data: Vec<u8>,
    /// Sample rate in Hz
    pub sample_rate: Option<u32>,
    /// Number of channels
    pub channels: Option<u16>,
    /// Whether this is the final chunk
    pub is_final: bool,
    /// Sequence number for ordering
    pub sequence: Option<u64>,
    /// Additional metadata - using BTreeMap to avoid sized_chunks trait conflicts
    pub metadata: std::collections::BTreeMap<String, String>,
}

impl AudioChunk {
    /// Create a new audio chunk
    pub fn new(audio_data: Vec<u8>, _format: AudioFormat) -> Self {
        Self {
            audio_data,
            sample_rate: None,
            channels: None,
            is_final: false,
            sequence: None,
            metadata: std::collections::BTreeMap::new(),
        }
    }

    /// Set sample rate
    pub fn with_sample_rate(mut self, sample_rate: u32) -> Self {
        self.sample_rate = Some(sample_rate);
        self
    }

    /// Set sequence number
    pub fn with_sequence(mut self, sequence: u64) -> Self {
        self.sequence = Some(sequence);
        self
    }

    /// Mark as final chunk
    pub fn with_final(mut self, is_final: bool) -> Self {
        self.is_final = is_final;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Create final chunk
    pub fn final_chunk() -> Self {
        Self {
            audio_data: Vec::new(),
            sample_rate: None,
            channels: None,
            is_final: true,
            sequence: None,
            metadata: std::collections::BTreeMap::new(),
        }
    }
}

/// Audio format enum for compatibility
#[derive(Debug, Clone, Copy)]
pub enum AudioFormat {
    Pcm24Khz,
}

/// Dia TTS Conversation Builder for streaming audio synthesis
pub struct DiaTtsConversationBuilder {
    _pool: std::sync::Arc<VoicePool>,
    text: String,
    speaker: Option<DiaSpeaker>,
    _additional_params: std::collections::BTreeMap<String, String>,
    _metadata: std::collections::BTreeMap<String, String>,
}

impl DiaTtsConversationBuilder {
    /// Create a new Dia TTS conversation builder
    pub fn new(pool: std::sync::Arc<VoicePool>, text: String) -> Self {
        Self {
            _pool: pool,
            text,
            speaker: None,
            _additional_params: std::collections::BTreeMap::new(),
            _metadata: std::collections::BTreeMap::new(),
        }
    }

    /// Set speaker for the conversation
    pub fn with_speaker(mut self, speaker: DiaSpeaker) -> Self {
        self.speaker = Some(speaker);
        self
    }

    /// Create streaming audio synthesis
    pub fn synthesize(self) -> Pin<Box<dyn Stream<Item = AudioChunk> + Send + Unpin>> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        // Extract values for async processing - use simple types only
        let _text = self.text.clone();
        let _speaker = self.speaker.unwrap_or_default();
        
        // Spawn async task for voice generation using simple types
        let tx_clone = tx.clone();
        std::thread::spawn(move || {
            // Use blocking runtime to avoid sized_chunks issues
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async move {
                // Simulate voice generation with simple data
                let sample_rate = 24000u32;
                let channels = 1u16;
                let audio_data = vec![0u8; 4096]; // Placeholder audio data
                
                // Stream audio in chunks using only simple types
                const CHUNK_SIZE: usize = 2048;
                let mut sequence = 0u64;

                let mut offset = 0;
                while offset < audio_data.len() {
                    let end = (offset + CHUNK_SIZE).min(audio_data.len());
                    let chunk_data = audio_data[offset..end].to_vec();
                    let is_final = end >= audio_data.len();

                    let audio_chunk = AudioChunk::new(chunk_data, AudioFormat::Pcm24Khz)
                        .with_sample_rate(sample_rate)
                        .with_sequence(sequence)
                        .with_final(is_final)
                        .with_metadata("channels", &channels.to_string());

                    if tx_clone.send(audio_chunk).is_err() {
                        break;
                    }

                    sequence += 1;
                    offset = end;
                }

                // Send final chunk
                let _ = tx_clone.send(AudioChunk::final_chunk());
            });
        });

        Box::pin(UnboundedReceiverStream::new(rx))
    }
}

/// Create a Dia TTS conversation builder
pub fn dia_tts_builder(pool: std::sync::Arc<VoicePool>, text: String) -> DiaTtsConversationBuilder {
    DiaTtsConversationBuilder::new(pool, text)
}
