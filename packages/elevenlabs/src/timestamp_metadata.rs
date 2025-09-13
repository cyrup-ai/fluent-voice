//! Comprehensive timestamp metadata structures for ElevenLabs TTS
//!
//! This module provides detailed timing information for synthesized speech,
//! including character-level timestamps, audio chunk timing, and export capabilities.

use crate::endpoints::genai::tts::Alignment;
use serde::{Deserialize, Serialize};
use std::time::SystemTime;

/// Comprehensive timestamp metadata for TTS synthesis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimestampMetadata {
    /// When synthesis started (high precision)
    pub synthesis_start: SystemTime,
    /// When synthesis completed
    pub synthesis_end: Option<SystemTime>,
    /// Audio chunks with precise timing
    pub audio_chunks: Vec<AudioChunkTimestamp>,
    /// Character-level alignments from ElevenLabs API
    pub character_alignments: Vec<CharacterTimestamp>,
    /// Word-level aggregated timestamps (derived from characters)
    pub word_alignments: Option<Vec<WordTimestamp>>,
    /// Total audio duration in milliseconds
    pub total_duration_ms: Option<u64>,
    /// Processing time in milliseconds
    pub processing_time_ms: Option<u64>,
    /// Voice and model metadata
    pub synthesis_metadata: SynthesisMetadata,
}

/// Audio chunk with timing information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioChunkTimestamp {
    /// Chunk sequence number
    pub chunk_id: usize,
    /// Start time in milliseconds relative to audio start
    pub start_ms: u64,
    /// End time in milliseconds relative to audio start
    pub end_ms: u64,
    /// Original text segment for this chunk
    pub text_segment: String,
    /// Speaker ID (for multi-speaker scenarios)
    pub speaker_id: Option<String>,
    /// Audio format for this chunk
    pub format: String,
    /// Chunk size in bytes
    pub size_bytes: usize,
}

/// Character-level timestamp from ElevenLabs API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CharacterTimestamp {
    /// The character or character sequence
    pub character: String,
    /// Start time in seconds (from ElevenLabs API)
    pub start_seconds: f32,
    /// End time in seconds (from ElevenLabs API)
    pub end_seconds: f32,
    /// Character position in original text
    pub text_position: usize,
}

/// Word-level timestamp aggregated from characters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordTimestamp {
    /// The word
    pub word: String,
    /// Start time in seconds (aggregated from characters)
    pub start_seconds: f32,
    /// End time in seconds (aggregated from characters)
    pub end_seconds: f32,
    /// Word position in original text
    pub word_position: usize,
    /// Character positions that make up this word
    pub character_range: (usize, usize),
}

/// Synthesis configuration and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisMetadata {
    /// Voice ID used
    pub voice_id: String,
    /// Model used for synthesis
    pub model_id: String,
    /// Original text
    pub text: String,
    /// Voice settings applied
    pub voice_settings: Option<VoiceSettingsSnapshot>,
    /// Output format
    pub output_format: String,
    /// Language detected/specified
    pub language: Option<String>,
}

impl Default for SynthesisMetadata {
    fn default() -> Self {
        Self {
            voice_id: String::new(),
            model_id: String::new(),
            text: String::new(),
            voice_settings: None,
            output_format: String::new(),
            language: None,
        }
    }
}

/// Snapshot of voice settings used for synthesis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceSettingsSnapshot {
    pub stability: f32,
    pub similarity_boost: f32,
    pub style: Option<f32>,
    pub use_speaker_boost: Option<bool>,
    pub speed: Option<f32>,
}

/// Convert ElevenLabs Alignment to CharacterTimestamp vector
impl From<&Alignment> for Vec<CharacterTimestamp> {
    fn from(alignment: &Alignment) -> Self {
        alignment
            .characters
            .iter()
            .enumerate()
            .map(|(i, character)| CharacterTimestamp {
                character: character.clone(),
                start_seconds: alignment.character_start_times_seconds[i],
                end_seconds: alignment.character_end_times_seconds[i],
                text_position: i,
            })
            .collect()
    }
}

impl TimestampMetadata {
    /// Create new metadata instance with start time
    pub fn new() -> Self {
        Self {
            synthesis_start: SystemTime::now(),
            synthesis_end: None,
            audio_chunks: Vec::new(),
            character_alignments: Vec::new(),
            word_alignments: None,
            total_duration_ms: None,
            processing_time_ms: None,
            synthesis_metadata: SynthesisMetadata::default(),
        }
    }

    /// Add character alignment data from ElevenLabs response
    pub fn add_alignment(&mut self, alignment: &Alignment) {
        self.character_alignments
            .extend(Vec::<CharacterTimestamp>::from(alignment));
        self.generate_word_alignments();
    }

    /// Add an audio chunk with timing information
    pub fn add_chunk(&mut self, chunk: AudioChunkTimestamp) {
        self.audio_chunks.push(chunk);
        self.update_total_duration();
    }

    /// Finalize metadata when synthesis completes
    pub fn finalize(&mut self) -> Result<(), crate::engine::FluentVoiceError> {
        self.synthesis_end = Some(SystemTime::now());
        self.processing_time_ms = Some(
            self.synthesis_end
                .unwrap()
                .duration_since(self.synthesis_start)
                .map_err(|e| {
                    crate::engine::FluentVoiceError::ConfigError(format!(
                        "Time calculation error: {}",
                        e
                    ))
                })?
                .as_millis() as u64,
        );

        if let Some(last_chunk) = self.audio_chunks.last() {
            self.total_duration_ms = Some(last_chunk.end_ms);
        }

        Ok(())
    }

    /// Generate word-level timestamps from character alignments
    fn generate_word_alignments(&mut self) {
        if self.character_alignments.is_empty() {
            return;
        }

        let mut words = Vec::new();
        let mut current_word = String::new();
        let mut word_start_seconds = 0.0f32;
        let mut word_start_pos = 0;
        let mut char_start_pos = 0;

        for (i, char_timestamp) in self.character_alignments.iter().enumerate() {
            if char_timestamp.character.trim().is_empty() || char_timestamp.character == " " {
                // End of word - save current word if not empty
                if !current_word.is_empty() {
                    words.push(WordTimestamp {
                        word: current_word.clone(),
                        start_seconds: word_start_seconds,
                        end_seconds: if i > 0 {
                            self.character_alignments[i - 1].end_seconds
                        } else {
                            char_timestamp.end_seconds
                        },
                        word_position: word_start_pos,
                        character_range: (char_start_pos, i),
                    });
                    current_word.clear();
                    word_start_pos += 1;
                }
                char_start_pos = i + 1;
            } else {
                // Part of current word
                if current_word.is_empty() {
                    word_start_seconds = char_timestamp.start_seconds;
                    char_start_pos = i;
                }
                current_word.push_str(&char_timestamp.character);
            }
        }

        // Handle final word
        if !current_word.is_empty() {
            words.push(WordTimestamp {
                word: current_word,
                start_seconds: word_start_seconds,
                end_seconds: self.character_alignments.last().unwrap().end_seconds,
                word_position: word_start_pos,
                character_range: (char_start_pos, self.character_alignments.len()),
            });
        }

        self.word_alignments = Some(words);
    }

    /// Update total duration from chunks
    fn update_total_duration(&mut self) {
        if let Some(last_chunk) = self.audio_chunks.last() {
            self.total_duration_ms = Some(last_chunk.end_ms);
        }
    }

    /// Serialize to comprehensive JSON
    pub fn to_json(&self) -> Result<String, crate::engine::FluentVoiceError> {
        serde_json::to_string_pretty(self).map_err(|e| {
            crate::engine::FluentVoiceError::ConfigError(format!("Timestamp JSON error: {}", e))
        })
    }
}

impl Default for TimestampMetadata {
    fn default() -> Self {
        Self::new()
    }
}
