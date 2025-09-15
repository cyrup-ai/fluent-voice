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

/// Complete synthesis context for timestamp metadata generation
#[derive(Debug, Clone)]
pub struct SynthesisContext {
    pub voice_id: String,
    pub model_id: String,
    pub text: String,
    pub voice_settings: Option<VoiceSettingsSnapshot>,
    pub output_format: String,
    pub language: Option<String>,
}

impl SynthesisContext {
    /// Create SynthesisContext from TtsBuilder context
    pub fn from_tts_builder(
        voice_id: &str,
        model_id: &str,
        text: Option<&str>,
        voice_settings: Option<&crate::shared::VoiceSettings>,
        output_format: &str,
        language_code: Option<&str>,
    ) -> Self {
        Self {
            voice_id: voice_id.to_string(),
            model_id: model_id.to_string(),
            text: text.unwrap_or("").to_string(),
            voice_settings: voice_settings.map(|vs| VoiceSettingsSnapshot {
                // ElevenLabs API defaults as per their official documentation
                stability: vs.stability.unwrap_or(0.5), // Default stability
                similarity_boost: vs.similarity_boost.unwrap_or(0.75), // Default similarity boost
                style: vs.style,
                use_speaker_boost: vs.use_speaker_boost,
                speed: vs.speed,
            }),
            output_format: output_format.to_string(),
            language: language_code.map(|s| s.to_string()),
        }
    }
}

/// Validate that all alignment arrays have consistent lengths
fn validate_alignment_consistency(
    alignment: &Alignment,
) -> Result<(), crate::engine::FluentVoiceError> {
    let char_count = alignment.characters.len();
    let start_count = alignment.character_start_times_seconds.len();
    let end_count = alignment.character_end_times_seconds.len();

    if char_count != start_count || char_count != end_count {
        return Err(crate::engine::FluentVoiceError::AlignmentValidationError(
            format!(
                "Alignment array length mismatch: {} characters, {} start times, {} end times",
                char_count, start_count, end_count
            ),
        ));
    }

    if char_count == 0 {
        return Err(crate::engine::FluentVoiceError::AlignmentValidationError(
            "Empty alignment data received from ElevenLabs API".into(),
        ));
    }

    Ok(())
}

/// Validate timing values are logical and well-formed
fn validate_timing_logic(alignment: &Alignment) -> Result<(), crate::engine::FluentVoiceError> {
    for (i, (&start, &end)) in alignment
        .character_start_times_seconds
        .iter()
        .zip(&alignment.character_end_times_seconds)
        .enumerate()
    {
        // Check for negative timing values
        if start < 0.0 || end < 0.0 {
            return Err(crate::engine::FluentVoiceError::AlignmentValidationError(
                format!(
                    "Negative timing values at character {}: start={:.3}s, end={:.3}s",
                    i, start, end
                ),
            ));
        }

        // Check for invalid timing order
        if start >= end {
            return Err(crate::engine::FluentVoiceError::AlignmentValidationError(
                format!(
                    "Invalid timing order at character {}: start={:.3}s >= end={:.3}s",
                    i, start, end
                ),
            ));
        }

        // Check for unreasonable timing values (> 1 hour)
        if end > 3600.0 {
            return Err(crate::engine::FluentVoiceError::AlignmentValidationError(
                format!(
                    "Unreasonable timing value at character {}: end={:.3}s (> 1 hour)",
                    i, end
                ),
            ));
        }
    }

    // Validate timing sequence is monotonic
    let mut prev_end = 0.0;
    for (i, &start) in alignment.character_start_times_seconds.iter().enumerate() {
        if start < prev_end {
            return Err(crate::engine::FluentVoiceError::AlignmentValidationError(
                format!(
                    "Non-monotonic timing sequence at character {}: start={:.3}s < previous_end={:.3}s",
                    i, start, prev_end
                ),
            ));
        }
        prev_end = alignment.character_end_times_seconds[i];
    }

    Ok(())
}

/// Validate character data integrity and format
fn validate_character_data(alignment: &Alignment) -> Result<(), crate::engine::FluentVoiceError> {
    for (i, character) in alignment.characters.iter().enumerate() {
        // Check for empty characters
        if character.is_empty() {
            return Err(crate::engine::FluentVoiceError::AlignmentValidationError(
                format!("Empty character string at position {}", i),
            ));
        }

        // Check for valid UTF-8 (should be guaranteed by String type, but explicit check)
        if !character.is_ascii() {
            // Log warning for non-ASCII characters but don't fail
            tracing::warn!("Non-ASCII character at position {}: '{}'", i, character);
        }

        // Check for reasonable character length (prevent extremely long strings)
        if character.len() > 10 {
            return Err(crate::engine::FluentVoiceError::AlignmentValidationError(
                format!(
                    "Unusually long character string at position {}: '{}' ({} bytes)",
                    i,
                    character,
                    character.len()
                ),
            ));
        }
    }

    Ok(())
}

/// Comprehensive validation for ElevenLabs alignment data
pub fn validate_alignment(alignment: &Alignment) -> Result<(), crate::engine::FluentVoiceError> {
    validate_alignment_consistency(alignment)?;
    validate_timing_logic(alignment)?;
    validate_character_data(alignment)?;

    tracing::debug!(
        "Alignment validation passed: {} characters, {:.3}s duration",
        alignment.characters.len(),
        alignment.character_end_times_seconds.last().unwrap_or(&0.0)
    );

    Ok(())
}

/// Convert ElevenLabs Alignment to CharacterTimestamp vector
impl From<&Alignment> for Vec<CharacterTimestamp> {
    fn from(alignment: &Alignment) -> Self {
        // ✅ ADD VALIDATION HERE
        if let Err(e) = validate_alignment(alignment) {
            tracing::error!("Alignment validation failed: {}", e);
            // Return empty vec or handle error appropriately
            return Vec::new();
        }

        // Existing implementation with enhanced bounds checking
        alignment
            .characters
            .iter()
            .enumerate()
            .filter_map(|(i, character)| {
                // Enhanced bounds checking (already exists)
                if i < alignment.character_start_times_seconds.len()
                    && i < alignment.character_end_times_seconds.len()
                {
                    Some(CharacterTimestamp {
                        character: character.clone(),
                        start_seconds: alignment.character_start_times_seconds[i],
                        end_seconds: alignment.character_end_times_seconds[i],
                        text_position: i,
                    })
                } else {
                    tracing::warn!("Array bounds mismatch at character {}", i);
                    None
                }
            })
            .collect()
    }
}

/// Configuration context for timestamp generation
#[derive(Debug, Clone)]
pub struct TimestampConfiguration {
    pub granularity: fluent_voice_domain::timestamps::TimestampsGranularity,
    pub word_timestamps: fluent_voice_domain::timestamps::WordTimestamps,
    pub diarization: fluent_voice_domain::timestamps::Diarization,
    pub punctuation: fluent_voice_domain::timestamps::Punctuation,
}

impl Default for TimestampConfiguration {
    fn default() -> Self {
        Self {
            granularity: fluent_voice_domain::timestamps::TimestampsGranularity::Word,
            word_timestamps: fluent_voice_domain::timestamps::WordTimestamps::On,
            diarization: fluent_voice_domain::timestamps::Diarization::Off,
            punctuation: fluent_voice_domain::timestamps::Punctuation::On,
        }
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

    /// Filter timestamp data based on domain granularity setting
    pub fn filter_by_granularity(
        &mut self,
        granularity: fluent_voice_domain::timestamps::TimestampsGranularity,
    ) {
        use fluent_voice_domain::timestamps::TimestampsGranularity;
        match granularity {
            TimestampsGranularity::None => {
                self.character_alignments.clear();
                self.word_alignments = None;
            }
            TimestampsGranularity::Word => {
                self.character_alignments.clear(); // Keep only word-level
            }
            TimestampsGranularity::Character => {
                // Keep all data - most detailed level
            }
        }
    }

    /// Filter word timestamps based on domain setting
    pub fn filter_by_word_timestamps(
        &mut self,
        setting: fluent_voice_domain::timestamps::WordTimestamps,
    ) {
        use fluent_voice_domain::timestamps::WordTimestamps;
        match setting {
            WordTimestamps::Off => {
                self.word_alignments = None;
            }
            WordTimestamps::On => {
                if self.word_alignments.is_none() && !self.character_alignments.is_empty() {
                    self.generate_word_alignments();
                }
            }
        }
    }

    /// Filter speaker information based on diarization setting
    pub fn filter_by_diarization(&mut self, setting: fluent_voice_domain::timestamps::Diarization) {
        use fluent_voice_domain::timestamps::Diarization;
        match setting {
            Diarization::Off => {
                // Clear speaker IDs from audio chunks
                for chunk in &mut self.audio_chunks {
                    chunk.speaker_id = None;
                }
            }
            Diarization::On => {
                // Preserve existing speaker information
            }
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
                .expect("synthesis_end should be set immediately above")
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
            if let Some(last_char) = self.character_alignments.last() {
                words.push(WordTimestamp {
                    word: current_word,
                    start_seconds: word_start_seconds,
                    end_seconds: last_char.end_seconds,
                    word_position: word_start_pos,
                    character_range: (char_start_pos, self.character_alignments.len()),
                });
            }
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
