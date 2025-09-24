//! Timestamp metadata types and configuration for voice processing.
//!
//! This module provides comprehensive timing information for both TTS synthesis
//! and STT transcription, including character-level timestamps, audio chunk timing,
//! and export capabilities that are engine-agnostic.

use serde::{Deserialize, Serialize};
use std::time::SystemTime;

/// Granularity level for timestamp information in transcripts.
///
/// Controls how detailed the timing information should be in the
/// transcription output. Different engines may support different
/// levels of granularity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimestampsGranularity {
    /// No timestamp information included.
    None,
    /// Timestamps at word boundaries.
    Word,
    /// Timestamps at character level (if supported by engine).
    Character,
}

/// Toggle for word-level timestamp inclusion.
///
/// When enabled, each transcribed word will include timing information
/// indicating when it was spoken relative to the audio stream start.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WordTimestamps {
    /// Do not include word timestamps.
    Off,
    /// Include timing information for each word.
    On,
}

/// Toggle for speaker diarization in multi-speaker audio.
///
/// When enabled, the transcription will attempt to identify and
/// label different speakers in the audio stream. This is useful
/// for conversations, meetings, and interviews.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Diarization {
    /// Single speaker mode - no speaker labeling.
    Off,
    /// Multi-speaker mode with speaker identification.
    On,
}

/// Toggle for automatic punctuation insertion.
///
/// When enabled, the transcription engine will automatically
/// insert punctuation marks (periods, commas, question marks, etc.)
/// based on speech patterns and pauses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Punctuation {
    /// Raw transcription without automatic punctuation.
    Off,
    /// Include automatic punctuation in transcripts.
    On,
}

// === Comprehensive Timestamp Metadata Types ===

/// Comprehensive timestamp metadata for TTS synthesis and STT transcription
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimestampMetadata {
    /// When synthesis/transcription started (high precision)
    pub synthesis_start: SystemTime,
    /// When synthesis/transcription completed
    pub synthesis_end: Option<SystemTime>,
    /// Audio chunks with precise timing
    pub audio_chunks: Vec<AudioChunkTimestamp>,
    /// Character-level alignments
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

/// Character-level timestamp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CharacterTimestamp {
    /// The character or character sequence
    pub character: String,
    /// Start time in seconds
    pub start_seconds: f32,
    /// End time in seconds
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

/// Synthesis/transcription configuration and metadata
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SynthesisMetadata {
    /// Voice ID used
    pub voice_id: String,
    /// Model used for synthesis/transcription
    pub model_id: String,
    /// Original text
    pub text: String,
    /// Voice settings applied (engine-specific JSON)
    pub voice_settings: Option<serde_json::Value>,
    /// Output format
    pub output_format: String,
    /// Language detected/specified
    pub language: Option<String>,
}

/// Configuration context for timestamp generation
#[derive(Debug, Clone)]
pub struct TimestampConfiguration {
    pub granularity: TimestampsGranularity,
    pub word_timestamps: WordTimestamps,
    pub diarization: Diarization,
    pub punctuation: Punctuation,
}

impl Default for TimestampConfiguration {
    fn default() -> Self {
        Self {
            granularity: TimestampsGranularity::Word,
            word_timestamps: WordTimestamps::On,
            diarization: Diarization::Off,
            punctuation: Punctuation::On,
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

    /// Filter timestamp data based on granularity setting
    pub fn filter_by_granularity(&mut self, granularity: TimestampsGranularity) {
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

    /// Filter word timestamps based on setting
    pub fn filter_by_word_timestamps(&mut self, setting: WordTimestamps) {
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
    pub fn filter_by_diarization(&mut self, setting: Diarization) {
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

    /// Add character alignment data
    pub fn add_character_alignments(&mut self, alignments: Vec<CharacterTimestamp>) {
        self.character_alignments.extend(alignments);
        self.generate_word_alignments();
    }

    /// Add an audio chunk with timing information
    pub fn add_chunk(&mut self, chunk: AudioChunkTimestamp) {
        self.audio_chunks.push(chunk);
        self.update_total_duration();
    }

    /// Generate word-level timestamps from character alignments
    pub fn generate_word_alignments(&mut self) {
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
        if !current_word.is_empty()
            && let Some(last_char) = self.character_alignments.last()
        {
            words.push(WordTimestamp {
                word: current_word,
                start_seconds: word_start_seconds,
                end_seconds: last_char.end_seconds,
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

    /// Finalize metadata when synthesis completes
    pub fn finalize(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let end_time = SystemTime::now();
        self.synthesis_end = Some(end_time);
        self.processing_time_ms = Some(
            end_time
                .duration_since(self.synthesis_start)
                .map_err(|e| format!("Time calculation error: {}", e))?
                .as_millis() as u64,
        );

        if let Some(last_chunk) = self.audio_chunks.last() {
            self.total_duration_ms = Some(last_chunk.end_ms);
        }

        Ok(())
    }

    /// Serialize to JSON
    pub fn to_json(&self) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        serde_json::to_string_pretty(self).map_err(Into::into)
    }

    /// Convert to SRT subtitle format
    pub fn to_srt(&self) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let mut srt = String::new();

        if let Some(words) = &self.word_alignments {
            for (i, word) in words.iter().enumerate() {
                let start_ms = (word.start_seconds * 1000.0) as u64;
                let end_ms = (word.end_seconds * 1000.0) as u64;

                srt.push_str(&format!("{}\n", i + 1));
                srt.push_str(&format!(
                    "{:02}:{:02}:{:02},{:03} --> {:02}:{:02}:{:02},{:03}\n",
                    start_ms / 3600000,
                    (start_ms % 3600000) / 60000,
                    (start_ms % 60000) / 1000,
                    start_ms % 1000,
                    end_ms / 3600000,
                    (end_ms % 3600000) / 60000,
                    (end_ms % 60000) / 1000,
                    end_ms % 1000
                ));
                srt.push_str(&format!("{}\n\n", word.word));
            }
        }

        Ok(srt)
    }

    /// Convert to WebVTT subtitle format
    pub fn to_vtt(&self) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let mut vtt = String::from("WEBVTT\n\n");

        if let Some(words) = &self.word_alignments {
            for word in words {
                let start_ms = (word.start_seconds * 1000.0) as u64;
                let end_ms = (word.end_seconds * 1000.0) as u64;

                vtt.push_str(&format!(
                    "{:02}:{:02}:{:02}.{:03} --> {:02}:{:02}:{:02}.{:03}\n",
                    start_ms / 3600000,
                    (start_ms % 3600000) / 60000,
                    (start_ms % 60000) / 1000,
                    start_ms % 1000,
                    end_ms / 3600000,
                    (end_ms % 3600000) / 60000,
                    (end_ms % 60000) / 1000,
                    end_ms % 1000
                ));
                vtt.push_str(&format!("{}\n\n", word.word));
            }
        }

        Ok(vtt)
    }
}

impl Default for TimestampMetadata {
    fn default() -> Self {
        Self::new()
    }
}
