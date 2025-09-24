//! Entry point structures for TTS and STT operations.

use super::default_tts_builder::DefaultTtsBuilder;
use crate::stt_conversation::{SttConversationBuilder, SttPostChunkBuilder};
use crate::tts_conversation::TtsConversationBuilder;
use fluent_voice_domain::{SpeechSource, VadMode, VoiceError};

/// Entry point for TTS operations providing .conversation() method
pub struct TtsEntry;

impl Default for TtsEntry {
    fn default() -> Self {
        Self::new()
    }
}

impl TtsEntry {
    pub fn new() -> Self {
        Self
    }

    /// Create a new TTS conversation builder
    pub fn conversation(self) -> impl TtsConversationBuilder {
        DefaultTtsBuilder::new()
    }
}

/// Entry point for STT operations providing .conversation() method  
pub struct SttEntry;

impl Default for SttEntry {
    fn default() -> Self {
        Self::new()
    }
}

impl SttEntry {
    pub fn new() -> Self {
        Self
    }

    /// Create a new STT conversation builder
    pub fn conversation(self) -> impl SttConversationBuilder {
        crate::engines::DefaultSTTConversationBuilder::new()
    }

    /// Delegate method for with_source - forwards to conversation builder
    pub fn with_source(self, src: SpeechSource) -> impl SttConversationBuilder {
        self.conversation().with_source(src)
    }

    /// Delegate method for vad_mode - forwards to conversation builder
    pub fn vad_mode(self, mode: VadMode) -> impl SttConversationBuilder {
        self.conversation().vad_mode(mode)
    }

    /// Delegate method for on_prediction - forwards to conversation builder
    pub fn on_prediction<F>(self, f: F) -> impl SttConversationBuilder
    where
        F: FnMut(String, String) + Send + 'static,
    {
        self.conversation().on_prediction(f)
    }

    /// Delegate method for noise_reduction - forwards to conversation builder
    pub fn noise_reduction(self, level: fluent_voice_domain::NoiseReduction) -> impl SttConversationBuilder {
        self.conversation().noise_reduction(level)
    }

    /// Delegate method for language_hint - forwards to conversation builder
    pub fn language_hint(self, lang: fluent_voice_domain::Language) -> impl SttConversationBuilder {
        self.conversation().language_hint(lang)
    }

    /// Delegate method for diarization - forwards to conversation builder
    pub fn diarization(self, d: fluent_voice_domain::Diarization) -> impl SttConversationBuilder {
        self.conversation().diarization(d)
    }

    /// Delegate method for word_timestamps - forwards to conversation builder
    pub fn word_timestamps(self, w: fluent_voice_domain::WordTimestamps) -> impl SttConversationBuilder {
        self.conversation().word_timestamps(w)
    }

    /// Delegate method for timestamps_granularity - forwards to conversation builder
    pub fn timestamps_granularity(self, g: fluent_voice_domain::TimestampsGranularity) -> impl SttConversationBuilder {
        self.conversation().timestamps_granularity(g)
    }

    /// Delegate method for punctuation - forwards to conversation builder
    pub fn punctuation(self, p: fluent_voice_domain::Punctuation) -> impl SttConversationBuilder {
        self.conversation().punctuation(p)
    }

    /// Delegate method for on_chunk - forwards to conversation builder
    pub fn on_chunk<F>(self, f: F) -> impl SttPostChunkBuilder
    where
        F: FnMut(
                Result<fluent_voice_domain::TranscriptionSegmentImpl, VoiceError>,
            ) -> fluent_voice_domain::TranscriptionSegmentImpl
            + Send
            + 'static,
    {
        self.conversation().on_chunk(f)
    }
}

/// Post-chunk entry point that provides listen() method
pub struct SttPostChunkEntry<B> {
    builder: B,
}

impl<B> SttPostChunkEntry<B>
where
    B: SttPostChunkBuilder,
{
    /// Start listening for transcription
    pub fn listen<M>(self, matcher: M) -> cyrup_sugars::prelude::AsyncStream<fluent_voice_domain::TranscriptionSegmentImpl>
    where
        M: FnOnce(Result<B::Conversation, VoiceError>) -> cyrup_sugars::prelude::AsyncStream<fluent_voice_domain::TranscriptionSegmentImpl> + Send + 'static,
    {
        self.builder.listen(matcher)
    }
}
