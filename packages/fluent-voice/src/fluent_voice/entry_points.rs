//! Entry point structures for TTS and STT operations.

use super::default_tts_builder::DefaultTtsBuilder;
use crate::stt_conversation::{SttConversationBuilder, SttPostChunkBuilder};
use crate::tts_conversation::TtsConversationBuilder;
use fluent_voice_domain::{SpeechSource, VadMode, VoiceError};

/// Entry point for TTS operations providing .conversation() method
pub struct TtsEntry;

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
    pub fn listen<M, R>(self, matcher: M) -> R
    where
        M: FnOnce(Result<B::Conversation, VoiceError>) -> R + Send + 'static,
        R: futures_core::Stream<Item = fluent_voice_domain::TranscriptionSegmentImpl>
            + Send
            + Unpin
            + 'static,
    {
        self.builder.listen(matcher)
    }
}
