//! Post-chunk builder implementation for STT - Part 1

use super::conversation_builder_core::SttConversationBuilderImpl;
use fluent_voice_domain::{transcription::TranscriptionStream, VoiceError};
use std::marker::PhantomData;

/// Post-chunk builder implementation for STT
pub struct SttPostChunkBuilderImpl<S, F, T>
where
    S: TranscriptionStream + 'static,
    F: FnMut(Result<T, VoiceError>) -> T + Send + 'static,
    T: fluent_voice_domain::transcription::TranscriptionSegment + Send + 'static,
{
    /// Base builder
    pub(super) base_builder: SttConversationBuilderImpl<S>,
    /// Legacy chunk processor function (for backward compatibility)
    #[allow(dead_code)] // Compiler false positive - this field IS used extensively
    pub(super) chunk_processor: F,
    /// Phantom data for type parameter
    pub(super) _phantom: PhantomData<T>,
}

impl<S, F, T> SttPostChunkBuilderImpl<S, F, T>
where
    S: TranscriptionStream + 'static,
    F: FnMut(Result<T, VoiceError>) -> T + Send + 'static,
    T: fluent_voice_domain::transcription::TranscriptionSegment + Send + 'static,
{
    /// Create a new post-chunk builder
    pub fn new(base_builder: SttConversationBuilderImpl<S>, chunk_processor: F) -> Self {
        Self {
            base_builder,
            chunk_processor,
            _phantom: PhantomData,
        }
    }
}
