//! ChunkHandler trait implementation for STT conversation builder

use super::conversation_builder_core::SttConversationBuilderImpl;
use super::transcription_segment_wrapper::TranscriptionSegmentWrapper;
use cyrup_sugars::prelude::ChunkHandler;
use fluent_voice_domain::{transcription::TranscriptionStream, VoiceError};

/// ChunkHandler implementation for STT conversation builder
impl<S> ChunkHandler<TranscriptionSegmentWrapper, VoiceError> for SttConversationBuilderImpl<S>
where
    S: TranscriptionStream + 'static,
{
    fn on_chunk<F>(mut self, handler: F) -> Self
    where
        F: Fn(Result<TranscriptionSegmentWrapper, VoiceError>) -> TranscriptionSegmentWrapper
            + Send
            + Sync
            + 'static,
    {
        // Convert the handler to work with the underlying TranscriptionSegmentImpl type
        let converted_handler =
            move |result: Result<fluent_voice_domain::TranscriptionSegmentImpl, VoiceError>| {
                let wrapper_result = result.map(TranscriptionSegmentWrapper::from);
                let wrapper_output = handler(wrapper_result);
                wrapper_output.into()
            };
        self.chunk_handler = Some(Box::new(converted_handler));
        self
    }
}
