//! ChunkHandler implementation and arrow syntax macros

use super::audio_chunk_wrapper::AudioChunkWrapper;
use super::builder_core::TtsConversationBuilderImpl;
use cyrup_sugars::prelude::ChunkHandler;
use fluent_voice_domain::VoiceError;
use futures::Stream;

impl<AudioStream> ChunkHandler<AudioChunkWrapper, VoiceError>
    for TtsConversationBuilderImpl<AudioStream>
where
    AudioStream: Stream<Item = fluent_voice_domain::AudioChunk> + Send + Unpin + 'static,
{
    fn on_chunk<F>(mut self, handler: F) -> Self
    where
        F: Fn(Result<AudioChunkWrapper, VoiceError>) -> AudioChunkWrapper + Send + Sync + 'static,
    {
        // Convert the handler to work with the underlying AudioChunk type
        let converted_handler =
            move |result: Result<fluent_voice_domain::AudioChunk, VoiceError>| {
                let wrapper_result = result.map(AudioChunkWrapper::from);
                let wrapper_output = handler(wrapper_result);
                wrapper_output.into()
            };
        self.chunk_handler = Some(Box::new(converted_handler));
        self
    }
}

/// Arrow syntax transformation macro for cyrup_sugars compatibility
#[macro_export]
macro_rules! arrow_transform {
    ($matcher:expr) => {
        |result| {
            // Enable JSON syntax transformation
            $crate::enable_json_syntax!();
            $matcher(result)
        }
    };
}
