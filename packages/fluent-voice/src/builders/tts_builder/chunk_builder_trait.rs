//! TtsConversationChunkBuilder trait implementation - Part 1

use super::builder_core::TtsConversationBuilderImpl;
use super::conversation_impl::TtsConversationImpl;
use crate::tts_conversation::TtsConversationChunkBuilder;
use fluent_voice_domain::VoiceError;
use futures::Stream;

impl<AudioStream> TtsConversationBuilderImpl<AudioStream> where
    AudioStream: Stream<Item = fluent_voice_domain::AudioChunk> + Send + Unpin + 'static
{
}

impl<AudioStream> TtsConversationChunkBuilder for TtsConversationBuilderImpl<AudioStream>
where
    AudioStream: Stream<Item = fluent_voice_domain::AudioChunk> + Send + Unpin + 'static,
{
    type Conversation = TtsConversationImpl<AudioStream>;

    fn synthesize(
        self,
    ) -> impl futures_core::Stream<Item = fluent_voice_domain::AudioChunk> + Send + Unpin {
        let lines = self.lines;
        let chunk_handler = self.chunk_handler;
        let voice_clone_path = self.voice_clone_path;
        let prelude = self.prelude;
        let postlude = self.postlude;
        let engine_config = self.engine_config;

        // Create stream using tokio_stream for proper async handling
        use tokio::sync::mpsc;
        use tokio_stream::wrappers::UnboundedReceiverStream;

        let (tx, rx) =
            mpsc::unbounded_channel::<Result<fluent_voice_domain::AudioChunk, VoiceError>>();

        // Spawn async task to generate audio using dia-voice
        tokio::spawn(async move {
            super::chunk_synthesis::process_synthesis_task(
                tx,
                lines,
                voice_clone_path,
                prelude,
                postlude,
                engine_config,
            )
            .await;
        });

        // Apply ChunkHandler to transform Result<AudioChunk, VoiceError> to AudioChunk
        use futures::StreamExt;
        let stream = UnboundedReceiverStream::new(rx).map(move |result| {
            if let Some(ref handler) = chunk_handler {
                handler(result)
            } else {
                // Default behavior: unwrap Ok, create bad_chunk for Err
                result.unwrap_or_else(|e| {
                    // Create error chunk using the constructor
                    fluent_voice_domain::AudioChunk::with_metadata(
                        Vec::new(),
                        0,
                        0,
                        None,
                        Some(format!("[ERROR] {}", e)),
                        None,
                    )
                })
            }
        });

        // Return AudioStream wrapper to enable .play() method
        let stream = Box::pin(stream);
        crate::audio_stream::AudioStream::new(stream)
    }
}
