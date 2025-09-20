//! SttPostChunkBuilder trait implementation - Part 2

use super::conversation_impl::SttConversationImpl;
use super::microphone_builder::MicrophoneBuilderImpl;
use super::post_chunk_builder::SttPostChunkBuilderImpl;
use super::transcription_builder::TranscriptionBuilderImpl;
use fluent_voice_domain::{transcription::TranscriptionStream, VoiceError};

impl<S, F, T> crate::stt_conversation::SttPostChunkBuilder for SttPostChunkBuilderImpl<S, F, T>
where
    S: TranscriptionStream + 'static,
    F: FnMut(Result<T, VoiceError>) -> T + Send + 'static,
    T: fluent_voice_domain::transcription::TranscriptionSegment + Send + 'static,
{
    type Conversation = SttConversationImpl<S>;

    fn with_microphone(
        self,
        device: impl Into<String>,
    ) -> impl crate::stt_conversation::MicrophoneBuilder {
        let SttPostChunkBuilderImpl { base_builder, .. } = self;
        let super::conversation_builder_core::SttConversationBuilderImpl {
            vad_mode,
            noise_reduction,
            language_hint,
            diarization,
            word_timestamps,
            timestamps_granularity,
            punctuation,
            stream_fn,
            ..
        } = base_builder;

        MicrophoneBuilderImpl::new(
            device.into(),
            vad_mode,
            noise_reduction,
            language_hint,
            diarization,
            word_timestamps,
            timestamps_granularity,
            punctuation,
            stream_fn,
        )
    }

    fn transcribe(
        self,
        path: impl Into<String>,
    ) -> impl crate::stt_conversation::TranscriptionBuilder {
        let SttPostChunkBuilderImpl { base_builder, .. } = self;
        let super::conversation_builder_core::SttConversationBuilderImpl {
            vad_mode,
            noise_reduction,
            language_hint,
            diarization,
            word_timestamps,
            timestamps_granularity,
            punctuation,
            stream_fn,
            ..
        } = base_builder;

        TranscriptionBuilderImpl::new(
            path.into(),
            vad_mode,
            noise_reduction,
            language_hint,
            diarization,
            word_timestamps,
            timestamps_granularity,
            punctuation,
            stream_fn,
        )
    }

    fn listen<M, ST>(self, matcher: M) -> ST
    where
        M: FnOnce(Result<Self::Conversation, VoiceError>) -> ST + Send + 'static,
        ST: futures_core::Stream<Item = fluent_voice_domain::TranscriptionSegmentImpl>
            + Send
            + Unpin
            + 'static,
    {
        // Build the conversation result with chunk processor integration
        let stream_fn = self.base_builder.stream_fn;
        let conversation_result = Ok(SttConversationImpl {
            source: self.base_builder.source,
            vad_mode: self.base_builder.vad_mode,
            noise_reduction: self.base_builder.noise_reduction,
            language_hint: self.base_builder.language_hint,
            diarization: self.base_builder.diarization,
            word_timestamps: self.base_builder.word_timestamps,
            timestamps_granularity: self.base_builder.timestamps_granularity,
            punctuation: self.base_builder.punctuation,
            stream_fn,
        });

        // Apply the matcher closure to the conversation result
        // The matcher contains the JSON syntax transformed by listen! macro
        matcher(conversation_result)
    }
}
