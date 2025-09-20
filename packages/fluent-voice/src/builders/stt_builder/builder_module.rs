//! Builder module with factory functions

use super::conversation_builder_core::SttConversationBuilderImpl;
use fluent_voice_domain::{
    language::Language,
    noise_reduction::NoiseReduction,
    speech_source::SpeechSource,
    timestamps::{Diarization, Punctuation, TimestampsGranularity, WordTimestamps},
    transcription::TranscriptionStream,
    vad_mode::VadMode,
};

/// Create a new STT conversation builder
pub fn stt_conversation_builder<S, F>(stream_fn: F) -> SttConversationBuilderImpl<S>
where
    S: TranscriptionStream + 'static,
    F: FnOnce(
            Option<SpeechSource>,
            Option<VadMode>,
            Option<NoiseReduction>,
            Option<Language>,
            Option<Diarization>,
            Option<WordTimestamps>,
            Option<TimestampsGranularity>,
            Option<Punctuation>,
        ) -> S
        + Send
        + 'static,
{
    SttConversationBuilderImpl::new(stream_fn)
}
