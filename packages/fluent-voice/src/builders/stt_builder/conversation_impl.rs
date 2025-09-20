//! SttConversationImpl struct and SttConversation trait implementation

use fluent_voice_domain::{
    language::Language,
    noise_reduction::NoiseReduction,
    speech_source::SpeechSource,
    timestamps::{Diarization, Punctuation, TimestampsGranularity, WordTimestamps},
    transcription::TranscriptionStream,
    vad_mode::VadMode,
};

/// Base STT conversation session implementation.
pub struct SttConversationImpl<S> {
    /// Audio source configuration
    pub source: Option<SpeechSource>,
    /// Voice activity detection mode
    pub vad_mode: Option<VadMode>,
    /// Noise reduction level
    pub noise_reduction: Option<NoiseReduction>,
    /// Language hint for recognition
    pub language_hint: Option<Language>,
    /// Speaker diarization setting
    pub diarization: Option<Diarization>,
    /// Word-level timestamp setting
    pub word_timestamps: Option<WordTimestamps>,
    /// Timestamp granularity setting
    pub timestamps_granularity: Option<TimestampsGranularity>,
    /// Punctuation setting
    pub punctuation: Option<Punctuation>,
    /// Function to convert configuration to transcript stream
    pub(super) stream_fn: Box<
        dyn FnOnce(
                Option<SpeechSource>,
                Option<VadMode>,
                Option<NoiseReduction>,
                Option<Language>,
                Option<Diarization>,
                Option<WordTimestamps>,
                Option<TimestampsGranularity>,
                Option<Punctuation>,
            ) -> S
            + Send,
    >,
}

impl<S> crate::stt_conversation::SttConversation for SttConversationImpl<S>
where
    S: TranscriptionStream,
{
    type Stream = S;

    fn into_stream(self) -> Self::Stream {
        (self.stream_fn)(
            self.source,
            self.vad_mode,
            self.noise_reduction,
            self.language_hint,
            self.diarization,
            self.word_timestamps,
            self.timestamps_granularity,
            self.punctuation,
        )
    }
}
