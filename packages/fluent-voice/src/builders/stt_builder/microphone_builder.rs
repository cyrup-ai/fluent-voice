//! Microphone-specific STT builder implementation - Part 1

use fluent_voice_domain::{
    language::Language,
    noise_reduction::NoiseReduction,
    speech_source::SpeechSource,
    timestamps::{Diarization, Punctuation, TimestampsGranularity, WordTimestamps},
    transcription::TranscriptionStream,
    vad_mode::VadMode,
};

/// Microphone-specific STT builder implementation.
pub struct MicrophoneBuilderImpl<S> {
    /// Device identifier
    pub(super) device: String,
    /// Voice activity detection mode
    pub(super) vad_mode: Option<VadMode>,
    /// Noise reduction level
    pub(super) noise_reduction: Option<NoiseReduction>,
    /// Language hint for recognition
    pub(super) language_hint: Option<Language>,
    /// Speaker diarization setting
    pub(super) diarization: Option<Diarization>,
    /// Word-level timestamp setting
    pub(super) word_timestamps: Option<WordTimestamps>,
    /// Timestamp granularity setting
    pub(super) timestamps_granularity: Option<TimestampsGranularity>,
    /// Punctuation setting
    pub(super) punctuation: Option<Punctuation>,
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

impl<S> MicrophoneBuilderImpl<S>
where
    S: TranscriptionStream,
{
    /// Create a new microphone builder.
    pub fn new<F>(
        device: String,
        vad_mode: Option<VadMode>,
        noise_reduction: Option<NoiseReduction>,
        language_hint: Option<Language>,
        diarization: Option<Diarization>,
        word_timestamps: Option<WordTimestamps>,
        timestamps_granularity: Option<TimestampsGranularity>,
        punctuation: Option<Punctuation>,
        stream_fn: F,
    ) -> Self
    where
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
        Self {
            device,
            vad_mode,
            noise_reduction,
            language_hint,
            diarization,
            word_timestamps,
            timestamps_granularity,
            punctuation,
            stream_fn: Box::new(stream_fn),
        }
    }
}
