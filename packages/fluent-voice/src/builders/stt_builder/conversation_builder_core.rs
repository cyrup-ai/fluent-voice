//! Core SttConversationBuilderImpl struct and methods

use super::transcript_impl::TranscriptImpl;
use super::transcription_builder::TranscriptionBuilderImpl;
use core::future::Future;
use fluent_voice_domain::{
    language::Language,
    noise_reduction::NoiseReduction,
    speech_source::SpeechSource,
    timestamps::{Diarization, Punctuation, TimestampsGranularity, WordTimestamps},
    transcription::TranscriptionStream,
    vad_mode::VadMode,
    VoiceError,
};

/// Base STT conversation builder implementation.
pub struct SttConversationBuilderImpl<S> {
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
    /// Prediction callback
    pub prediction_processor: Option<Box<dyn FnMut(String, String) + Send + 'static>>,
    /// Engine configuration parameters
    pub engine_config: std::collections::HashMap<String, String>,
    /// ChunkHandler for Result<TranscriptionSegmentImpl, VoiceError> -> TranscriptionSegmentImpl conversion
    pub chunk_handler: Option<
        Box<
            dyn Fn(
                    Result<fluent_voice_domain::TranscriptionSegmentImpl, VoiceError>,
                ) -> fluent_voice_domain::TranscriptionSegmentImpl
                + Send
                + Sync
                + 'static,
        >,
    >,
    /// Function to convert configuration to transcript stream
    pub stream_fn: Box<
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

impl<S> SttConversationBuilderImpl<S>
where
    S: TranscriptionStream + 'static,
{
    /// Create a new STT conversation builder with a custom processing function.
    pub fn new<F>(stream_fn: F) -> Self
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
            source: None,
            vad_mode: None,
            noise_reduction: None,
            language_hint: None,
            diarization: None,
            word_timestamps: None,
            timestamps_granularity: None,
            punctuation: None,
            prediction_processor: None,
            engine_config: std::collections::HashMap::new(),
            chunk_handler: None,
            stream_fn: Box::new(stream_fn),
        }
    }

    /// Convenience method to collect all transcript segments into a complete transcript.
    ///
    /// This method matches the example in the README.md and is equivalent to
    /// calling `.transcribe()` followed by `.collect()`.
    pub fn collect(self) -> impl Future<Output = Result<TranscriptImpl<S>, VoiceError>> + Send {
        // Convert to transcription builder with an empty path since we already have the source
        let path = "".to_string();
        let transcription_builder = TranscriptionBuilderImpl::new(
            path,
            self.vad_mode,
            self.noise_reduction,
            self.language_hint,
            self.diarization,
            self.word_timestamps,
            self.timestamps_granularity,
            self.punctuation,
            self.stream_fn,
        );

        // Use the existing collect method on TranscriptionBuilder
        async move { transcription_builder.create_transcript().await }
    }

    /// Convert to post-chunk builder for advanced chunk processing
    pub fn post_chunk<F, T>(
        self,
        chunk_processor: F,
    ) -> super::post_chunk_builder::SttPostChunkBuilderImpl<S, F, T>
    where
        F: FnMut(Result<T, VoiceError>) -> T + Send + 'static,
        T: fluent_voice_domain::transcription::TranscriptionSegment + Send + 'static,
    {
        super::post_chunk_builder::SttPostChunkBuilderImpl::new(self, chunk_processor)
    }
}
