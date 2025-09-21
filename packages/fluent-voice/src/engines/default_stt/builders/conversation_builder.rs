//! STT Conversation Builder for Default Engine
//!
//! Provides the conversation builder implementation that creates conversations
//! using the default STT engine components.

use super::super::config::{VadConfig, WakeWordConfig};
use super::super::conversation::DefaultSTTConversation;
use super::super::types::SendableClosure;
use crate::stt_conversation::SttConversationBuilder;
use fluent_voice_domain::{
    Diarization, Language, NoiseReduction, Punctuation, SpeechSource, TimestampsGranularity,
    TranscriptionSegmentImpl, VadMode, VoiceError, WordTimestamps,
};

/// Default STT Conversation Builder
///
/// Builds conversations using the default STT engine with VAD, wake word detection,
/// and Whisper transcription capabilities.
pub struct DefaultSTTConversationBuilder {
    /// VAD configuration for voice activity detection
    pub vad_config: VadConfig,
    /// Wake word configuration for activation detection
    pub wake_word_config: WakeWordConfig,
    /// Audio source configuration
    pub speech_source: Option<SpeechSource>,
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
    /// Error handler with SendableClosure wrapper
    pub error_handler:
        Option<SendableClosure<Box<dyn FnMut(VoiceError) -> String + Send + 'static>>>,
    /// Wake word handler with SendableClosure wrapper
    pub wake_handler: Option<SendableClosure<Box<dyn FnMut(String) + Send + 'static>>>,
    /// Turn detection handler with SendableClosure wrapper
    pub turn_handler:
        Option<SendableClosure<Box<dyn FnMut(Option<String>, String) + Send + 'static>>>,
    /// Prediction processor with SendableClosure wrapper
    pub prediction_processor:
        Option<SendableClosure<Box<dyn FnMut(String, String) + Send + 'static>>>,
    /// Chunk handler with SendableClosure wrapper for real-time transcription processing
    pub chunk_handler: Option<
        SendableClosure<
            Box<
                dyn FnMut(Result<TranscriptionSegmentImpl, VoiceError>) -> TranscriptionSegmentImpl
                    + Send
                    + 'static,
            >,
        >,
    >,
}

impl Default for DefaultSTTConversationBuilder {
    fn default() -> Self {
        Self {
            vad_config: VadConfig::default(),
            wake_word_config: WakeWordConfig::default(),
            speech_source: None,
            vad_mode: Some(VadMode::Accurate),
            noise_reduction: Some(NoiseReduction::Low),
            language_hint: Some(Language::ENGLISH_US),
            diarization: Some(Diarization::Off),
            word_timestamps: Some(WordTimestamps::Off),
            timestamps_granularity: Some(TimestampsGranularity::Word),
            punctuation: Some(Punctuation::On),
            error_handler: None,
            wake_handler: None,
            turn_handler: None,
            prediction_processor: None,
            chunk_handler: None,
        }
    }
}

impl DefaultSTTConversationBuilder {
    pub fn new() -> Self {
        Self::default()
    }
}

impl SttConversationBuilder for DefaultSTTConversationBuilder {
    type Conversation = DefaultSTTConversation;

    fn with_source(mut self, source: SpeechSource) -> Self {
        self.speech_source = Some(source);
        self
    }

    fn vad_mode(mut self, mode: VadMode) -> Self {
        self.vad_mode = Some(mode);
        self
    }

    fn noise_reduction(mut self, level: NoiseReduction) -> Self {
        self.noise_reduction = Some(level);
        self
    }

    fn language_hint(mut self, lang: Language) -> Self {
        self.language_hint = Some(lang);
        self
    }

    fn diarization(mut self, d: Diarization) -> Self {
        self.diarization = Some(d);
        self
    }

    fn word_timestamps(mut self, w: WordTimestamps) -> Self {
        self.word_timestamps = Some(w);
        self
    }

    fn timestamps_granularity(mut self, g: TimestampsGranularity) -> Self {
        self.timestamps_granularity = Some(g);
        self
    }

    fn punctuation(mut self, p: Punctuation) -> Self {
        self.punctuation = Some(p);
        self
    }

    fn on_prediction<F>(mut self, f: F) -> Self
    where
        F: FnMut(String, String) + Send + 'static,
    {
        self.prediction_processor = Some(SendableClosure(Box::new(f)));
        self
    }

    fn on_chunk<F>(self, f: F) -> impl crate::stt_conversation::SttPostChunkBuilder<Conversation = <Self as crate::stt_conversation::SttConversationBuilder>::Conversation>
    where
        F: FnMut(Result<TranscriptionSegmentImpl, VoiceError>) -> TranscriptionSegmentImpl
            + Send
            + 'static,
    {
        // Pass the closure to the post-chunk builder which connects to working DefaultSTTConversation
        crate::stt_conversation::SttPostChunkBuilderImpl::new(self, Box::new(f))
    }

    fn on_result<F>(mut self, f: F) -> Self
    where
        F: FnMut(VoiceError) -> String + Send + 'static,
    {
        self.error_handler = Some(SendableClosure(Box::new(f)));
        self
    }

    fn on_wake<F>(mut self, f: F) -> Self
    where
        F: FnMut(String) + Send + 'static,
    {
        self.wake_handler = Some(SendableClosure(Box::new(f)));
        self
    }

    fn on_turn_detected<F>(mut self, f: F) -> Self
    where
        F: FnMut(Option<String>, String) + Send + 'static,
    {
        self.turn_handler = Some(SendableClosure(Box::new(f)));
        self
    }
}

impl DefaultSTTConversationBuilder {
    /// Build a real conversation with chunk processor from post-chunk builder
    pub fn build_real_conversation_with_chunk_processor(
        mut self,
        chunk_processor: Box<
            dyn FnMut(Result<TranscriptionSegmentImpl, VoiceError>) -> TranscriptionSegmentImpl
                + Send
                + 'static,
        >,
    ) -> Result<DefaultSTTConversation, VoiceError> {
        // Set the chunk processor from the post-chunk builder
        self.chunk_handler = Some(SendableClosure(chunk_processor));

        // Create real conversation using the existing new_from_builder method
        DefaultSTTConversation::new_from_builder(self)
    }
}
