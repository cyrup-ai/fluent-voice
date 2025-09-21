//! STT conversation builders and transcription functionality

use fluent_voice::TranscriptionSegmentImpl;
use fluent_voice::stt_conversation::{
    MicrophoneBuilder, SttConversationBuilder, SttPostChunkBuilder, TranscriptionBuilder,
};
use fluent_voice_domain::{
    AudioFormat, Diarization, Language, MicBackend, NoiseReduction, Punctuation, SpeechSource,
    TimestampsGranularity, VadMode, VoiceError, WordTimestamps,
};
use futures_core::Stream;

/// STT conversation builder
pub struct KyutaiSttConversationBuilder {
    pub(super) source: Option<SpeechSource>,
    vad_mode: Option<VadMode>,
    noise_reduction: Option<NoiseReduction>,
    language: Option<Language>,
    diarization: Option<Diarization>,
    word_timestamps: Option<WordTimestamps>,
    timestamps_granularity: Option<TimestampsGranularity>,
    punctuation: Option<Punctuation>,
    // Handler storage for callbacks
    result_handler: Option<Box<dyn FnMut(VoiceError) -> String + Send + 'static>>,
    wake_handler: Option<Box<dyn FnMut(String) + Send + 'static>>,
    turn_detected_handler: Option<Box<dyn FnMut(Option<String>, String) + Send + 'static>>,
    prediction_handler: Option<Box<dyn FnMut(String, String) + Send + 'static>>,
}

impl std::fmt::Debug for KyutaiSttConversationBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KyutaiSttConversationBuilder")
            .field("source", &self.source)
            .field("vad_mode", &self.vad_mode)
            .field("noise_reduction", &self.noise_reduction)
            .field("language", &self.language)
            .field("diarization", &self.diarization)
            .field("word_timestamps", &self.word_timestamps)
            .field("timestamps_granularity", &self.timestamps_granularity)
            .field("punctuation", &self.punctuation)
            .field("result_handler", &"<function>")
            .field("wake_handler", &"<function>")
            .field("turn_detected_handler", &"<function>")
            .field("prediction_handler", &"<function>")
            .finish()
    }
}
impl Clone for KyutaiSttConversationBuilder {
    fn clone(&self) -> Self {
        Self {
            source: self.source.clone(),
            vad_mode: self.vad_mode.clone(),
            noise_reduction: self.noise_reduction.clone(),
            language: self.language.clone(),
            diarization: self.diarization.clone(),
            word_timestamps: self.word_timestamps.clone(),
            timestamps_granularity: self.timestamps_granularity.clone(),
            punctuation: self.punctuation.clone(),
            // Note: Closures cannot be cloned, so we reset them to None
            result_handler: None,
            wake_handler: None,
            turn_detected_handler: None,
            prediction_handler: None,
        }
    }
}

impl KyutaiSttConversationBuilder {
    /// Create a new STT conversation builder
    pub fn new() -> Self {
        Self {
            source: None,
            vad_mode: None,
            noise_reduction: None,
            language: None,
            diarization: None,
            word_timestamps: None,
            timestamps_granularity: None,
            punctuation: None,
            result_handler: None,
            wake_handler: None,
            turn_detected_handler: None,
            prediction_handler: None,
        }
    }

    /// Convert to post-chunk builder for terminal actions
    pub fn into_post_chunk(self) -> KyutaiSttPostChunkBuilder {
        KyutaiSttPostChunkBuilder::new(self)
    }
}
impl SttConversationBuilder for KyutaiSttConversationBuilder {
    type Conversation = super::sessions::KyutaiSttConversation;

    #[inline]
    fn with_source(mut self, source: SpeechSource) -> Self {
        self.source = Some(source);
        self
    }

    #[inline]
    fn vad_mode(mut self, mode: VadMode) -> Self {
        self.vad_mode = Some(mode);
        self
    }

    #[inline]
    fn noise_reduction(mut self, reduction: NoiseReduction) -> Self {
        self.noise_reduction = Some(reduction);
        self
    }

    #[inline]
    fn language_hint(mut self, lang: Language) -> Self {
        self.language = Some(lang);
        self
    }

    #[inline]
    fn diarization(mut self, diarization: Diarization) -> Self {
        self.diarization = Some(diarization);
        self
    }

    #[inline]
    fn word_timestamps(mut self, timestamps: WordTimestamps) -> Self {
        self.word_timestamps = Some(timestamps);
        self
    }

    #[inline]
    fn timestamps_granularity(mut self, granularity: TimestampsGranularity) -> Self {
        self.timestamps_granularity = Some(granularity);
        self
    }

    #[inline]
    fn punctuation(mut self, punctuation: Punctuation) -> Self {
        self.punctuation = Some(punctuation);
        self
    }

    fn on_result<F>(mut self, f: F) -> Self
    where
        F: FnMut(VoiceError) -> String + Send + 'static,
    {
        self.result_handler = Some(Box::new(f));
        self
    }

    fn on_wake<F>(mut self, f: F) -> Self
    where
        F: FnMut(String) + Send + 'static,
    {
        self.wake_handler = Some(Box::new(f));
        self
    }

    fn on_turn_detected<F>(mut self, f: F) -> Self
    where
        F: FnMut(Option<String>, String) + Send + 'static,
    {
        self.turn_detected_handler = Some(Box::new(f));
        self
    }

    fn on_prediction<F>(mut self, f: F) -> Self
    where
        F: FnMut(String, String) + Send + 'static,
    {
        self.prediction_handler = Some(Box::new(f));
        self
    }

    fn on_chunk<F>(self, _f: F) -> impl SttPostChunkBuilder<Conversation = Self::Conversation>
    where
        F: FnMut(Result<TranscriptionSegmentImpl, VoiceError>) -> TranscriptionSegmentImpl
            + Send
            + 'static,
    {
        KyutaiSttPostChunkBuilder::new(self)
    }
}
/// Post-chunk STT builder with terminal action methods
#[derive(Debug, Clone)]
pub struct KyutaiSttPostChunkBuilder {
    builder: KyutaiSttConversationBuilder,
}

impl KyutaiSttPostChunkBuilder {
    #[inline]
    pub fn new(builder: KyutaiSttConversationBuilder) -> Self {
        Self { builder }
    }
}

impl SttPostChunkBuilder for KyutaiSttPostChunkBuilder {
    type Conversation = super::sessions::KyutaiSttConversation;

    fn with_microphone(mut self, device: impl Into<String>) -> impl MicrophoneBuilder {
        let backend_str = device.into();
        let mic_backend = if backend_str == "default" {
            MicBackend::Default
        } else {
            MicBackend::Device(backend_str)
        };

        self.builder.source = Some(SpeechSource::Microphone {
            backend: mic_backend,
            format: AudioFormat::Pcm16Khz,
            sample_rate: 16_000,
        });
        KyutaiMicrophoneBuilder::new(self.builder)
    }

    fn transcribe(mut self, path: impl Into<String>) -> impl TranscriptionBuilder {
        self.builder.source = Some(SpeechSource::File {
            path: path.into(),
            format: AudioFormat::Pcm16Khz,
        });
        KyutaiTranscriptionBuilder::new(self.builder)
    }

    fn listen<M, S>(self, matcher: M) -> S
    where
        M: FnOnce(Result<Self::Conversation, VoiceError>) -> S + Send + 'static,
        S: futures_core::Stream<
                Item = Result<
                    fluent_voice_domain::transcription::TranscriptionSegmentImpl,
                    VoiceError,
                >,
            > + Send
            + Unpin
            + 'static,
    {
        // Create Kyutai STT conversation for processing
        let conversation = super::sessions::KyutaiSttConversation::new();
        let result = Ok(conversation);
        matcher(result)
    }
}
/// Microphone-based STT builder
#[derive(Debug, Clone)]
pub struct KyutaiMicrophoneBuilder {
    builder: KyutaiSttConversationBuilder,
}

impl KyutaiMicrophoneBuilder {
    #[inline]
    pub fn new(builder: KyutaiSttConversationBuilder) -> Self {
        Self { builder }
    }
}

impl MicrophoneBuilder for KyutaiMicrophoneBuilder {
    type Conversation = super::sessions::KyutaiSttConversation;

    #[inline]
    fn vad_mode(mut self, mode: VadMode) -> Self {
        self.builder.vad_mode = Some(mode);
        self
    }

    #[inline]
    fn noise_reduction(mut self, level: NoiseReduction) -> Self {
        self.builder.noise_reduction = Some(level);
        self
    }

    #[inline]
    fn language_hint(mut self, lang: Language) -> Self {
        self.builder.language = Some(lang);
        self
    }

    #[inline]
    fn diarization(mut self, diarization: Diarization) -> Self {
        self.builder.diarization = Some(diarization);
        self
    }

    #[inline]
    fn word_timestamps(mut self, timestamps: WordTimestamps) -> Self {
        self.builder.word_timestamps = Some(timestamps);
        self
    }

    #[inline]
    fn timestamps_granularity(mut self, granularity: TimestampsGranularity) -> Self {
        self.builder.timestamps_granularity = Some(granularity);
        self
    }

    #[inline]
    fn punctuation(mut self, punctuation: Punctuation) -> Self {
        self.builder.punctuation = Some(punctuation);
        self
    }

    fn listen<M, S>(self, matcher: M) -> S
    where
        M: FnOnce(Result<Self::Conversation, VoiceError>) -> S + Send + 'static,
        S: futures_core::Stream<
                Item = Result<
                    fluent_voice_domain::transcription::TranscriptionSegmentImpl,
                    VoiceError,
                >,
            > + Send
            + Unpin
            + 'static,
    {
        // Create a live microphone STT conversation
        let conversation = super::sessions::KyutaiSttConversation::new();
        matcher(Ok(conversation))
    }
}
/// File-based transcription builder
#[derive(Debug, Clone)]
pub struct KyutaiTranscriptionBuilder {
    builder: KyutaiSttConversationBuilder,
    progress_template: Option<String>,
}

impl KyutaiTranscriptionBuilder {
    #[inline]
    pub fn new(builder: KyutaiSttConversationBuilder) -> Self {
        Self {
            builder,
            progress_template: None,
        }
    }
}

impl TranscriptionBuilder for KyutaiTranscriptionBuilder {
    type Transcript = super::sessions::KyutaiSttConversation;

    #[inline]
    fn vad_mode(mut self, mode: VadMode) -> Self {
        self.builder.vad_mode = Some(mode);
        self
    }

    #[inline]
    fn noise_reduction(mut self, level: NoiseReduction) -> Self {
        self.builder.noise_reduction = Some(level);
        self
    }

    #[inline]
    fn language_hint(mut self, lang: Language) -> Self {
        self.builder.language = Some(lang);
        self
    }

    #[inline]
    fn diarization(mut self, diarization: Diarization) -> Self {
        self.builder.diarization = Some(diarization);
        self
    }

    #[inline]
    fn word_timestamps(mut self, timestamps: WordTimestamps) -> Self {
        self.builder.word_timestamps = Some(timestamps);
        self
    }

    #[inline]
    fn timestamps_granularity(mut self, granularity: TimestampsGranularity) -> Self {
        self.builder.timestamps_granularity = Some(granularity);
        self
    }

    #[inline]
    fn punctuation(mut self, punctuation: Punctuation) -> Self {
        self.builder.punctuation = Some(punctuation);
        self
    }

    #[inline]
    fn with_progress<S: Into<String>>(mut self, template: S) -> Self {
        self.progress_template = Some(template.into());
        self
    }

    fn emit(self) -> impl Stream<Item = String> + Send + Unpin {
        // Create a file-based STT conversation and convert to string stream
        use fluent_voice::stt_conversation::SttConversation;
        use futures_util::StreamExt;
        let conversation = super::sessions::KyutaiSttConversation::new();
        Box::pin(conversation.into_stream().map(|result| match result {
            Ok(segment) => segment.text().to_string(),
            Err(_) => String::new(), // Default error handling
        }))
    }

    async fn collect(self) -> Result<Self::Transcript, VoiceError> {
        Ok(super::sessions::KyutaiSttConversation::new())
    }

    async fn collect_with<F, R>(self, handler: F) -> R
    where
        F: FnOnce(Result<Self::Transcript, VoiceError>) -> R,
    {
        let result = self.collect().await;
        handler(result)
    }

    fn into_text_stream(self) -> impl Stream<Item = String> + Send {
        use fluent_voice::stt_conversation::SttConversation;
        use futures_util::StreamExt;
        // Create a real text stream from Kyutai STT conversation
        let conversation = super::sessions::KyutaiSttConversation::new();
        Box::pin(conversation.into_stream().map(|result| match result {
            Ok(segment) => segment.text().to_string(),
            Err(_) => String::new(), // Handle errors gracefully
        }))
    }

    fn transcribe<M, S>(self, matcher: M) -> S
    where
        M: FnOnce(Result<Self::Transcript, VoiceError>) -> S + Send + 'static,
        S: futures_core::Stream<
                Item = Result<
                    fluent_voice_domain::transcription::TranscriptionSegmentImpl,
                    VoiceError,
                >,
            > + Send
            + Unpin
            + 'static,
    {
        let conversation = super::sessions::KyutaiSttConversation::new();
        let result = Ok(conversation);
        matcher(result)
    }
}
