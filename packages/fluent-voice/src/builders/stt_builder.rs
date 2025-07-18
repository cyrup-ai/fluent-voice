//! Concrete STT builder implementation
//!
//! This module provides a non-macro implementation of the STT conversation builders
//! that can be used as a base for engine-specific implementations.

use crate::audio_chunk::transcript_stream_to_string_stream;
use crate::stt_conversation::TranscriptionBuilder;
use core::future::Future;
use fluent_voice_domain::VoiceError;
use fluent_voice_domain::{
    language::Language,
    noise_reduction::NoiseReduction,
    speech_source::SpeechSource,
    timestamps::{Diarization, Punctuation, TimestampsGranularity, WordTimestamps},
    transcript::{TranscriptSegment, TranscriptStream},
    vad_mode::VadMode,
};
use futures_core::Stream;
use std::pin::Pin;
// We use futures::stream instead of futures_util::stream since futures is in the dependencies
use futures::StreamExt;
use futures::stream;
use std::marker::PhantomData;

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
    stream_fn: Box<
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
    S: TranscriptStream,
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
    /// Engine configuration parameters
    pub engine_config: std::collections::HashMap<String, String>,
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
    S: TranscriptStream + 'static,
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
            engine_config: std::collections::HashMap::new(),
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
        transcription_builder.collect()
    }
}

impl<S> crate::stt_conversation::SttConversationBuilder for SttConversationBuilderImpl<S>
where
    S: TranscriptStream + 'static,
{
    type Conversation = SttConversationImpl<S>;

    fn with_source(mut self, src: SpeechSource) -> Self {
        self.source = Some(src);
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

    fn on_chunk<F, T>(self, f: F) -> impl crate::stt_conversation::SttPostChunkBuilder
    where
        F: FnMut(Result<T, VoiceError>) -> T + Send + 'static,
        T: fluent_voice_domain::transcript::TranscriptSegment + Send + 'static,
    {
        SttPostChunkBuilderImpl::new(self, f)
    }

    fn on_result<F>(self, _f: F) -> Self
    where
        F: FnMut(VoiceError) -> String + Send + 'static,
    {
        // Store the result processor for error handling
        // For now, we'll just return self until we implement storage
        self
    }

    fn on_wake<F>(self, _f: F) -> Self
    where
        F: FnMut(String) + Send + 'static,
    {
        // Store the wake processor for wake word detection
        // For now, we'll just return self until we implement storage
        self
    }

    fn on_turn_detected<F>(self, _f: F) -> Self
    where
        F: FnMut(Option<String>, String) + Send + 'static,
    {
        // Store the turn detection processor
        // For now, we'll just return self until we implement storage
        self
    }
}

/// Post-chunk builder implementation for STT
pub struct SttPostChunkBuilderImpl<S, F, T>
where
    S: TranscriptStream + 'static,
    F: FnMut(Result<T, VoiceError>) -> T + Send + 'static,
    T: fluent_voice_domain::transcript::TranscriptSegment + Send + 'static,
{
    /// Base builder
    base_builder: SttConversationBuilderImpl<S>,
    /// Chunk processor function
    #[allow(dead_code)]
    chunk_processor: F,
    /// Phantom data for type parameter
    _phantom: PhantomData<T>,
}

impl<S, F, T> SttPostChunkBuilderImpl<S, F, T>
where
    S: TranscriptStream + 'static,
    F: FnMut(Result<T, VoiceError>) -> T + Send + 'static,
    T: fluent_voice_domain::transcript::TranscriptSegment + Send + 'static,
{
    /// Create a new post-chunk builder
    pub fn new(base_builder: SttConversationBuilderImpl<S>, chunk_processor: F) -> Self {
        Self {
            base_builder,
            chunk_processor,
            _phantom: PhantomData,
        }
    }
}

impl<S, F, T> crate::stt_conversation::SttPostChunkBuilder for SttPostChunkBuilderImpl<S, F, T>
where
    S: TranscriptStream + 'static,
    F: FnMut(Result<T, VoiceError>) -> T + Send + 'static,
    T: fluent_voice_domain::transcript::TranscriptSegment + Send + 'static,
{
    type Conversation = SttConversationImpl<S>;

    fn with_microphone(
        self,
        device: impl Into<String>,
    ) -> impl crate::stt_conversation::MicrophoneBuilder {
        let SttConversationBuilderImpl {
            vad_mode,
            noise_reduction,
            language_hint,
            diarization,
            word_timestamps,
            timestamps_granularity,
            punctuation,
            stream_fn,
            ..
        } = self.base_builder;

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
        let SttConversationBuilderImpl {
            vad_mode,
            noise_reduction,
            language_hint,
            diarization,
            word_timestamps,
            timestamps_granularity,
            punctuation,
            stream_fn,
            ..
        } = self.base_builder;

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

    fn listen<M, R>(self, matcher: M) -> impl std::future::Future<Output = R> + Send
    where
        M: FnOnce(Result<Self::Conversation, VoiceError>) -> R + Send + 'static,
        R: Send + 'static,
    {
        async move {
            let conversation_result = Ok(SttConversationImpl {
                source: self.base_builder.source,
                vad_mode: self.base_builder.vad_mode,
                noise_reduction: self.base_builder.noise_reduction,
                language_hint: self.base_builder.language_hint,
                diarization: self.base_builder.diarization,
                word_timestamps: self.base_builder.word_timestamps,
                timestamps_granularity: self.base_builder.timestamps_granularity,
                punctuation: self.base_builder.punctuation,
                stream_fn: self.base_builder.stream_fn,
            });

            // Call the matcher with the result
            matcher(conversation_result)
        }
    }
}

impl<S> SttConversationBuilderImpl<S>
where
    S: TranscriptStream + 'static,
{
    /// Configure engine parameters using JSON object syntax
    pub fn engine_config(
        mut self,
        config: impl Into<hashbrown::HashMap<&'static str, &'static str>>,
    ) -> Self {
        let config_map = config.into();
        for (k, v) in config_map {
            self.engine_config.insert(k.to_string(), v.to_string());
        }
        self
    }
}

/// Microphone-specific STT builder implementation.
pub struct MicrophoneBuilderImpl<S> {
    /// Device identifier
    device: String,
    /// Voice activity detection mode
    vad_mode: Option<VadMode>,
    /// Noise reduction level
    noise_reduction: Option<NoiseReduction>,
    /// Language hint for recognition
    language_hint: Option<Language>,
    /// Speaker diarization setting
    diarization: Option<Diarization>,
    /// Word-level timestamp setting
    word_timestamps: Option<WordTimestamps>,
    /// Timestamp granularity setting
    timestamps_granularity: Option<TimestampsGranularity>,
    /// Punctuation setting
    punctuation: Option<Punctuation>,
    /// Function to convert configuration to transcript stream
    stream_fn: Box<
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
    S: TranscriptStream,
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

impl<S> crate::stt_conversation::MicrophoneBuilder for MicrophoneBuilderImpl<S>
where
    S: TranscriptStream + 'static,
{
    type Conversation = SttConversationImpl<S>;

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

    fn listen<M, R>(self, matcher: M) -> impl std::future::Future<Output = R> + Send
    where
        M: FnOnce(Result<Self::Conversation, VoiceError>) -> R + Send + 'static,
        R: Send + 'static,
    {
        async move {
            // Use the device string to determine the backend
            let backend = if self.device == "default" || self.device.is_empty() {
                fluent_voice_domain::MicBackend::Default
            } else {
                fluent_voice_domain::MicBackend::Device(self.device)
            };

            let source = Some(SpeechSource::Microphone {
                backend,
                format: fluent_voice_domain::AudioFormat::Pcm16Khz,
                sample_rate: 16_000,
            });

            let conversation_result = Ok(SttConversationImpl {
                source,
                vad_mode: self.vad_mode,
                noise_reduction: self.noise_reduction,
                language_hint: self.language_hint,
                diarization: self.diarization,
                word_timestamps: self.word_timestamps,
                timestamps_granularity: self.timestamps_granularity,
                punctuation: self.punctuation,
                stream_fn: self.stream_fn,
            });

            // Call the matcher with the result
            matcher(conversation_result)
        }
    }
}

/// Transcript collection type for file transcription.
pub struct TranscriptImpl<S: TranscriptStream> {
    /// The transcript stream
    pub stream: S,
}

/// File transcription builder implementation.
pub struct TranscriptionBuilderImpl<S> {
    /// Path to the audio file
    path: String,
    /// Progress message template
    progress_template: Option<String>,
    /// Voice activity detection mode
    vad_mode: Option<VadMode>,
    /// Noise reduction level
    noise_reduction: Option<NoiseReduction>,
    /// Language hint for recognition
    language_hint: Option<Language>,
    /// Speaker diarization setting
    diarization: Option<Diarization>,
    /// Word-level timestamp setting
    word_timestamps: Option<WordTimestamps>,
    /// Timestamp granularity setting
    timestamps_granularity: Option<TimestampsGranularity>,
    /// Punctuation setting
    punctuation: Option<Punctuation>,
    /// Function to convert configuration to transcript stream
    stream_fn: Box<
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

impl<S> TranscriptionBuilderImpl<S>
where
    S: TranscriptStream,
{
    /// Create a new transcription builder.
    pub fn new<F>(
        path: String,
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
            path,
            progress_template: None,
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

    /// Create a transcript stream implementation.
    async fn create_transcript(self) -> Result<TranscriptImpl<S>, VoiceError> {
        let source = Some(SpeechSource::File {
            path: self.path,
            format: fluent_voice_domain::AudioFormat::Pcm16Khz,
        });

        let stream = (self.stream_fn)(
            source,
            self.vad_mode,
            self.noise_reduction,
            self.language_hint,
            self.diarization,
            self.word_timestamps,
            self.timestamps_granularity,
            self.punctuation,
        );

        Ok(TranscriptImpl { stream })
    }
}

impl<S> crate::stt_conversation::TranscriptionBuilder for TranscriptionBuilderImpl<S>
where
    S: TranscriptStream + 'static,
{
    type Transcript = TranscriptImpl<S>;

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

    fn with_progress<S2: Into<String>>(mut self, template: S2) -> Self {
        self.progress_template = Some(template.into());
        self
    }

    fn emit(self) -> impl Stream<Item = String> + Send + Unpin {
        // Create a stream that delegates to the async result when polled
        let stream_fut = async move {
            match self.create_transcript().await {
                Ok(transcript) => {
                    // Convert transcript stream to string stream
                    let text_stream = transcript_stream_to_string_stream(transcript.stream);
                    Box::pin(text_stream) as Pin<Box<dyn Stream<Item = String> + Send>>
                }
                Err(e) => {
                    // Log error and return empty stream
                    log::error!("Transcription error: {}", e);
                    Box::pin(stream::empty::<String>())
                        as Pin<Box<dyn Stream<Item = String> + Send>>
                }
            }
        };

        Box::pin(stream::once(stream_fut).flatten()) as Pin<Box<dyn Stream<Item = String> + Send>>
    }

    fn collect(self) -> impl Future<Output = Result<Self::Transcript, VoiceError>> + Send {
        async move { self.create_transcript().await }
    }

    fn collect_with<F, R>(self, handler: F) -> impl Future<Output = R> + Send
    where
        F: FnOnce(Result<Self::Transcript, VoiceError>) -> R + Send + 'static,
    {
        async move {
            let result = self.collect().await;
            handler(result)
        }
    }

    fn into_text_stream(self) -> impl Stream<Item = String> + Send {
        // Use Box<dyn Stream> to erase the complex return type
        let stream_fut = async move {
            match self.create_transcript().await {
                Ok(transcript) => {
                    // Use map to transform successful segments to text
                    let text_stream = transcript.stream.map(|result| match result {
                        Ok(segment) => segment.text().to_string(),
                        Err(_) => "".to_string(), // Empty string for errors
                    });
                    Box::pin(text_stream) as Pin<Box<dyn Stream<Item = String> + Send>>
                }
                Err(_) => {
                    // Return empty stream on error
                    Box::pin(stream::empty::<String>())
                        as Pin<Box<dyn Stream<Item = String> + Send>>
                }
            }
        };

        // Create a stream that delegates to the async result when polled
        Box::pin(stream::once(stream_fut).flatten()) as Pin<Box<dyn Stream<Item = String> + Send>>
    }
}

/// Module with builder factory functions
pub mod builder {
    use super::*;

    /// Create a new STT conversation builder
    pub fn stt_conversation_builder<S, F>(stream_fn: F) -> SttConversationBuilderImpl<S>
    where
        S: TranscriptStream + 'static,
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
}
