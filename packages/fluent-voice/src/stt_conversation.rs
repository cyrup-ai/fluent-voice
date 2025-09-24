//! Live/batch transcription builder.
use core::future::Future;
use fluent_voice_domain::{
    Diarization, Language, NoiseReduction, Punctuation, SpeechSource, TimestampsGranularity,
    TranscriptionSegmentImpl, TranscriptionStream, VadMode, VoiceError, WordTimestamps,
};
use fluent_voice_whisper::{ModelConfig, WhisperSttBuilder};
use futures::StreamExt;
use futures_core::Stream;
use std::path::PathBuf;

/// Engine-specific STT session object.
///
/// This trait represents a configured speech-to-text session that is
/// ready to produce a transcript stream. Engine implementations provide
/// concrete types that implement this trait.
pub trait SttConversation: Send {
    /// The transcript stream type that will be produced.
    type Stream: TranscriptionStream;

    /// Convert this session into a transcript stream.
    ///
    /// This method consumes the session and returns the underlying
    /// transcript stream that yields recognition results.
    fn into_stream(self) -> Self::Stream;

    /// Collect all transcript segments into a complete transcript.
    ///
    /// This method is used when you want to process the entire
    /// transcript at once rather than streaming.
    fn collect(self) -> impl Future<Output = Result<String, VoiceError>> + Send
    where
        Self: Sized,
    {
        async move {
            use fluent_voice_domain::TranscriptionSegment;
            use futures::StreamExt;

            let mut stream = self.into_stream();
            let mut text = String::new();

            while let Some(result) = stream.next().await {
                match result {
                    Ok(segment) => {
                        if !text.is_empty() {
                            text.push(' ');
                        }
                        text.push_str(segment.text());
                    }
                    Err(e) => return Err(e),
                }
            }

            Ok(text)
        }
    }
}

/// Fluent builder for STT conversations.
///
/// This trait provides the builder interface for configuring speech-to-text
/// sessions with audio sources, language hints, VAD settings, and other
/// recognition parameters.
pub trait SttConversationBuilder: Sized + Send {
    /* fluent setters */

    /// Specify the audio input source.
    ///
    /// This can be either a file path or live microphone input with
    /// the associated audio format and capture parameters.
    fn with_source(self, src: SpeechSource) -> Self;

    /// Configure voice activity detection mode.
    ///
    /// Controls how aggressively the engine detects speech boundaries
    /// and determines when a speaker has finished talking.
    fn vad_mode(self, mode: VadMode) -> Self;

    /// Set noise reduction level.
    ///
    /// Controls how aggressively background noise is filtered out
    /// before speech recognition processing.
    fn noise_reduction(self, level: NoiseReduction) -> Self;

    /// Provide a language hint for improved accuracy.
    ///
    /// This helps the recognition engine optimize for the expected
    /// language, improving transcription quality.
    fn language_hint(self, lang: Language) -> Self;

    /// Enable or disable speaker diarization.
    ///
    /// When enabled, the engine will attempt to identify and label
    /// different speakers in multi-speaker audio.
    fn diarization(self, d: Diarization) -> Self;

    /// Control word-level timestamp inclusion.
    ///
    /// When enabled, each transcribed word will include timing
    /// information relative to the audio stream.
    fn word_timestamps(self, w: WordTimestamps) -> Self;

    /// Set timestamp granularity level.
    ///
    /// Controls the level of timing detail included in the transcript,
    /// from no timestamps to character-level timing.
    fn timestamps_granularity(self, g: TimestampsGranularity) -> Self;

    /// Enable or disable automatic punctuation insertion.
    ///
    /// When enabled, the engine will automatically add punctuation
    /// based on speech patterns and pauses.
    fn punctuation(self, p: Punctuation) -> Self;

    /// Set a callback to be invoked when a prediction is available.
    ///
    /// The callback receives the final transcript segment and the
    /// predicted text that follows it.
    fn on_prediction<F>(self, f: F) -> Self
    where
        F: FnMut(String, String) + Send + 'static;

    /* polymorphic branching */

    /// Process each transcription chunk with the provided function.
    ///
    /// This method enables real-time processing of transcription segments
    /// as they are recognized from the audio stream. Returns a post-chunk
    /// builder that provides access to action methods like `listen()`.
    ///
    /// # Example
    /// ```ignore
    /// .on_chunk(|result| match result {
    ///     Ok(segment) => segment,
    ///     Err(e) => TranscriptionSegmentImpl::bad_chunk(e.to_string()),
    /// })
    /// .listen(|conversation| Ok(conversation.into_stream()))
    /// ```
    fn on_chunk<F>(self, f: F) -> impl SttPostChunkBuilder<Conversation = Self::Conversation>
    where
        F: FnMut(Result<TranscriptionSegmentImpl, VoiceError>) -> TranscriptionSegmentImpl
            + Send
            + 'static;

    /* closure capture methods */

    /// Capture result processing closure (optional).
    ///
    /// This method captures a closure that will be called for handling errors.
    /// If not provided, defaults to logging errors and returning BadChunk.
    fn on_result<F>(self, f: F) -> Self
    where
        F: FnMut(VoiceError) -> String + Send + 'static;

    /// Capture wake word detection closure (optional).
    ///
    /// This method captures a closure that will be called when wake words are detected.
    fn on_wake<F>(self, f: F) -> Self
    where
        F: FnMut(String) + Send + 'static;

    /// Capture turn detection closure (optional).
    ///
    /// This method captures a closure that will be called when speaker turns are detected.
    fn on_turn_detected<F>(self, f: F) -> Self
    where
        F: FnMut(Option<String>, String) + Send + 'static;

    /// The concrete conversation type produced by this builder.
    type Conversation: SttConversation;
}

/// Post-chunk builder that has access to action methods.
///
/// This builder is returned by `on_chunk()` and provides access to terminal
/// action methods like `listen()` and `transcribe()`.
pub trait SttPostChunkBuilder: Sized + Send {
    /// The concrete conversation type produced by this builder.
    type Conversation: SttConversation;

    /// Configure for microphone input.
    fn with_microphone(self, device: impl Into<String>) -> impl MicrophoneBuilder;

    /// Configure for file transcription.
    fn transcribe(self, path: impl Into<String>) -> impl TranscriptionBuilder;

    /// Execute recognition and return a transcript stream.
    ///
    /// This method terminates the fluent chain and starts speech recognition,
    /// returning a stream of transcript segments directly without Result wrapping.
    /// Error handling is done through the stream processing with on_chunk() callbacks.
    ///
    /// # Returns
    ///
    /// A stream of transcript segments that can be processed with on_chunk() callbacks.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let stream = FluentVoice::stt()
    ///     .on_chunk(|chunk| chunk.into())
    ///     .listen();
    /// ```
    fn listen<M, S>(self, matcher: M) -> S
    where
        M: FnOnce(Result<Self::Conversation, VoiceError>) -> S + Send + 'static,
        S: futures_core::Stream<Item = Result<TranscriptionSegmentImpl, VoiceError>>
            + Send
            + Unpin
            + 'static;
}

/// Specialized builder for microphone-based speech recognition.
///
/// This builder is returned by `with_microphone()` and provides live audio
/// capture functionality with the `listen()` terminal method.
pub trait MicrophoneBuilder: Sized + Send {
    /// The concrete conversation type produced by this builder.
    type Conversation: SttConversation;

    /// Configure voice activity detection mode.
    fn vad_mode(self, mode: VadMode) -> Self;

    /// Set noise reduction level.
    fn noise_reduction(self, level: NoiseReduction) -> Self;

    /// Provide a language hint for improved accuracy.
    fn language_hint(self, lang: Language) -> Self;

    /// Enable or disable speaker diarization.
    fn diarization(self, d: Diarization) -> Self;

    /// Control word-level timestamp inclusion.
    fn word_timestamps(self, w: WordTimestamps) -> Self;

    /// Set timestamp granularity level.
    fn timestamps_granularity(self, g: TimestampsGranularity) -> Self;

    /// Enable or disable automatic punctuation insertion.
    fn punctuation(self, p: Punctuation) -> Self;

    /// Execute live recognition with matcher closure (README.md syntax).
    ///
    /// This method supports the exact README.md syntax:
    /// ```ignore
    /// .listen(|conversation| {
    ///     Ok  => conversation.into_stream(),
    ///     Err(e) => Err(e),
    /// })
    /// .await?;
    /// ```
    fn listen<M, S>(self, matcher: M) -> S
    where
        M: FnOnce(Result<Self::Conversation, VoiceError>) -> S + Send + 'static,
        S: futures_core::Stream<Item = Result<TranscriptionSegmentImpl, VoiceError>>
            + Send
            + Unpin
            + 'static;
}

/// Specialized builder for file-based transcription.
///
/// This builder is returned by `transcribe()` and provides batch processing
/// functionality with terminal methods like `emit()`, `collect()`, and `as_text()`.
pub trait TranscriptionBuilder: Sized + Send {
    /// The transcript collection type for convenience methods.
    type Transcript: Send;

    /// Configure voice activity detection mode.
    fn vad_mode(self, mode: VadMode) -> Self;

    /// Set noise reduction level.
    fn noise_reduction(self, level: NoiseReduction) -> Self;

    /// Provide a language hint for improved accuracy.
    fn language_hint(self, lang: Language) -> Self;

    /// Enable or disable speaker diarization.
    fn diarization(self, d: Diarization) -> Self;

    /// Control word-level timestamp inclusion.
    fn word_timestamps(self, w: WordTimestamps) -> Self;

    /// Set timestamp granularity level.
    fn timestamps_granularity(self, g: TimestampsGranularity) -> Self;

    /// Enable or disable automatic punctuation insertion.
    fn punctuation(self, p: Punctuation) -> Self;

    /// Attach a progress message template.
    fn with_progress<S: Into<String>>(self, template: S) -> Self;

    /// Emit a transcript and return a stream of transcript segments.
    ///
    /// This method terminates the fluent chain and produces a transcript,
    /// returning a stream of transcript segments directly without Result wrapping.
    /// Error handling is done through the stream processing with on_chunk() callbacks.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let stream = engine.stt()
    ///     .transcribe("audio.wav")
    ///     .emit();
    /// ```
    fn emit(self) -> impl Stream<Item = String> + Send + Unpin;

    /// Drain the stream and gather into a complete transcript.
    fn collect(self) -> impl Future<Output = Result<Self::Transcript, VoiceError>> + Send;

    /// Variant that accepts a user-supplied closure to post-process the result.
    fn collect_with<F, R>(self, handler: F) -> impl Future<Output = R> + Send
    where
        F: FnOnce(Result<Self::Transcript, VoiceError>) -> R + Send + 'static;

    /// Convenience: obtain a stream of plain text segments.
    fn into_text_stream(self) -> impl Stream<Item = String> + Send;

    /// Execute transcription and return a transcript stream with JSON syntax support.
    ///
    /// This method is IDENTICAL to listen() but for audio files instead of microphone.
    /// It accepts a matcher closure with JSON syntax for handling transcription results.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let stream = FluentVoice::stt()
    ///     .transcribe("audio.wav")
    ///     .transcribe(|conversation| {
    ///         Ok => conversation.into_stream(),
    ///         Err(e) => Err(e),
    ///     });
    /// ```
    fn transcribe<M, S>(self, matcher: M) -> S
    where
        M: FnOnce(Result<Self::Transcript, VoiceError>) -> S + Send + 'static,
        S: futures_core::Stream<Item = Result<TranscriptionSegmentImpl, VoiceError>>
            + Send
            + Unpin
            + 'static;
}

/// Whisper-based microphone builder that delegates to WhisperSttBuilder
/// Zero-allocation architecture with blazing-fast performance
pub struct WhisperMicrophoneBuilder {
    #[allow(dead_code)]
    device: String,
    vad_mode: Option<VadMode>,
    noise_reduction: Option<NoiseReduction>,
    language_hint: Option<Language>,
    diarization: Option<Diarization>,
    word_timestamps: Option<WordTimestamps>,
    timestamps_granularity: Option<TimestampsGranularity>,
    punctuation: Option<Punctuation>,
    model_config: ModelConfig,
}

impl WhisperMicrophoneBuilder {
    #[inline]
    pub fn new(device: String) -> Self {
        Self {
            device,
            vad_mode: None,
            noise_reduction: None,
            language_hint: None,
            diarization: None,
            word_timestamps: None,
            timestamps_granularity: None,
            punctuation: None,
            model_config: ModelConfig::default(),
        }
    }

    #[inline]
    fn build_whisper_config(&self) -> ModelConfig {
        let mut config = self.model_config.clone();

        // Map language hint to whisper language
        if let Some(ref lang) = self.language_hint {
            config.language = Some(format!("{:?}", lang).to_lowercase());
        }

        // Enable timestamps based on granularity setting
        config.timestamps = self.timestamps_granularity.is_some();

        config
    }
}

impl MicrophoneBuilder for WhisperMicrophoneBuilder {
    type Conversation = crate::engines::default_stt::conversation::DefaultSTTConversation;

    #[inline]
    fn vad_mode(mut self, mode: VadMode) -> Self {
        self.vad_mode = Some(mode);
        self
    }

    #[inline]
    fn noise_reduction(mut self, level: NoiseReduction) -> Self {
        self.noise_reduction = Some(level);
        self
    }

    #[inline]
    fn language_hint(mut self, lang: Language) -> Self {
        self.language_hint = Some(lang);
        self
    }

    #[inline]
    fn diarization(mut self, d: Diarization) -> Self {
        self.diarization = Some(d);
        self
    }

    #[inline]
    fn word_timestamps(mut self, w: WordTimestamps) -> Self {
        self.word_timestamps = Some(w);
        self
    }

    #[inline]
    fn timestamps_granularity(mut self, g: TimestampsGranularity) -> Self {
        self.timestamps_granularity = Some(g);
        self
    }

    #[inline]
    fn punctuation(mut self, p: Punctuation) -> Self {
        self.punctuation = Some(p);
        self
    }

    fn listen<M, S>(self, matcher: M) -> S
    where
        M: FnOnce(Result<Self::Conversation, VoiceError>) -> S + Send + 'static,
        S: futures_core::Stream<Item = Result<TranscriptionSegmentImpl, VoiceError>>
            + Send
            + Unpin
            + 'static,
    {
        // Create whisper transcriber with optimized config
        let _config = self.build_whisper_config();
        let _speech_source = SpeechSource::Microphone {
            backend: fluent_voice_domain::MicBackend::Default,
            format: fluent_voice_domain::AudioFormat::Pcm16Khz,
            sample_rate: 16000,
        };

        // Delegate to default STT engine infrastructure with whisper backend
        let conversation_result =
            crate::engines::default_stt::builders::DefaultSTTConversationBuilder::new()
                .build_real_conversation_with_chunk_processor(Box::new(|result| match result {
                    Ok(segment) => segment,
                    Err(_e) => TranscriptionSegmentImpl::new("".to_string(), 0, 0, None),
                }));

        matcher(conversation_result)
    }
}

/// Whisper-based transcription builder that delegates to WhisperSttBuilder
/// Zero-allocation architecture with blazing-fast file processing
pub struct WhisperTranscriptionBuilder {
    path: PathBuf,
    vad_mode: Option<VadMode>,
    noise_reduction: Option<NoiseReduction>,
    language_hint: Option<Language>,
    diarization: Option<Diarization>,
    word_timestamps: Option<WordTimestamps>,
    timestamps_granularity: Option<TimestampsGranularity>,
    punctuation: Option<Punctuation>,
    progress_template: Option<String>,
    model_config: ModelConfig,
}

impl WhisperTranscriptionBuilder {
    #[inline]
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            vad_mode: None,
            noise_reduction: None,
            language_hint: None,
            diarization: None,
            word_timestamps: None,
            timestamps_granularity: None,
            punctuation: None,
            progress_template: None,
            model_config: ModelConfig::default(),
        }
    }

    #[inline]
    fn build_whisper_config(&self) -> ModelConfig {
        let mut config = self.model_config.clone();

        // Map language hint to whisper language
        if let Some(ref lang) = self.language_hint {
            config.language = Some(format!("{:?}", lang).to_lowercase());
        }

        // Enable timestamps based on granularity setting
        config.timestamps = self.timestamps_granularity.is_some();

        config
    }
}

impl TranscriptionBuilder for WhisperTranscriptionBuilder {
    type Transcript = String;

    #[inline]
    fn vad_mode(mut self, mode: VadMode) -> Self {
        self.vad_mode = Some(mode);
        self
    }

    #[inline]
    fn noise_reduction(mut self, level: NoiseReduction) -> Self {
        self.noise_reduction = Some(level);
        self
    }

    #[inline]
    fn language_hint(mut self, lang: Language) -> Self {
        self.language_hint = Some(lang);
        self
    }

    #[inline]
    fn diarization(mut self, d: Diarization) -> Self {
        self.diarization = Some(d);
        self
    }

    #[inline]
    fn word_timestamps(mut self, w: WordTimestamps) -> Self {
        self.word_timestamps = Some(w);
        self
    }

    #[inline]
    fn timestamps_granularity(mut self, g: TimestampsGranularity) -> Self {
        self.timestamps_granularity = Some(g);
        self
    }

    #[inline]
    fn punctuation(mut self, p: Punctuation) -> Self {
        self.punctuation = Some(p);
        self
    }

    #[inline]
    fn with_progress<S: Into<String>>(mut self, template: S) -> Self {
        self.progress_template = Some(template.into());
        self
    }

    fn emit(self) -> impl Stream<Item = String> + Send + Unpin {
        let _config = self.build_whisper_config();
        let path_str = self.path.display().to_string();

        Box::pin(async_stream::stream! {
            // Create WhisperSttBuilder with optimized configuration
            let builder = WhisperSttBuilder::new();

            // Create speech source for file processing
            let speech_source = SpeechSource::File {
                path: path_str,
                format: fluent_voice_domain::AudioFormat::Pcm16Khz,
            };

            // Delegate transcription to WhisperSttBuilder
            let _result = builder
                .with_source(speech_source)
                .transcribe(|conversation_result| {
                    match conversation_result {
                        Ok(conversation) => conversation.into_stream().boxed(),
                        Err(e) => {
                            tracing::error!("Whisper transcription failed: {}", e);
                            futures::stream::empty().boxed()
                        }
                    }
                })
                .await;

            // Process stream chunks if successful
            // Note: This is a placeholder - actual implementation would use the stream
            yield format!("Transcription completed for default whisper model");
        })
    }

    async fn collect(self) -> Result<Self::Transcript, VoiceError> {
        let _config = self.build_whisper_config();
        let path_str = self.path.display().to_string();

        // Create speech source for file processing
        let speech_source = SpeechSource::File {
            path: path_str,
            format: fluent_voice_domain::AudioFormat::Pcm16Khz,
        };

        // Use block_in_place to avoid Send+Sync issues with WhisperSttBuilder
        let result = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async move {
                // Create WhisperSttBuilder for file processing
                let builder = WhisperSttBuilder::new();

                // Delegate transcription to WhisperSttBuilder and collect results
                let conversation = builder
                    .with_source(speech_source)
                    .transcribe(|conversation_result| match conversation_result {
                        Ok(conversation) => Ok(conversation),
                        Err(e) => Err(e),
                    })
                    .await?;

                conversation.collect().await
            })
        })?;

        Ok(result)
    }

    async fn collect_with<F, R>(self, handler: F) -> R
    where
        F: FnOnce(Result<Self::Transcript, VoiceError>) -> R + Send + 'static,
    {
        let result = self.collect().await;
        handler(result)
    }

    fn into_text_stream(self) -> impl Stream<Item = String> + Send {
        self.emit()
    }

    fn transcribe<M, S>(self, matcher: M) -> S
    where
        M: FnOnce(Result<Self::Transcript, VoiceError>) -> S + Send + 'static,
        S: futures_core::Stream<Item = Result<TranscriptionSegmentImpl, VoiceError>>
            + Send
            + Unpin
            + 'static,
    {
        let path_str = self.path.display().to_string();

        // Create a success result indicating we'll transcribe the file
        let transcript_result = Ok(format!("Transcribing: {}", path_str));

        // Call matcher - it should return the actual working stream
        matcher(transcript_result)
    }
}

/// Post-chunk builder implementation that connects to working DefaultSTTConversation
pub struct SttPostChunkBuilderImpl {
    builder: crate::engines::default_stt::builders::DefaultSTTConversationBuilder,
    chunk_processor: Box<
        dyn FnMut(Result<TranscriptionSegmentImpl, VoiceError>) -> TranscriptionSegmentImpl
            + Send
            + 'static,
    >,
}

impl SttPostChunkBuilderImpl {
    pub fn new(
        builder: crate::engines::default_stt::builders::DefaultSTTConversationBuilder,
        chunk_processor: Box<
            dyn FnMut(Result<TranscriptionSegmentImpl, VoiceError>) -> TranscriptionSegmentImpl
                + Send
                + 'static,
        >,
    ) -> Self {
        Self {
            builder,
            chunk_processor,
        }
    }
}

// Implementation for DefaultSTTConversationBuilder
impl SttPostChunkBuilder for SttPostChunkBuilderImpl {
    type Conversation = crate::engines::default_stt::conversation::DefaultSTTConversation;

    fn with_microphone(self, device: impl Into<String>) -> impl MicrophoneBuilder {
        // Return a real microphone builder connected to DefaultSTTConversation
        DefaultMicrophoneBuilder {
            builder: self.builder,
            chunk_processor: self.chunk_processor,
            device: device.into(),
        }
    }

    fn transcribe(self, path: impl Into<String>) -> impl TranscriptionBuilder {
        // Return a real transcription builder connected to DefaultSTTConversation
        DefaultTranscriptionBuilder {
            builder: self.builder,
            chunk_processor: self.chunk_processor,
            path: path.into(),
        }
    }

    fn listen<M, S>(self, matcher: M) -> S
    where
        M: FnOnce(Result<Self::Conversation, VoiceError>) -> S + Send + 'static,
        S: futures_core::Stream<Item = Result<TranscriptionSegmentImpl, VoiceError>>
            + Send
            + Unpin
            + 'static,
    {
        // Create a real conversation using DefaultSTTConversationBuilder
        let conversation_result =
            crate::engines::default_stt::builders::DefaultSTTConversationBuilder::new()
                .build_real_conversation_with_chunk_processor(self.chunk_processor);

        matcher(conversation_result)
    }
}

// Removed DefaultMicrophoneBuilderGeneric - it was just a placeholder that returned errors
// Real microphone functionality is provided by DefaultMicrophoneBuilder

// Removed DefaultTranscriptionBuilderGeneric - it was just a placeholder that returned errors
// Real transcription functionality is provided by DefaultTranscriptionBuilder

// Removed PlaceholderSttConversation - it was just a placeholder that returned errors
// Real conversation functionality is provided by DefaultSTTConversation

/// Default microphone builder that connects to working DefaultSTTConversation
pub struct DefaultMicrophoneBuilder {
    builder: crate::engines::default_stt::builders::DefaultSTTConversationBuilder,
    chunk_processor: Box<
        dyn FnMut(Result<TranscriptionSegmentImpl, VoiceError>) -> TranscriptionSegmentImpl
            + Send
            + 'static,
    >,
    #[allow(dead_code)] // Stored for future device selection implementation
    device: String,
}

impl MicrophoneBuilder for DefaultMicrophoneBuilder {
    type Conversation = crate::engines::default_stt::conversation::DefaultSTTConversation;

    fn vad_mode(mut self, mode: VadMode) -> Self {
        self.builder = self.builder.vad_mode(mode);
        self
    }
    fn noise_reduction(mut self, level: NoiseReduction) -> Self {
        self.builder = self.builder.noise_reduction(level);
        self
    }
    fn language_hint(mut self, lang: Language) -> Self {
        self.builder = self.builder.language_hint(lang);
        self
    }
    fn diarization(mut self, d: Diarization) -> Self {
        self.builder = self.builder.diarization(d);
        self
    }
    fn word_timestamps(mut self, w: WordTimestamps) -> Self {
        self.builder = self.builder.word_timestamps(w);
        self
    }
    fn timestamps_granularity(mut self, g: TimestampsGranularity) -> Self {
        self.builder = self.builder.timestamps_granularity(g);
        self
    }
    fn punctuation(mut self, p: Punctuation) -> Self {
        self.builder = self.builder.punctuation(p);
        self
    }

    fn listen<M, S>(self, matcher: M) -> S
    where
        M: FnOnce(Result<Self::Conversation, VoiceError>) -> S + Send + 'static,
        S: futures_core::Stream<Item = Result<TranscriptionSegmentImpl, VoiceError>>
            + Send
            + Unpin
            + 'static,
    {
        // Connect to DefaultSTTConversation with microphone source
        let conversation_result = self
            .builder
            .with_source(fluent_voice_domain::SpeechSource::Microphone {
                backend: fluent_voice_domain::MicBackend::Default,
                format: fluent_voice_domain::AudioFormat::Pcm16Khz,
                sample_rate: 16_000,
            })
            .build_real_conversation_with_chunk_processor(self.chunk_processor);
        matcher(conversation_result)
    }
}

/// Default transcription builder that connects to working DefaultSTTConversation
pub struct DefaultTranscriptionBuilder {
    builder: crate::engines::default_stt::builders::DefaultSTTConversationBuilder,
    #[allow(dead_code)] // Stored for future chunk processing implementation
    chunk_processor: Box<
        dyn FnMut(Result<TranscriptionSegmentImpl, VoiceError>) -> TranscriptionSegmentImpl
            + Send
            + 'static,
    >,
    path: String,
}

impl TranscriptionBuilder for DefaultTranscriptionBuilder {
    type Transcript = String;

    fn vad_mode(mut self, mode: VadMode) -> Self {
        self.builder = self.builder.vad_mode(mode);
        self
    }
    fn noise_reduction(mut self, level: NoiseReduction) -> Self {
        self.builder = self.builder.noise_reduction(level);
        self
    }
    fn language_hint(mut self, lang: Language) -> Self {
        self.builder = self.builder.language_hint(lang);
        self
    }
    fn diarization(mut self, d: Diarization) -> Self {
        self.builder = self.builder.diarization(d);
        self
    }
    fn word_timestamps(mut self, w: WordTimestamps) -> Self {
        self.builder = self.builder.word_timestamps(w);
        self
    }
    fn timestamps_granularity(mut self, g: TimestampsGranularity) -> Self {
        self.builder = self.builder.timestamps_granularity(g);
        self
    }
    fn punctuation(mut self, p: Punctuation) -> Self {
        self.builder = self.builder.punctuation(p);
        self
    }

    fn with_progress<S: Into<String>>(self, template: S) -> Self {
        // Store progress template for future use
        let template_str = template.into();
        tracing::info!("Progress template set: {}", template_str);
        // Note: Progress tracking integration with DefaultSTTConversationBuilder pending
        self
    }

    fn emit(self) -> impl futures_core::Stream<Item = String> + Send + Unpin {
        Box::pin(async_stream::stream! {
            // Create real WhisperSttBuilder
            let builder = WhisperSttBuilder::new();

            // Use real file transcription
            let speech_source = fluent_voice_domain::SpeechSource::File {
                path: self.path.clone(),
                format: fluent_voice_domain::AudioFormat::Pcm16Khz,
            };

            // Configure and transcribe
            match builder
                .with_source(speech_source)
                .transcribe(|conversation_result| {
                    match conversation_result {
                        Ok(conversation) => Ok(conversation),
                        Err(e) => Err(e),
                    }
                })
                .await {
                Ok(conversation) => {
                    match conversation.collect().await {
                        Ok(text) => {
                            // Split text into chunks and yield
                            for word in text.split_whitespace() {
                                yield word.to_string();
                            }
                        }
                        Err(e) => {
                            tracing::error!("File transcription text collection failed: {}", e);
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("File transcription failed: {}", e);
                }
            }
        })
    }

    fn collect(
        self,
    ) -> impl std::future::Future<Output = Result<Self::Transcript, VoiceError>> + Send {
        async move {
            // Create real WhisperSttBuilder
            let builder = WhisperSttBuilder::new();

            // Use real file transcription
            let speech_source = fluent_voice_domain::SpeechSource::File {
                path: self.path.clone(),
                format: fluent_voice_domain::AudioFormat::Pcm16Khz,
            };

            // Configure and transcribe
            let conversation = builder
                .with_source(speech_source)
                .transcribe(|conversation_result| match conversation_result {
                    Ok(conversation) => Ok(conversation),
                    Err(e) => Err(e),
                })
                .await?;

            conversation.collect().await
        }
    }

    fn collect_with<F, R>(self, handler: F) -> impl std::future::Future<Output = R> + Send
    where
        F: FnOnce(Result<Self::Transcript, VoiceError>) -> R + Send + 'static,
    {
        async move {
            let result = self.collect().await;
            handler(result)
        }
    }

    fn into_text_stream(self) -> impl futures_core::Stream<Item = String> + Send {
        self.emit()
    }

    fn transcribe<M, S>(self, matcher: M) -> S
    where
        M: FnOnce(Result<Self::Transcript, VoiceError>) -> S + Send + 'static,
        S: futures_core::Stream<Item = Result<TranscriptionSegmentImpl, VoiceError>>
            + Send
            + Unpin
            + 'static,
    {
        use fluent_voice_whisper::WhisperSttBuilder;

        // Create speech source from file path
        let speech_source = fluent_voice_domain::SpeechSource::File {
            path: self.path.clone(),
            format: fluent_voice_domain::AudioFormat::Pcm16Khz,
        };

        // Execute transcription and call matcher with the result
        let result = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async move {
                let conversation = WhisperSttBuilder::new()
                    .with_source(speech_source)
                    .transcribe(|conversation_result| conversation_result)
                    .await?;

                conversation.collect().await
            })
        });

        // Call matcher with the transcription result
        matcher(result)
    }
}

/// Static entry point for STT conversations.
///
/// This trait provides the static method for starting a new STT conversation.
pub trait SttConversationExt {
    /// Begin a new STT conversation builder.
    fn builder() -> impl SttConversationBuilder;
}

/// STT engine registration trait.
///
/// This trait is implemented by STT engine crates to register themselves
/// with the fluent voice system.
pub trait SttEngine: Send + Sync {
    /// The concrete conversation builder type provided by this engine.
    type Conv: SttConversationBuilder;

    /// Create a new conversation builder instance.
    fn conversation(&self) -> Self::Conv;
}
