//! Live/batch transcription builder.
use core::future::Future;
use fluent_voice_domain::{
    Diarization, Language, NoiseReduction, Punctuation, SpeechSource, TimestampsGranularity,
    TranscriptionSegmentImpl, TranscriptionStream, VadMode, VoiceError, WordTimestamps,
};
use futures_core::Stream;

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
        S: futures_core::Stream<Item = TranscriptionSegmentImpl> + Send + Unpin + 'static;
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
        S: futures_core::Stream<Item = TranscriptionSegmentImpl> + Send + Unpin + 'static;
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
        S: futures_core::Stream<Item = TranscriptionSegmentImpl> + Send + Unpin + 'static;
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
