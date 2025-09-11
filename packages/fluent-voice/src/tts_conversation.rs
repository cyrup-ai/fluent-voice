//! TTS conversation builder traits.

use fluent_voice_domain::{
    AudioChunk, AudioFormat, Language, ModelId, PronunciationDictId, RequestId, Similarity,
    Speaker, SpeakerBoost, Stability, StyleExaggeration, TtsConversation, VoiceError,
};
use futures_core::Stream;

/// Builder trait for TTS conversations.
///
/// This trait provides the fluent API for configuring and executing
/// text-to-speech synthesis. All builder methods belong in fluent-voice package.
pub trait TtsConversationBuilder: Sized + Send {
    /// The concrete conversation type produced by this builder.
    type Conversation: TtsConversation;

    /// The chunk builder type for processing synthesis chunks.
    type ChunkBuilder: TtsConversationChunkBuilder;

    /// Add a speaker to the conversation.
    fn with_speaker<S: Speaker>(self, speaker: S) -> Self;

    /// Configure voice cloning from audio file path.
    fn with_voice_clone_path(self, path: std::path::PathBuf) -> Self;

    /// Set the language for the conversation.
    fn language(self, lang: Language) -> Self;

    /// Set the model to use for synthesis.
    fn model(self, model: ModelId) -> Self;

    /// Set the stability setting for synthesis.
    fn stability(self, stability: Stability) -> Self;

    /// Set the similarity setting for synthesis.
    fn similarity(self, similarity: Similarity) -> Self;

    /// Set the speaker boost setting.
    fn speaker_boost(self, boost: SpeakerBoost) -> Self;

    /// Set the style exaggeration setting.
    fn style_exaggeration(self, exaggeration: StyleExaggeration) -> Self;

    /// Set the output audio format.
    fn output_format(self, format: AudioFormat) -> Self;

    /// Add a pronunciation dictionary.
    fn pronunciation_dictionary(self, dict_id: PronunciationDictId) -> Self;

    /// Set the random seed for synthesis.
    fn seed(self, seed: u64) -> Self;

    /// Set the previous text for context.
    fn previous_text(self, text: impl Into<String>) -> Self;

    /// Set the next text for context.
    fn next_text(self, text: impl Into<String>) -> Self;

    /// Set previous request IDs for context.
    fn previous_request_ids(self, request_ids: Vec<RequestId>) -> Self;

    /// Set next request IDs for context.
    fn next_request_ids(self, request_ids: Vec<RequestId>) -> Self;

    /// Set additional parameters using array-tuples syntax.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// .additional_params([("beta", "true"), ("debug", "false")])
    /// ```
    fn additional_params<P>(self, params: P) -> Self
    where
        P: Into<std::collections::HashMap<String, String>>;

    /// Set metadata using array-tuples syntax.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// .metadata([("key", "val"), ("foo", "bar")])
    /// ```
    fn metadata<M>(self, meta: M) -> Self
    where
        M: Into<std::collections::HashMap<String, String>>;

    /// Set a result processor callback.
    fn on_result<F>(self, processor: F) -> Self
    where
        F: FnOnce(Result<Self::Conversation, VoiceError>) + Send + 'static;

    /// Execute synthesis and return an audio stream with JSON syntax support.
    ///
    /// This method terminates the fluent chain and executes the TTS synthesis,
    /// returning a stream of audio chunks that can be processed with the fluent
    /// `.play()` method for real-time audio playback.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let stream = FluentVoice::tts()
    ///     .with_speaker(speaker)
    ///     .synthesize(|conversation| {
    ///         Ok => conversation.into_stream(),
    ///         Err(e) => Err(e),
    ///     });
    /// ```
    fn synthesize<M, S>(self, matcher: M) -> S
    where
        M: FnOnce(Result<Self::Conversation, VoiceError>) -> S + Send + 'static,
        S: Stream<Item = AudioChunk> + Send + Unpin + 'static;
}

/// Trait for chunk-by-chunk processing of TTS synthesis.
pub trait TtsConversationChunkBuilder: Sized + Send {
    /// The concrete conversation type produced by this chunk builder.
    type Conversation: TtsConversation;

    /// Terminal method that executes synthesis with chunk processing.
    fn synthesize(self) -> impl Stream<Item = AudioChunk> + Send + Unpin;
}

/// Extension trait for TTS conversation builders.
///
/// This trait provides the static method for starting a new TTS conversation.
pub trait TtsConversationExt {
    /// Begin a new TTS conversation builder.
    fn builder() -> impl TtsConversationBuilder;
}
