//! Multi-speaker conversation builder for TTS.
use core::future::Future;
use fluent_voice_domain::VoiceError;
use fluent_voice_domain::{
    audio_format::AudioFormat,
    language::Language,
    pronunciation_dict::{PronunciationDictId, RequestId},
    speaker::Speaker,
};
use futures_core::Stream;

/// Engine-specific conversation object.
///
/// This trait represents a completed TTS conversation that has been
/// configured and is ready to produce audio output. Engine implementations
/// provide concrete types that implement this trait.
pub trait TtsConversation: Send {
    /// Async audio stream (e.g. PCM i16 samples).
    ///
    /// The specific audio format and sample type depends on the engine
    /// implementation, but typically streams PCM audio samples.
    type AudioStream: Stream<Item = i16> + Send + Unpin;

    /// Convert this conversation into an audio stream.
    ///
    /// This method consumes the conversation and returns the underlying
    /// audio stream that can be used to play or process the synthesized audio.
    fn into_stream(self) -> Self::AudioStream;
}

/// Fluent builder for multi-speaker TTS conversations.
///
/// This trait provides the fluent interface for building conversations
/// with multiple speakers, language settings, and other TTS parameters.
/// Engine implementations provide concrete builder types.
pub trait TtsConversationBuilder: Sized + Send {
    /// Add a speaker turn to the conversation.
    ///
    /// Speakers are processed in the order they are added to the conversation.
    /// Each speaker can have different voice settings, text content, and
    /// expressive parameters.
    ///
    /// # Arguments
    ///
    /// * `speaker` - A configured speaker instance
    fn with_speaker<S: Speaker>(self, speaker: S) -> Self;

    /// Set a global language override for the conversation.
    ///
    /// This language setting may override individual speaker language
    /// settings, depending on the engine implementation.
    ///
    /// # Arguments
    ///
    /// * `lang` - The language specification using BCP-47 format
    fn language(self, lang: Language) -> Self;

    /// Set the model ID for synthesis.
    ///
    /// This identifies which specific TTS model to use when multiple
    /// are available from the engine provider.
    ///
    /// # Arguments
    ///
    /// * `model` - The model identifier
    fn model(self, model: crate::model_id::ModelId) -> Self;

    /// Set the voice stability parameter.
    ///
    /// Controls how consistent the voice characteristics remain throughout
    /// the generated audio. Higher values increase consistency but may
    /// reduce expressiveness.
    ///
    /// # Arguments
    ///
    /// * `stability` - Stability value between 0.0 and 1.0
    fn stability(self, stability: crate::stability::Stability) -> Self;

    /// Set the voice similarity parameter.
    ///
    /// Controls how closely the synthesized voice matches the original voice.
    /// Higher values increase similarity but may affect naturalness.
    ///
    /// # Arguments
    ///
    /// * `similarity` - Similarity value between 0.0 and 1.0
    fn similarity(self, similarity: crate::similarity::Similarity) -> Self;

    /// Enable or disable speaker boost.
    ///
    /// When enabled, enhances the distinction between different speakers
    /// in multi-speaker conversations.
    ///
    /// # Arguments
    ///
    /// * `boost` - Whether to enable or disable speaker boost
    fn speaker_boost(self, boost: crate::speaker_boost::SpeakerBoost) -> Self;

    /// Set the style exaggeration level.
    ///
    /// Controls how strongly the voice style and emotions are expressed.
    /// Higher values create more dramatic, expressive speech.
    ///
    /// # Arguments
    ///
    /// * `exaggeration` - Style intensity between 0.0 and 1.0
    fn style_exaggeration(self, exaggeration: crate::style_exaggeration::StyleExaggeration)
    -> Self;

    /// Set the output audio format.
    ///
    /// Controls the encoding, sample rate, and quality of the generated audio.
    ///
    /// # Arguments
    ///
    /// * `format` - Desired audio output format
    fn output_format(self, format: AudioFormat) -> Self;

    /// Add a pronunciation dictionary for custom word pronunciations.
    ///
    /// Up to 3 pronunciation dictionaries can be applied to improve
    /// accuracy for domain-specific terms, names, or technical vocabulary.
    ///
    /// # Arguments
    ///
    /// * `dict_id` - Pronunciation dictionary identifier
    fn pronunciation_dictionary(self, dict_id: PronunciationDictId) -> Self;

    /// Set a deterministic seed for consistent output.
    ///
    /// When set, the same input will always produce the same audio output,
    /// useful for testing and reproducible results.
    ///
    /// # Arguments
    ///
    /// * `seed` - Deterministic seed value
    fn seed(self, seed: u64) -> Self;

    /// Provide previous text for context continuity.
    ///
    /// Helps the engine understand context from preceding speech segments
    /// to improve prosody and natural flow in multi-part synthesis.
    ///
    /// # Arguments
    ///
    /// * `text` - Text that was spoken before this conversation
    fn previous_text(self, text: impl Into<String>) -> Self;

    /// Provide following text for context continuity.
    ///
    /// Helps the engine anticipate what comes next to improve prosody
    /// and natural transitions in multi-part synthesis.
    ///
    /// # Arguments
    ///
    /// * `text` - Text that will be spoken after this conversation
    fn next_text(self, text: impl Into<String>) -> Self;

    /// Reference previous request IDs for context continuity.
    ///
    /// Links this synthesis request to previous ones for improved
    /// consistency across a series of related speech segments.
    ///
    /// # Arguments
    ///
    /// * `request_ids` - Previous synthesis request identifiers
    fn previous_request_ids(self, request_ids: Vec<RequestId>) -> Self;

    /// Reference following request IDs for context continuity.
    ///
    /// Links this synthesis request to future ones for improved
    /// consistency across a series of related speech segments.
    ///
    /// # Arguments
    ///
    /// * `request_ids` - Following synthesis request identifiers
    fn next_request_ids(self, request_ids: Vec<RequestId>) -> Self;

    /// Terminal method that executes synthesis with cyrup-sugars syntax.
    ///
    /// This method terminates the fluent chain and executes the TTS synthesis.
    /// Uses cyrup-sugars proc macro to enable Ok => expr, Err => expr syntax.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let audio = conversation
    ///     .synthesize(|conversation| {
    ///         Ok => conversation.into_stream(),
    ///         Err(e) => Err(e),
    ///     })
    ///     .await?;
    /// ```
    fn synthesize<F, R>(self, matcher: F) -> impl Future<Output = R> + Send
    where
        F: FnOnce(Result<Self::Conversation, VoiceError>) -> R + Send + 'static;

    /// The concrete conversation type produced by this builder.
    type Conversation: TtsConversation;
}

/// Static entry point for TTS conversations.
///
/// This trait provides the static method for starting a new TTS conversation.
/// Engine implementations typically implement this on a marker struct or
/// their main engine type.
///
/// # Examples
///
/// ```ignore
/// use fluent_voice::prelude::*;
///
/// let conversation = MyEngine::tts();
/// ```
pub trait TtsConversationExt {
    /// Begin a new TTS conversation builder.
    ///
    /// # Returns
    ///
    /// A new conversation builder instance.
    fn builder() -> impl TtsConversationBuilder;
}
