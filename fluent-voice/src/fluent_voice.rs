//! Unified entry point trait for TTS and STT operations.

use crate::{stt_conversation::SttConversationBuilder, tts_conversation::TtsConversationBuilder};

/// Unified entry point trait for Text-to-Speech and Speech-to-Text operations.
///
/// This trait provides the main entry points for the fluent voice API, allowing
/// users to start TTS or STT operations with a consistent interface.
///
/// # Examples
///
/// ```ignore
/// use fluent_voice::prelude::*;
///
/// // TTS usage
/// let audio = MyEngine::tts()
///     .with_speaker(
///         Speaker::named("Alice")
///             .speak("Hello, world!")
///             .build()
///     )
///     .synthesize(|conversation| {
///         Ok => conversation.into_stream(),
///         Err(e) => Err(e),
///     })
///     .await?;
///
/// // STT microphone usage
/// let stream = MyEngine::stt()
///     .with_microphone("default")
///     .listen(|conversation| {
///         Ok => conversation.into_stream(),
///         Err(e) => Err(e),
///     })
///     .await?;
///
/// // STT file transcription usage
/// let transcript = MyEngine::stt()
///     .transcribe("audio.wav")
///     .emit(|transcript| {
///         Ok => transcript.into_stream(),
///         Err(e) => Err(e),
///     })
///     .await?;
/// ```
pub trait FluentVoice {
    /// Begin a new TTS conversation builder.
    ///
    /// This method returns a conversation builder that can be used to configure
    /// speakers, voice settings, and other TTS parameters before synthesis.
    ///
    /// # Returns
    ///
    /// A new TTS conversation builder instance.
    fn tts() -> impl TtsConversationBuilder;

    /// Begin a new STT conversation builder.
    ///
    /// This method returns a conversation builder that can be used to configure
    /// audio sources, language hints, VAD settings, and other recognition
    /// parameters before starting transcription.
    ///
    /// # Returns
    ///
    /// A new STT conversation builder instance.
    fn stt() -> impl SttConversationBuilder;
}
