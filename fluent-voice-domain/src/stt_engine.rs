//! Registration trait for STT engine implementations.
use crate::stt_conversation::SttConversationBuilder;

/// Registration trait for STT engine implementations.
///
/// This trait is implemented by STT engine crates to register themselves
/// with the fluent voice system. It provides a way for engines to expose
/// their conversation builder functionality for speech-to-text operations.
///
/// # Examples
///
/// Engine implementations typically look like this:
///
/// ```ignore
/// use fluent_voice::{stt_engine::SttEngine, fluent_voice::FluentVoice};
///
/// pub struct MySttEngine;
///
/// // Implement FluentVoice for the main entry points
/// impl FluentVoice for MySttEngine {
///     fn tts() -> impl TtsConversationBuilder {
///         MyTtsConversationBuilder::new()
///     }
///
///     fn stt() -> impl SttConversationBuilder {
///         MySttConversationBuilder::new()
///     }
/// }
///
/// // Also implement SttEngine for registration
/// impl SttEngine for MySttEngine {
///     type Conv = MySttConversationBuilder;
///
///     fn conversation(&self) -> Self::Conv {
///         MySttConversationBuilder::new()
///     }
/// }
/// ```
pub trait SttEngine: Send + Sync {
    /// The concrete conversation builder type provided by this engine.
    type Conv: SttConversationBuilder;

    /// Create a new conversation builder instance.
    ///
    /// This method initializes a fresh conversation builder that can be
    /// used to configure audio sources, language hints, VAD settings,
    /// and other recognition parameters before starting transcription.
    ///
    /// # Returns
    ///
    /// A new conversation builder instance.
    fn conversation(&self) -> Self::Conv;
}
