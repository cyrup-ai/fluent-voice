//! Registration trait for a TTS engine crate.
use crate::tts_conversation::TtsConversationBuilder;

/// Registration trait for TTS engine implementations.
///
/// This trait is implemented by TTS engine crates to register themselves
/// with the fluent voice system. It provides a way for engines to expose
/// their conversation builder functionality.
///
/// # Examples
///
/// Engine implementations typically look like this:
///
/// ```ignore
/// use fluent_voice::tts_engine::TtsEngine;
///
/// pub struct MyTtsEngine;
///
/// impl TtsEngine for MyTtsEngine {
///     type Conv = MyConversationBuilder;
///
///     fn conversation(&self) -> Self::Conv {
///         MyConversationBuilder::new()
///     }
/// }
/// ```
pub trait TtsEngine: Send + Sync {
    /// The concrete conversation builder type provided by this engine.
    type Conv: TtsConversationBuilder;

    /// Create a new conversation builder instance.
    ///
    /// This method initializes a fresh conversation builder that can be
    /// used to configure speakers, language settings, and other TTS
    /// parameters before synthesis.
    ///
    /// # Returns
    ///
    /// A new conversation builder instance.
    fn conversation(&self) -> Self::Conv;
}
