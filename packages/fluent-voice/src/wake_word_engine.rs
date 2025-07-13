//! Registration trait for wake word detection engine implementations.

use crate::wake_word::WakeWordBuilder;

/// Registration trait for wake word detection engine implementations.
///
/// This trait is implemented by wake word engine crates to register themselves
/// with the fluent voice system. It provides a way for engines to expose
/// their wake word detection builder functionality.
///
/// # Examples
///
/// Engine implementations typically look like this:
///
/// ```ignore
/// use fluent_voice::{wake_word_engine::WakeWordEngine, fluent_voice::FluentVoice};
///
/// pub struct MyWakeWordEngine;
///
/// // Implement FluentVoice for the main entry points
/// impl FluentVoice for MyWakeWordEngine {
///     fn tts() -> impl TtsConversationBuilder {
///         MyTtsConversationBuilder::new()
///     }
///
///     fn stt() -> impl SttConversationBuilder {
///         MySttConversationBuilder::new()
///     }
///
///     fn wake_word() -> impl WakeWordBuilder {
///         MyWakeWordBuilder::new()
///     }
/// }
///
/// // Also implement WakeWordEngine for registration
/// impl WakeWordEngine for MyWakeWordEngine {
///     type Builder = MyWakeWordBuilder;
///
///     fn builder(&self) -> Self::Builder {
///         MyWakeWordBuilder::new()
///     }
/// }
/// ```
pub trait WakeWordEngine: Send + Sync {
    /// The concrete wake word builder type provided by this engine.
    type Builder: WakeWordBuilder;

    /// Create a new wake word builder instance.
    ///
    /// This method initializes a fresh wake word builder that can be
    /// used to configure wake word models, confidence thresholds,
    /// and other detection parameters before starting detection.
    ///
    /// # Returns
    ///
    /// A new wake word builder instance ready for configuration.
    fn builder(&self) -> Self::Builder;
}
