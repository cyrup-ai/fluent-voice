//! Wake word detection conversation traits and extension methods.

use crate::wake_word::WakeWordBuilder;

/// Extension trait for wake word detection functionality.
///
/// This trait provides convenient methods for creating wake word detection
/// builders on types that implement wake word functionality.
pub trait WakeWordConversationExt {
    /// Create a new wake word detection builder.
    ///
    /// This is a convenience method that delegates to the main FluentVoice trait
    /// implementation to create a wake word builder instance.
    ///
    /// # Returns
    ///
    /// A new wake word builder ready for configuration.
    fn builder() -> impl WakeWordBuilder;
}
