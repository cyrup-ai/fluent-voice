//! Runtime speaker instance (engine provides concrete type).

/// Runtime speaker handle that represents a configured voice with text to speak.
///
/// This trait is implemented by engine-specific speaker types that contain
/// all the configuration needed for a single speaking turn (voice ID, text,
/// speed modifiers, etc.).
pub trait Speaker: Clone + Send + Sync {
    /// Returns a unique identifier for this speaker instance.
    ///
    /// Used primarily for debugging and logging purposes.
    fn id(&self) -> &str;
}
