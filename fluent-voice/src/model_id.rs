//! Engine-specific model identifiers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelId {
    /// Multilingual model v2.
    MultilingualV2,
    /// Flash model v2.5 (low latency).
    FlashV2_5,
    /// Turbo model v2.5 (balanced speed/quality).
    TurboV2_5,
    /// ElevenLabs v3 Alpha model.
    ElevenV3Alpha,
    /// Custom engine-specific model identifier.
    Custom(&'static str),
}
