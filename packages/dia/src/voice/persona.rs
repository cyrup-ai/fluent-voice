//! Voice personality traits

/// Personality traits that affect voice delivery
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, clap::ValueEnum, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum VoicePersona {
    Sarcastic,
    Joking,
    Anxious,
    Confident,
    Gentle,
    Aggressive,
    Playful,
    Serious,
    Tired,
    Enthusiastic,
    Melancholic,
    Mysterious,
    Authoritative,
}
