//! Voice timber characteristics

/// Voice timber/texture characteristics
#[derive(Debug, Clone, Copy, PartialEq, clap::ValueEnum, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VoiceTimber {
    Thin,
    Rich,
    Gravelly,
    Smooth,
    Nasal,
    Breathy,
    Metallic,
    Warm,
    Hollow,
    Full,
}
