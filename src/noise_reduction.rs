//! Optional denoise level.

/// Noise reduction level for STT audio preprocessing.
///
/// Controls how aggressively the STT engine should filter out background
/// noise and non-speech audio before transcription. Higher levels may
/// improve transcription quality in noisy environments but could potentially
/// filter out quiet speech.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseReduction {
    /// No noise reduction applied.
    ///
    /// Audio is processed as-is without any noise filtering.
    /// Best for clean audio environments or when preserving
    /// all audio characteristics is important.
    Off,

    /// Light noise reduction.
    ///
    /// Applies minimal noise filtering to remove obvious background
    /// noise while preserving speech quality. Good for most environments
    /// with moderate background noise.
    Low,

    /// Aggressive noise reduction.
    ///
    /// Applies strong noise filtering to handle very noisy environments.
    /// May affect speech quality but can significantly improve transcription
    /// accuracy in challenging acoustic conditions.
    High,
}
