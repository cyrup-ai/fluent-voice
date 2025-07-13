//! Vocal speed modifications

/// Speed modifications for vocal delivery
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VocalSpeedMod {
    Nervous,
    Deliberate,
    Excited,
    Drowsy,
    Stuttering,
    Flowing,
    Choppy,
    Measured,
}
