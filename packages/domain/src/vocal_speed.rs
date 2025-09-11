//! Speaking rate multiplier (1.0 = normal).
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VocalSpeedMod(pub f32);
