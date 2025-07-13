//! Opaque voice identifier (UUID, slug, etc.).
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VoiceId(pub String);

impl VoiceId {
    /// Create a new voice identifier.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Get the underlying identifier string.
    pub fn id(&self) -> &str {
        &self.0
    }
}
