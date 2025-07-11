//! Pronunciation dictionary support for TTS.

/// Pronunciation dictionary identifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PronunciationDictId {
    /// Dictionary identifier.
    pub id: String,
    /// Optional version identifier.
    pub version: Option<String>,
}

impl PronunciationDictId {
    /// Create a new pronunciation dictionary identifier.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            version: None,
        }
    }

    /// Create a pronunciation dictionary identifier with version.
    pub fn with_version(id: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            version: Some(version.into()),
        }
    }

    /// Get the dictionary ID.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get the version if specified.
    pub fn version(&self) -> Option<&str> {
        self.version.as_deref()
    }
}

/// Request identifier for context continuity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestId(pub String);

impl RequestId {
    /// Create a new request identifier.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Get the underlying identifier string.
    pub fn id(&self) -> &str {
        &self.0
    }
}

impl From<String> for RequestId {
    fn from(id: String) -> Self {
        Self(id)
    }
}

impl From<&str> for RequestId {
    fn from(id: &str) -> Self {
        Self(id.to_string())
    }
}
