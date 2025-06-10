//! BCP-47 language tag (e.g. "en-US").
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Language(pub &'static str);

impl Language {
    /// Create a new language tag.
    pub const fn new(code: &'static str) -> Self {
        Self(code)
    }

    /// Get the underlying language code.
    pub const fn code(&self) -> &'static str {
        self.0
    }
}
