//! BCP-47 language tag (e.g. "en-US").
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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

    /// English (US)
    pub const ENGLISH_US: Self = Self("en-US");

    /// English (UK)
    pub const ENGLISH_UK: Self = Self("en-GB");

    /// Spanish
    pub const SPANISH: Self = Self("es-ES");

    /// French
    pub const FRENCH: Self = Self("fr-FR");

    /// German
    pub const GERMAN: Self = Self("de-DE");

    /// Japanese
    pub const JAPANESE: Self = Self("ja-JP");

    /// Chinese (Simplified)
    pub const CHINESE_SIMPLIFIED: Self = Self("zh-CN");

    /// Chinese (Traditional)
    pub const CHINESE_TRADITIONAL: Self = Self("zh-TW");
}
