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

    /// English (US)
    pub const EnglishUS: Self = Self("en-US");

    /// English (UK)
    pub const EnglishUK: Self = Self("en-GB");

    /// Spanish
    pub const Spanish: Self = Self("es-ES");

    /// French
    pub const French: Self = Self("fr-FR");

    /// German
    pub const German: Self = Self("de-DE");

    /// Japanese
    pub const Japanese: Self = Self("ja-JP");

    /// Chinese (Simplified)
    pub const ChineseSimplified: Self = Self("zh-CN");

    /// Chinese (Traditional)
    pub const ChineseTraditional: Self = Self("zh-TW");
}
