//! Concrete voice discovery builder implementation.

use core::future::Future;
use fluent_voice_domain::{
    language::Language,
    voice_id::VoiceId,
    voice_labels::{VoiceCategory, VoiceLabels, VoiceType},
    VoiceError,
};

/// Builder trait for voice discovery functionality.
pub trait VoiceDiscoveryBuilder: Sized + Send {
    /// The result type produced by this builder.
    type Result: Send;

    /// Set search term to filter voices.
    fn search(self, term: impl Into<String>) -> Self;

    /// Filter by voice category.
    fn category(self, category: VoiceCategory) -> Self;

    /// Filter by voice type.
    fn voice_type(self, voice_type: VoiceType) -> Self;

    /// Filter by language.
    fn language(self, language: Language) -> Self;

    /// Filter by voice labels.
    fn labels(self, labels: VoiceLabels) -> Self;

    /// Set page size for pagination.
    fn page_size(self, size: usize) -> Self;

    /// Set page token for pagination.
    fn page_token(self, token: impl Into<String>) -> Self;

    /// Sort results by creation date.
    fn sort_by_created(self) -> Self;

    /// Sort results by name.
    fn sort_by_name(self) -> Self;

    /// Discover voices with a matcher closure.
    fn discover<F, R>(self, matcher: F) -> impl Future<Output = R> + Send
    where
        F: FnOnce(Result<Self::Result, VoiceError>) -> R + Send + 'static;
}

/// Result type for voice discovery operations.
#[derive(Debug, Clone)]
pub struct VoiceDiscoveryResult {
    /// List of discovered voice IDs.
    pub voices: Vec<VoiceId>,
}

impl VoiceDiscoveryResult {
    /// Create a new voice discovery result.
    pub fn new(voices: Vec<VoiceId>) -> Self {
        Self { voices }
    }
}

/// Concrete voice discovery builder implementation.
pub struct VoiceDiscoveryBuilderImpl {
    search_term: Option<String>,
    category: Option<VoiceCategory>,
    voice_type: Option<VoiceType>,
    language: Option<Language>,
    labels: Option<VoiceLabels>,
    page_size: Option<usize>,
    page_token: Option<String>,
    sort_by_created: bool,
    sort_by_name: bool,
}

impl VoiceDiscoveryBuilderImpl {
    /// Create a new voice discovery builder.
    pub fn new() -> Self {
        Self {
            search_term: None,
            category: None,
            voice_type: None,
            language: None,
            labels: None,
            page_size: None,
            page_token: None,
            sort_by_created: false,
            sort_by_name: false,
        }
    }
}

impl Default for VoiceDiscoveryBuilderImpl {
    fn default() -> Self {
        Self::new()
    }
}

impl VoiceDiscoveryBuilder for VoiceDiscoveryBuilderImpl {
    type Result = VoiceDiscoveryResult;

    fn search(mut self, term: impl Into<String>) -> Self {
        self.search_term = Some(term.into());
        self
    }

    fn category(mut self, category: VoiceCategory) -> Self {
        self.category = Some(category);
        self
    }

    fn voice_type(mut self, voice_type: VoiceType) -> Self {
        self.voice_type = Some(voice_type);
        self
    }

    fn language(mut self, language: Language) -> Self {
        self.language = Some(language);
        self
    }

    fn labels(mut self, labels: VoiceLabels) -> Self {
        self.labels = Some(labels);
        self
    }

    fn page_size(mut self, size: usize) -> Self {
        self.page_size = Some(size);
        self
    }

    fn page_token(mut self, token: impl Into<String>) -> Self {
        self.page_token = Some(token.into());
        self
    }

    fn sort_by_created(mut self) -> Self {
        self.sort_by_created = true;
        self.sort_by_name = false;
        self
    }

    fn sort_by_name(mut self) -> Self {
        self.sort_by_name = true;
        self.sort_by_created = false;
        self
    }

    fn discover<F, R>(self, matcher: F) -> impl Future<Output = R> + Send
    where
        F: FnOnce(Result<Self::Result, VoiceError>) -> R + Send + 'static,
    {
        async move {
            // Placeholder implementation - returns empty result
            let result = VoiceDiscoveryResult::new(Vec::new());
            matcher(Ok(result))
        }
    }
}
