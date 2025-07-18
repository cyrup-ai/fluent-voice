//! Concrete voice discovery builder implementation.

use crate::{
    language::Language,
    voice_discovery::{VoiceDiscoveryBuilder, VoiceDiscoveryResult},
    voice_labels::{VoiceCategory, VoiceLabels, VoiceType},
};
use core::future::Future;
use fluent_voice_domain::VoiceError;

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
