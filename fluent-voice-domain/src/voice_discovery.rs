//! Voice discovery and management builder.

use crate::{
    language::Language,
    voice_error::VoiceError,
    voice_labels::{VoiceCategory, VoiceDetails, VoiceLabels, VoiceType},
};
use core::future::Future;

/// Result of voice discovery operations.
pub struct VoiceDiscoveryResult {
    /// List of discovered voices.
    pub voices: Vec<VoiceDetails>,
    /// Whether more results are available.
    pub has_more: bool,
    /// Total count of matching voices.
    pub total_count: Option<usize>,
    /// Token for next page of results.
    pub next_page_token: Option<String>,
}

impl VoiceDiscoveryResult {
    /// Create a new discovery result.
    pub fn new(voices: Vec<VoiceDetails>) -> Self {
        Self {
            voices,
            has_more: false,
            total_count: None,
            next_page_token: None,
        }
    }

    /// Create result with pagination info.
    pub fn with_pagination(
        voices: Vec<VoiceDetails>,
        has_more: bool,
        total_count: Option<usize>,
        next_page_token: Option<String>,
    ) -> Self {
        Self {
            voices,
            has_more,
            total_count,
            next_page_token,
        }
    }

    /// Get an iterator over the voices.
    pub fn into_iter(self) -> impl Iterator<Item = VoiceDetails> {
        self.voices.into_iter()
    }

    /// Get the number of voices in this result.
    pub fn len(&self) -> usize {
        self.voices.len()
    }

    /// Check if the result is empty.
    pub fn is_empty(&self) -> bool {
        self.voices.is_empty()
    }
}

/// Fluent builder for voice discovery operations.
///
/// This trait provides the interface for discovering and filtering voices
/// available from the engine provider.
pub trait VoiceDiscoveryBuilder: Sized + Send {
    /// The result type for voice discovery operations.
    type Result: Send;

    /// Filter voices by search term.
    ///
    /// Searches voice names, descriptions, and labels for the given term.
    ///
    /// # Arguments
    ///
    /// * `term` - Search term to filter by
    fn search(self, term: impl Into<String>) -> Self;

    /// Filter by voice category.
    ///
    /// # Arguments
    ///
    /// * `category` - Voice category to filter by
    fn category(self, category: VoiceCategory) -> Self;

    /// Filter by voice type.
    ///
    /// # Arguments
    ///
    /// * `voice_type` - Voice type to filter by
    fn voice_type(self, voice_type: VoiceType) -> Self;

    /// Filter by language support.
    ///
    /// # Arguments
    ///
    /// * `language` - Language the voice must support
    fn language(self, language: Language) -> Self;

    /// Filter by voice labels.
    ///
    /// # Arguments
    ///
    /// * `labels` - Voice labels to match
    fn labels(self, labels: VoiceLabels) -> Self;

    /// Set page size for pagination.
    ///
    /// # Arguments
    ///
    /// * `size` - Number of voices per page (typically 1-100)
    fn page_size(self, size: usize) -> Self;

    /// Set pagination token for next page.
    ///
    /// # Arguments
    ///
    /// * `token` - Next page token from previous result
    fn page_token(self, token: impl Into<String>) -> Self;

    /// Sort results by creation date.
    fn sort_by_created(self) -> Self;

    /// Sort results by name.
    fn sort_by_name(self) -> Self;

    /// Terminal method that executes voice discovery with a matcher closure.
    ///
    /// This method terminates the fluent chain and executes the voice discovery.
    /// The matcher closure receives either the discovery result on success
    /// or a `VoiceError` on failure, and returns the final result.
    ///
    /// # Arguments
    ///
    /// * `matcher` - Closure that handles success/error cases
    ///
    /// # Returns
    ///
    /// A future that resolves to the result of the matcher closure.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let voices = FluentVoice::voices()
    ///     .search("female")
    ///     .category(VoiceCategory::Premade)
    ///     .discover(|result| {
    ///         Ok => result.into_iter().collect(),
    ///         Err(e) => Err(e),
    ///     })
    ///     .await?;
    /// ```
    fn discover<F, R>(self, matcher: F) -> impl Future<Output = R> + Send
    where
        F: FnOnce(Result<Self::Result, VoiceError>) -> R + Send + 'static;
}

/// Static entry point for voice discovery.
///
/// This trait provides the static method for starting voice discovery operations.
/// Engine implementations typically implement this on a marker struct or
/// their main engine type.
///
/// # Examples
///
/// ```ignore
/// use fluent_voice::prelude::*;
///
/// let discovery = MyEngine::voices();
/// ```
pub trait VoiceDiscoveryExt {
    /// Begin a new voice discovery builder.
    ///
    /// # Returns
    ///
    /// A new voice discovery builder instance.
    fn builder() -> impl VoiceDiscoveryBuilder;
}
