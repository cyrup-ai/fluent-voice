//! Voice cloning builder for creating custom voices.

use crate::{
    model_id::ModelId, voice_error::VoiceError, voice_id::VoiceId, voice_labels::VoiceLabels,
};
use core::future::Future;

/// Result of voice cloning operations.
#[derive(Debug, Clone)]
pub struct VoiceCloneResult {
    /// The newly created voice ID.
    pub voice_id: VoiceId,
    /// Voice name as created.
    pub name: String,
    /// Voice description.
    pub description: Option<String>,
    /// Applied voice labels.
    pub labels: VoiceLabels,
    /// Whether the voice requires verification.
    pub requires_verification: bool,
    /// Whether the voice is immediately ready for use.
    pub is_ready: bool,
}

impl VoiceCloneResult {
    /// Create a new clone result.
    pub fn new(voice_id: VoiceId, name: String) -> Self {
        Self {
            voice_id,
            name,
            description: None,
            labels: VoiceLabels::new(),
            requires_verification: false,
            is_ready: true,
        }
    }

    /// Get the voice ID for the cloned voice.
    pub fn id(&self) -> &VoiceId {
        &self.voice_id
    }

    /// Get the voice name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Check if the voice is ready for immediate use.
    pub fn ready(&self) -> bool {
        self.is_ready && !self.requires_verification
    }
}

/// Fluent builder for voice cloning operations.
///
/// This trait provides the interface for creating custom voices from
/// audio samples using AI voice cloning technology.
pub trait VoiceCloneBuilder: Sized + Send {
    /// The result type for voice cloning operations.
    type Result: Send;

    /// Add audio samples for voice cloning.
    ///
    /// Multiple samples generally improve voice quality and consistency.
    /// Recommended: 1-10 minutes of clean, diverse speech samples.
    ///
    /// # Arguments
    ///
    /// * `samples` - Paths to audio sample files
    fn with_samples(self, samples: Vec<impl Into<String>>) -> Self;

    /// Add a single audio sample.
    ///
    /// # Arguments
    ///
    /// * `sample` - Path to audio sample file
    fn with_sample(self, sample: impl Into<String>) -> Self;

    /// Set the name for the cloned voice.
    ///
    /// # Arguments
    ///
    /// * `name` - Human-readable name for the voice
    fn name(self, name: impl Into<String>) -> Self;

    /// Set a description for the cloned voice.
    ///
    /// # Arguments
    ///
    /// * `description` - Description of the voice characteristics
    fn description(self, description: impl Into<String>) -> Self;

    /// Set voice characteristic labels.
    ///
    /// Labels help with voice organization and discovery.
    ///
    /// # Arguments
    ///
    /// * `labels` - Voice characteristic labels
    fn labels(self, labels: VoiceLabels) -> Self;

    /// Set the fine-tuning model to use.
    ///
    /// Different models may provide different quality/speed tradeoffs
    /// for the voice cloning process.
    ///
    /// # Arguments
    ///
    /// * `model` - Model ID for fine-tuning
    fn fine_tuning_model(self, model: ModelId) -> Self;

    /// Enable enhanced voice processing.
    ///
    /// May improve voice quality but increase processing time.
    fn enhanced_processing(self, enabled: bool) -> Self;

    /// Terminal method that executes voice cloning with a matcher closure.
    ///
    /// This method terminates the fluent chain and executes the voice cloning.
    /// The matcher closure receives either the clone result on success
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
    /// let voice = FluentVoice::clone_voice()
    ///     .from_samples(vec!["sample1.wav", "sample2.wav"])
    ///     .name("MyVoice")
    ///     .description("Custom voice for narration")
    ///     .create(|result| {
    ///         Ok => result.id().clone(),
    ///         Err(e) => Err(e),
    ///     })
    ///     .await?;
    /// ```
    fn create<F, R>(self, matcher: F) -> impl Future<Output = R> + Send
    where
        F: FnOnce(Result<Self::Result, VoiceError>) -> R + Send + 'static;
}

/// Static entry point for voice cloning.
///
/// This trait provides the static method for starting voice cloning operations.
/// Engine implementations typically implement this on a marker struct or
/// their main engine type.
///
/// # Examples
///
/// ```ignore
/// use fluent_voice::prelude::*;
///
/// let clone_builder = MyEngine::clone_voice();
/// ```
pub trait VoiceCloneExt {
    /// Begin a new voice cloning builder.
    ///
    /// # Returns
    ///
    /// A new voice cloning builder instance.
    fn builder() -> impl VoiceCloneBuilder;
}
