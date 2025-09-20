//! Concrete voice cloning builder implementation.

use core::future::Future;
use dia::voice::global_pool;
use fluent_voice_domain::{
    model_id::ModelId, voice_id::VoiceId, voice_labels::VoiceLabels, VoiceError,
};

/// Builder trait for voice cloning functionality.
pub trait VoiceCloneBuilder: Sized + Send {
    /// The result type produced by this builder.
    type Result: Send;

    /// Set multiple voice samples at once.
    fn with_samples(self, samples: Vec<impl Into<String>>) -> Self;

    /// Add a single voice sample.
    fn with_sample(self, sample: impl Into<String>) -> Self;

    /// Set the name for the cloned voice.
    fn name(self, name: impl Into<String>) -> Self;

    /// Set the description for the cloned voice.
    fn description(self, description: impl Into<String>) -> Self;

    /// Set voice labels/tags.
    fn labels(self, labels: VoiceLabels) -> Self;

    /// Set the fine-tuning model.
    fn fine_tuning_model(self, model: ModelId) -> Self;

    /// Enable enhanced processing.
    fn enhanced_processing(self, enabled: bool) -> Self;

    /// Create the cloned voice with a matcher closure.
    fn create<F, R>(self, matcher: F) -> impl Future<Output = R> + Send
    where
        F: FnOnce(Result<Self::Result, VoiceError>) -> R + Send + 'static;
}

/// Result type for voice cloning operations.
#[derive(Debug, Clone)]
pub struct VoiceCloneResult {
    /// The ID of the newly created voice.
    pub voice_id: VoiceId,
    /// The name of the cloned voice.
    pub name: String,
}

impl VoiceCloneResult {
    /// Create a new voice clone result.
    pub fn new(voice_id: VoiceId, name: String) -> Self {
        Self { voice_id, name }
    }
}

/// Concrete voice cloning builder implementation.
pub struct VoiceCloneBuilderImpl {
    samples: Vec<String>,
    name: Option<String>,
    description: Option<String>,
    labels: Option<VoiceLabels>,
    fine_tuning_model: Option<ModelId>,
    enhanced_processing: bool,
}

impl VoiceCloneBuilderImpl {
    /// Create a new voice cloning builder.
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
            name: None,
            description: None,
            labels: None,
            fine_tuning_model: None,
            enhanced_processing: false,
        }
    }
}

impl Default for VoiceCloneBuilderImpl {
    fn default() -> Self {
        Self::new()
    }
}

impl VoiceCloneBuilder for VoiceCloneBuilderImpl {
    type Result = VoiceCloneResult;

    fn with_samples(mut self, samples: Vec<impl Into<String>>) -> Self {
        self.samples = samples.into_iter().map(|s| s.into()).collect();
        self
    }

    fn with_sample(mut self, sample: impl Into<String>) -> Self {
        self.samples.push(sample.into());
        self
    }

    fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    fn labels(mut self, labels: VoiceLabels) -> Self {
        self.labels = Some(labels);
        self
    }

    fn fine_tuning_model(mut self, model: ModelId) -> Self {
        self.fine_tuning_model = Some(model);
        self
    }

    fn enhanced_processing(mut self, enabled: bool) -> Self {
        self.enhanced_processing = enabled;
        self
    }

    fn create<F, R>(self, matcher: F) -> impl Future<Output = R> + Send
    where
        F: FnOnce(Result<Self::Result, VoiceError>) -> R + Send + 'static,
    {
        async move {
            // Validate we have at least one sample
            if self.samples.is_empty() {
                return matcher(Err(VoiceError::Configuration(
                    "At least one voice sample is required for cloning".to_string(),
                )));
            }

            // Process the first sample (can be extended to handle multiple samples)
            let first_sample = &self.samples[0];
            let voice_name = self.name.unwrap_or_else(|| "Cloned Voice".to_string());

            // Load voice data through dia's voice pool system
            let pool = match global_pool() {
                Ok(pool) => pool,
                Err(e) => return matcher(Err(VoiceError::Configuration(format!("Failed to access voice pool: {}", e)))),
            };
            match pool.load_voice(&voice_name, first_sample) {
                Ok(_voice_data) => {
                    // Voice data is now loaded and cached in the pool
                    // The voice can be used by creating DiaSpeaker objects with this voice_name

                    // Return successful result with real voice ID
                    let voice_id = VoiceId::new(voice_name.clone());
                    let result = VoiceCloneResult::new(voice_id, voice_name);
                    matcher(Ok(result))
                }
                Err(e) => matcher(Err(VoiceError::ProcessingError(format!(
                    "Failed to load voice sample for cloning: {}",
                    e
                )))),
            }
        }
    }
}
