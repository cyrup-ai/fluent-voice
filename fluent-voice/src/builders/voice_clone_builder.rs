//! Concrete voice cloning builder implementation.

use crate::{
    model_id::ModelId,
    voice_clone::{VoiceCloneBuilder, VoiceCloneResult},

    voice_id::VoiceId,
    voice_labels::VoiceLabels,
};
use fluent_voice_domain::VoiceError;
use core::future::Future;

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

    fn from_samples(mut self, samples: Vec<impl Into<String>>) -> Self {
        self.samples = samples.into_iter().map(|s| s.into()).collect();
        self
    }

    fn from_sample(mut self, sample: impl Into<String>) -> Self {
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
            // Placeholder implementation - returns a dummy result
            let voice_id = VoiceId::new("dummy_cloned_voice");
            let name = self.name.unwrap_or_else(|| "Cloned Voice".to_string());
            let result = VoiceCloneResult::new(voice_id, name);
            matcher(Ok(result))
        }
    }
}
