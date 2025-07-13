//! Voice labeling and categorization system.

use std::collections::HashMap;

/// Voice category classification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VoiceCategory {
    /// Pre-made voices provided by the engine.
    Premade,
    /// Cloned custom voices.
    Cloned,
    /// AI-generated voices.
    Generated,
}

/// Voice type classification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VoiceType {
    /// User's personal custom voices.
    Personal,
    /// Community-created voices.
    Community,
    /// Default engine-provided voices.
    Default,
    /// Professional voice actor voices.
    Professional,
}

/// Voice characteristic labels for filtering and discovery.
#[derive(Debug, Clone, Default)]
pub struct VoiceLabels {
    labels: HashMap<String, String>,
}

impl VoiceLabels {
    /// Create a new empty voice labels collection.
    pub fn new() -> Self {
        Self {
            labels: HashMap::new(),
        }
    }

    /// Set the accent label.
    pub fn accent(mut self, accent: impl Into<String>) -> Self {
        self.labels.insert("accent".to_string(), accent.into());
        self
    }

    /// Set the gender label.
    pub fn gender(mut self, gender: impl Into<String>) -> Self {
        self.labels.insert("gender".to_string(), gender.into());
        self
    }

    /// Set the age label.
    pub fn age(mut self, age: impl Into<String>) -> Self {
        self.labels.insert("age".to_string(), age.into());
        self
    }

    /// Set the use case label.
    pub fn use_case(mut self, use_case: impl Into<String>) -> Self {
        self.labels.insert("use case".to_string(), use_case.into());
        self
    }

    /// Set a custom description label.
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.labels
            .insert("description".to_string(), description.into());
        self
    }

    /// Set an arbitrary custom label.
    pub fn custom_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.insert(key.into(), value.into());
        self
    }

    /// Get all labels as a map.
    pub fn labels(&self) -> &HashMap<String, String> {
        &self.labels
    }

    /// Get a specific label value.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.labels.get(key).map(|s| s.as_str())
    }
}

/// Voice details returned from discovery operations.
#[derive(Debug, Clone)]
pub struct VoiceDetails {
    /// Unique voice identifier.
    pub voice_id: crate::voice_id::VoiceId,
    /// Human-readable voice name.
    pub name: String,
    /// Voice category.
    pub category: VoiceCategory,
    /// Voice type.
    pub voice_type: Option<VoiceType>,
    /// Voice characteristic labels.
    pub labels: VoiceLabels,
    /// Optional description.
    pub description: Option<String>,
    /// Preview URL for voice sample.
    pub preview_url: Option<String>,
    /// Available subscription tiers.
    pub available_for_tiers: Vec<String>,
    /// Default voice settings.
    pub default_settings: Option<VoiceSettings>,
}

/// Voice-specific synthesis settings.
#[derive(Debug, Clone)]
pub struct VoiceSettings {
    /// Voice stability setting (0.0-1.0).
    pub stability: Option<f32>,
    /// Voice similarity boost (0.0-1.0).
    pub similarity_boost: Option<f32>,
    /// Style exaggeration level (0.0-1.0).
    pub style: Option<f32>,
    /// Whether speaker boost is enabled.
    pub use_speaker_boost: Option<bool>,
}

impl VoiceSettings {
    /// Create new voice settings.
    pub fn new() -> Self {
        Self {
            stability: None,
            similarity_boost: None,
            style: None,
            use_speaker_boost: None,
        }
    }

    /// Set stability value.
    pub fn stability(mut self, stability: f32) -> Self {
        self.stability = Some(stability);
        self
    }

    /// Set similarity boost value.
    pub fn similarity_boost(mut self, similarity: f32) -> Self {
        self.similarity_boost = Some(similarity);
        self
    }

    /// Set style exaggeration value.
    pub fn style(mut self, style: f32) -> Self {
        self.style = Some(style);
        self
    }

    /// Set speaker boost enabled.
    pub fn speaker_boost(mut self, enabled: bool) -> Self {
        self.use_speaker_boost = Some(enabled);
        self
    }
}

impl Default for VoiceSettings {
    fn default() -> Self {
        Self::new()
    }
}
