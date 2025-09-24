//! Parameter storage and session management for synthesis operations.

use fluent_voice_domain::VoiceError;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::SystemTime;

/// Comprehensive parameter storage for synthesis sessions
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SynthesisParameters {
    // Core voice parameters
    pub speaker_id: Option<String>, // Store as string instead of VoiceId
    pub voice_clone_path: Option<PathBuf>,
    pub language: Option<String>, // Store as string instead of Language
    pub speed_modifier: Option<f32>, // Store as f32 instead of VocalSpeedMod
    pub stability: Option<f32>,   // Store as f32 instead of Stability
    pub similarity: Option<f32>,  // Store as f32 instead of Similarity
    pub model_config: Option<String>, // Store as string instead of ModelId
    pub audio_format: Option<String>, // Store as string instead of AudioFormat

    // Additional parameters from builder
    pub additional_params: HashMap<String, String>,
    pub metadata: HashMap<String, String>,

    // Session tracking
    pub session_id: String,
    pub created_at: SystemTime,
    pub updated_at: SystemTime,
}

/// Session tracking for synthesis operations
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SynthesisSession {
    pub parameters: SynthesisParameters,
    pub status: SessionStatus,
    pub error_log: Vec<String>,
}

/// Status tracking for synthesis sessions
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum SessionStatus {
    Initialized,
    Processing,
    Completed,
    Failed(String),
}
impl Default for SynthesisParameters {
    fn default() -> Self {
        Self::new()
    }
}

impl SynthesisParameters {
    pub fn new() -> Self {
        let now = SystemTime::now();
        Self {
            speaker_id: None,
            voice_clone_path: None,
            language: None,
            speed_modifier: None,
            stability: None,
            similarity: None,
            model_config: None,
            audio_format: None,
            additional_params: HashMap::new(),
            metadata: HashMap::new(),
            session_id: Self::generate_session_id(),
            created_at: now,
            updated_at: now,
        }
    }

    fn generate_session_id() -> String {
        // Generate UUID-like string without external dependency
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        SystemTime::now().hash(&mut hasher);
        std::thread::current().id().hash(&mut hasher);

        format!("session_{:x}", hasher.finish())
    }

    pub fn validate(&self) -> Result<(), VoiceError> {
        // Validate voice configuration
        if self.speaker_id.is_none() && self.voice_clone_path.is_none() {
            return Err(VoiceError::Configuration(
                "Either speaker_id or voice_clone_path must be specified".to_string(),
            ));
        }

        // Validate voice clone file exists
        if let Some(ref path) = self.voice_clone_path {
            if !path.exists() {
                return Err(VoiceError::Configuration(format!(
                    "Voice clone file not found: {}",
                    path.display()
                )));
            }

            // Validate file extension
            if let Some(ext) = path.extension() {
                let ext_str = ext.to_string_lossy().to_lowercase();
                if !["wav", "mp3", "flac", "ogg"].contains(&ext_str.as_str()) {
                    return Err(VoiceError::Configuration(format!(
                        "Unsupported voice clone format: {}",
                        ext_str
                    )));
                }
            }
        }

        // Validate parameter ranges
        if let Some(stability) = self.stability {
            if !(0.0..=1.0).contains(&stability) {
                return Err(VoiceError::Configuration(format!(
                    "Stability must be between 0.0 and 1.0, got: {}",
                    stability
                )));
            }
        }

        if let Some(similarity) = self.similarity {
            if !(0.0..=1.0).contains(&similarity) {
                return Err(VoiceError::Configuration(format!(
                    "Similarity must be between 0.0 and 1.0, got: {}",
                    similarity
                )));
            }
        }

        if let Some(speed) = self.speed_modifier {
            if !(0.25..=4.0).contains(&speed) {
                return Err(VoiceError::Configuration(format!(
                    "Speed modifier must be between 0.25 and 4.0, got: {}",
                    speed
                )));
            }
        }

        // Validate additional parameters
        for (key, value) in &self.additional_params {
            if key.len() > 100 || value.len() > 1000 {
                return Err(VoiceError::Configuration(format!(
                    "Parameter too long: key='{}' (max 100), value length={} (max 1000)",
                    key,
                    value.len()
                )));
            }
        }

        Ok(())
    }

    pub fn merge_with(&mut self, other: &SynthesisParameters) {
        // Merge parameters while preserving existing values
        if other.speaker_id.is_some() {
            self.speaker_id = other.speaker_id.clone();
        }
        if other.voice_clone_path.is_some() {
            self.voice_clone_path = other.voice_clone_path.clone();
        }
        if other.language.is_some() {
            self.language = other.language.clone();
        }
        if other.speed_modifier.is_some() {
            self.speed_modifier = other.speed_modifier;
        }
        if other.stability.is_some() {
            self.stability = other.stability;
        }
        if other.similarity.is_some() {
            self.similarity = other.similarity;
        }
        if other.model_config.is_some() {
            self.model_config = other.model_config.clone();
        }
        if other.audio_format.is_some() {
            self.audio_format = other.audio_format.clone();
        }

        // Merge maps
        self.additional_params
            .extend(other.additional_params.clone());
        self.metadata.extend(other.metadata.clone());

        self.updated_at = SystemTime::now();
    }
}
