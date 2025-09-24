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
    /// List of discovered voice entries.
    pub voices: Vec<VoiceDiscoveryEntry>,
    /// Total count of available voices.
    pub total_count: usize,
    /// Token for next page (if available).
    pub next_page_token: Option<String>,
}

#[derive(Debug, Clone)]
pub struct VoiceDiscoveryEntry {
    /// Voice identifier.
    pub voice_id: VoiceId,
    /// Similarity score (0.0 to 1.0).
    pub similarity_score: f32,
    /// Speaker embedding vector.
    pub embedding: Vec<f32>,
    /// Voice metadata.
    pub metadata: VoiceMetadata,
}

#[derive(Debug, Clone)]
pub struct VoiceMetadata {
    /// Language of the voice.
    pub language: Option<Language>,
    /// Voice category.
    pub category: Option<VoiceCategory>,
    /// Voice type.
    pub voice_type: Option<VoiceType>,
    /// Voice labels.
    pub labels: Option<VoiceLabels>,
    /// Gender information.
    pub gender: Option<String>,
    /// Age range.
    pub age_range: Option<String>,
}

#[derive(Debug, Clone)]
struct VoiceDatabaseEntry {
    pub voice_id: VoiceId,
    pub audio_path: String,
    pub metadata: VoiceMetadata,
}

impl VoiceDiscoveryResult {
    /// Create a new voice discovery result.
    pub fn new(
        voices: Vec<VoiceDiscoveryEntry>,
        total_count: usize,
        next_page_token: Option<String>,
    ) -> Self {
        Self {
            voices,
            total_count,
            next_page_token,
        }
    }
}

/// Concrete voice discovery builder implementation.
pub struct VoiceDiscoveryBuilderImpl {
    search_term: Option<String>,
    search_audio: Option<String>,
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
            search_audio: None,
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

    /// Set audio file to search for similar voices.
    pub fn with_audio_search(mut self, audio_path: impl Into<String>) -> Self {
        self.search_audio = Some(audio_path.into());
        self
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
            match self.perform_speaker_recognition().await {
                Ok(discoveries) => {
                    let next_page_token = if discoveries.len() >= self.page_size.unwrap_or(50) {
                        Some(format!(
                            "page_{}",
                            (discoveries.len() / self.page_size.unwrap_or(50)) + 1
                        ))
                    } else {
                        None
                    };

                    let result = VoiceDiscoveryResult::new(
                        discoveries.clone(),
                        discoveries.len(),
                        next_page_token,
                    );
                    matcher(Ok(result))
                }
                Err(e) => matcher(Err(e)),
            }
        }
    }
}

impl VoiceDiscoveryBuilderImpl {
    async fn perform_speaker_recognition(&self) -> Result<Vec<VoiceDiscoveryEntry>, VoiceError> {
        self.load_ecapa_model().await?;
        let voice_database = self.get_voice_database().await?;
        let mut discoveries = Vec::new();

        for voice_entry in voice_database {
            let embedding = self.extract_embedding(&voice_entry.audio_path).await?;

            let similarity_score = if let Some(ref search_audio) = self.search_audio {
                let search_embedding = self.extract_embedding(search_audio).await?;
                self.cosine_similarity(&embedding, &search_embedding)
            } else {
                1.0
            };

            discoveries.push(VoiceDiscoveryEntry {
                voice_id: voice_entry.voice_id,
                similarity_score,
                embedding,
                metadata: voice_entry.metadata,
            });
        }

        discoveries.sort_by(|a, b| {
            b.similarity_score
                .partial_cmp(&a.similarity_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(discoveries)
    }

    async fn load_ecapa_model(&self) -> Result<(), VoiceError> {
        let python_script = r#"
from speechbrain.inference import SpeakerRecognition

verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)

print("ECAPA-TDNN model loaded successfully")
"#;

        let mut cmd = tokio::process::Command::new("python");
        cmd.arg("-c").arg(python_script);

        let output = cmd.output().await.map_err(|e| {
            VoiceError::AudioProcessing(format!("Failed to load ECAPA-TDNN model: {}", e))
        })?;

        if !output.status.success() {
            return Err(VoiceError::AudioProcessing(format!(
                "ECAPA-TDNN loading failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        Ok(())
    }

    async fn get_voice_database(&self) -> Result<Vec<VoiceDatabaseEntry>, VoiceError> {
        // Return empty database until proper voice database integration is implemented
        // Voice discovery will work with empty results and similarity matching can be tested
        // with the with_audio_search() method when real voice entries are eventually added
        Ok(Vec::new())
    }

    async fn extract_embedding(&self, audio_path: &str) -> Result<Vec<f32>, VoiceError> {
        let python_script = format!(
            r#"
from speechbrain.inference import SpeakerRecognition
import torch
import numpy as np

verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)

embedding = verification.encode_batch("{}")
print(",".join(map(str, embedding.flatten().tolist())))
"#,
            audio_path
        );

        let mut cmd = tokio::process::Command::new("python");
        cmd.arg("-c").arg(&python_script);

        let output = cmd.output().await.map_err(|e| {
            VoiceError::AudioProcessing(format!("Embedding extraction failed: {}", e))
        })?;

        if !output.status.success() {
            return Err(VoiceError::AudioProcessing(format!(
                "Embedding extraction failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        let embedding_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let embedding: Result<Vec<f32>, _> =
            embedding_str.split(',').map(|s| s.parse::<f32>()).collect();

        embedding
            .map_err(|e| VoiceError::AudioProcessing(format!("Failed to parse embedding: {}", e)))
    }

    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }
}
