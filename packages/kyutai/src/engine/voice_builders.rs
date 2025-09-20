//! Voice discovery, cloning, and speech-to-speech builders

use fluent_voice::builders::{SpeechToSpeechBuilder, VoiceCloneBuilder, VoiceDiscoveryBuilder};
use fluent_voice_domain::{
    AudioFormat, Language, ModelId, VoiceError, VoiceId,
    voice_labels::{VoiceCategory, VoiceDetails, VoiceLabels, VoiceType},
};

/// Voice discovery builder for Kyutai voices
#[derive(Debug, Clone)]
pub struct KyutaiVoiceDiscoveryBuilder {
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

impl KyutaiVoiceDiscoveryBuilder {
    #[inline]
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

impl VoiceDiscoveryBuilder for KyutaiVoiceDiscoveryBuilder {
    type Result = Vec<VoiceDetails>;

    #[inline]
    fn search(mut self, term: impl Into<String>) -> Self {
        self.search_term = Some(term.into());
        self
    }

    #[inline]
    fn category(mut self, category: VoiceCategory) -> Self {
        self.category = Some(category);
        self
    }
    #[inline]
    fn voice_type(mut self, voice_type: VoiceType) -> Self {
        self.voice_type = Some(voice_type);
        self
    }

    #[inline]
    fn language(mut self, language: Language) -> Self {
        self.language = Some(language);
        self
    }

    #[inline]
    fn labels(mut self, labels: VoiceLabels) -> Self {
        self.labels = Some(labels);
        self
    }

    #[inline]
    fn page_size(mut self, size: usize) -> Self {
        self.page_size = Some(size);
        self
    }

    #[inline]
    fn page_token(mut self, token: impl Into<String>) -> Self {
        self.page_token = Some(token.into());
        self
    }

    #[inline]
    fn sort_by_created(mut self) -> Self {
        self.sort_by_created = true;
        self
    }

    #[inline]
    fn sort_by_name(mut self) -> Self {
        self.sort_by_name = true;
        self
    }

    async fn discover<F, R>(self, matcher: F) -> R
    where
        F: FnOnce(Result<Self::Result, VoiceError>) -> R,
    {
        // Return empty voice list as Kyutai uses its own voice model
        matcher(Ok(Vec::new()))
    }
}
/// Voice cloning builder
#[derive(Debug, Clone)]
pub struct KyutaiVoiceCloneBuilder {
    _phantom: std::marker::PhantomData<()>,
}

impl KyutaiVoiceCloneBuilder {
    #[inline]
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl VoiceCloneBuilder for KyutaiVoiceCloneBuilder {
    type Result = VoiceDetails;

    #[inline]
    fn with_samples(self, _samples: Vec<impl Into<String>>) -> Self {
        self
    }

    #[inline]
    fn with_sample(self, _sample: impl Into<String>) -> Self {
        self
    }

    #[inline]
    fn name(self, _name: impl Into<String>) -> Self {
        self
    }

    #[inline]
    fn description(self, _description: impl Into<String>) -> Self {
        self
    }

    #[inline]
    fn labels(self, _labels: VoiceLabels) -> Self {
        self
    }

    #[inline]
    fn fine_tuning_model(self, _model: ModelId) -> Self {
        self
    }

    #[inline]
    fn enhanced_processing(self, _enabled: bool) -> Self {
        self
    }

    async fn create<F, R>(self, matcher: F) -> R
    where
        F: FnOnce(Result<Self::Result, VoiceError>) -> R,
    {
        matcher(Err(VoiceError::ProcessingError(
            "Voice cloning requires speaker PCM samples in TTS synthesis".to_string(),
        )))
    }
}
/// Speech-to-speech conversion builder
#[derive(Debug, Clone)]
pub struct KyutaiSpeechToSpeechBuilder {
    _phantom: std::marker::PhantomData<()>,
}

impl KyutaiSpeechToSpeechBuilder {
    #[inline]
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl SpeechToSpeechBuilder for KyutaiSpeechToSpeechBuilder {
    type Session = super::sessions::KyutaiSpeechToSpeechSession;

    #[inline]
    fn with_audio_source(self, _path: impl Into<String>) -> Self {
        self
    }

    #[inline]
    fn with_audio_data(self, _data: Vec<u8>) -> Self {
        self
    }

    #[inline]
    fn target_voice(self, _voice_id: VoiceId) -> Self {
        self
    }

    #[inline]
    fn model(self, _model: ModelId) -> Self {
        self
    }

    #[inline]
    fn preserve_emotion(self, _preserve: bool) -> Self {
        self
    }

    #[inline]
    fn preserve_style(self, _preserve: bool) -> Self {
        self
    }

    #[inline]
    fn preserve_timing(self, _preserve: bool) -> Self {
        self
    }

    #[inline]
    fn output_format(self, _format: AudioFormat) -> Self {
        self
    }

    #[inline]
    fn stability(self, _stability: f32) -> Self {
        self
    }

    #[inline]
    fn similarity_boost(self, _boost: f32) -> Self {
        self
    }

    async fn convert<F, R>(self, matcher: F) -> R
    where
        F: FnOnce(Result<Self::Session, VoiceError>) -> R,
    {
        matcher(Err(VoiceError::ProcessingError(
            "Speech-to-speech conversion requires audio processing pipeline".to_string(),
        )))
    }
}
