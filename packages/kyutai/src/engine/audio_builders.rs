//! Audio isolation and sound effects builders

use fluent_voice::builders::{AudioIsolationBuilder, SoundEffectsBuilder};
use fluent_voice_domain::{AudioFormat, VoiceError};

/// Audio isolation builder
#[derive(Debug, Clone)]
pub struct KyutaiAudioIsolationBuilder {
    _phantom: std::marker::PhantomData<()>,
}

impl KyutaiAudioIsolationBuilder {
    #[inline]
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl AudioIsolationBuilder for KyutaiAudioIsolationBuilder {
    type Session = super::sessions::KyutaiAudioIsolationSession;

    #[inline]
    fn with_file(self, _path: impl Into<String>) -> Self {
        self
    }

    #[inline]
    fn with_audio_data(self, _data: Vec<u8>) -> Self {
        self
    }

    #[inline]
    fn isolate_voices(self, _isolate: bool) -> Self {
        self
    }

    #[inline]
    fn remove_background(self, _remove: bool) -> Self {
        self
    }

    #[inline]
    fn reduce_noise(self, _reduce: bool) -> Self {
        self
    }

    #[inline]
    fn isolation_strength(self, _strength: f32) -> Self {
        self
    }

    #[inline]
    fn output_format(self, _format: AudioFormat) -> Self {
        self
    }

    async fn process<F, R>(self, matcher: F) -> R
    where
        F: FnOnce(Result<Self::Session, VoiceError>) -> R,
    {
        matcher(Err(VoiceError::ProcessingError(
            "Audio isolation requires dedicated audio processing models".to_string(),
        )))
    }
}
/// Sound effects generation builder
#[derive(Debug, Clone)]
pub struct KyutaiSoundEffectsBuilder {
    _phantom: std::marker::PhantomData<()>,
}

impl KyutaiSoundEffectsBuilder {
    #[inline]
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl SoundEffectsBuilder for KyutaiSoundEffectsBuilder {
    type Session = super::sessions::KyutaiSoundEffectsSession;

    #[inline]
    fn describe(self, _description: impl Into<String>) -> Self {
        self
    }

    #[inline]
    fn duration_seconds(self, _duration: f32) -> Self {
        self
    }

    #[inline]
    fn intensity(self, _intensity: f32) -> Self {
        self
    }

    #[inline]
    fn mood(self, _mood: impl Into<String>) -> Self {
        self
    }

    #[inline]
    fn environment(self, _environment: impl Into<String>) -> Self {
        self
    }

    #[inline]
    fn output_format(self, _format: AudioFormat) -> Self {
        self
    }

    #[inline]
    fn seed(self, _seed: u64) -> Self {
        self
    }

    async fn generate<F, R>(self, matcher: F) -> R
    where
        F: FnOnce(Result<Self::Session, VoiceError>) -> R,
    {
        matcher(Err(VoiceError::ProcessingError(
            "Sound effects generation requires specialized audio synthesis models".to_string(),
        )))
    }
}
