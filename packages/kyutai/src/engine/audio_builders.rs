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
    fn with_audio_data(self, _data: Vec<u8>) -> Self {
        self
    }

    #[inline]
    fn add_reverb(self, _room_size: f32, _damping: f32) -> Self {
        self
    }

    #[inline]
    fn add_echo(self, _delay_ms: u32, _decay: f32) -> Self {
        self
    }

    #[inline]
    fn pitch_shift(self, _semitones: f32) -> Self {
        self
    }

    #[inline]
    fn low_pass(self, _frequency: u32) -> Self {
        self
    }

    #[inline]
    fn high_pass(self, _frequency: u32) -> Self {
        self
    }

    #[inline]
    fn add_chorus(self, _speed: f32, _depth: f32) -> Self {
        self
    }

    #[inline]
    fn output_format(self, _format: AudioFormat) -> Self {
        self
    }

    fn process<F, R>(self, matcher: F) -> impl core::future::Future<Output = R> + Send
    where
        F: FnOnce(Result<Self::Session, VoiceError>) -> R + Send + 'static,
    {
        async move {
            matcher(Err(VoiceError::ProcessingError(
                "Sound effects generation requires specialized audio synthesis models".to_string(),
            )))
        }
    }
}
