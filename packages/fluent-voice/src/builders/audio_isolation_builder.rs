//! Concrete audio isolation builder implementation.

use crate::audio_isolation::{AudioIsolationBuilder, AudioIsolationSession};
use core::future::Future;
use fluent_voice_domain::VoiceError;
use fluent_voice_domain::audio_format::AudioFormat;
use futures_core::Stream;

/// Concrete audio isolation session implementation.
pub struct AudioIsolationSessionImpl {
    _source: String,
}

impl AudioIsolationSession for AudioIsolationSessionImpl {
    type AudioStream = Box<dyn Stream<Item = i16> + Send + Unpin>;

    fn into_stream(self) -> Self::AudioStream {
        // Placeholder implementation - returns empty stream
        Box::new(futures::stream::empty::<i16>())
    }
}

/// Concrete audio isolation builder implementation.
pub struct AudioIsolationBuilderImpl {
    source: Option<String>,
    audio_data: Option<Vec<u8>>,
    isolate_voices: bool,
    remove_background: bool,
    reduce_noise: bool,
    isolation_strength: Option<f32>,
    output_format: Option<AudioFormat>,
}

impl AudioIsolationBuilderImpl {
    /// Create a new audio isolation builder.
    pub fn new() -> Self {
        Self {
            source: None,
            audio_data: None,
            isolate_voices: false,
            remove_background: false,
            reduce_noise: false,
            isolation_strength: None,
            output_format: None,
        }
    }
}

impl Default for AudioIsolationBuilderImpl {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioIsolationBuilder for AudioIsolationBuilderImpl {
    type Session = AudioIsolationSessionImpl;

    fn with_file(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    fn with_audio_data(mut self, data: Vec<u8>) -> Self {
        self.audio_data = Some(data);
        self
    }

    fn isolate_voices(mut self, isolate: bool) -> Self {
        self.isolate_voices = isolate;
        self
    }

    fn remove_background(mut self, remove: bool) -> Self {
        self.remove_background = remove;
        self
    }

    fn reduce_noise(mut self, reduce: bool) -> Self {
        self.reduce_noise = reduce;
        self
    }

    fn isolation_strength(mut self, strength: f32) -> Self {
        self.isolation_strength = Some(strength);
        self
    }

    fn output_format(mut self, format: AudioFormat) -> Self {
        self.output_format = Some(format);
        self
    }

    fn process<F, R>(self, matcher: F) -> impl Future<Output = R> + Send
    where
        F: FnOnce(Result<Self::Session, VoiceError>) -> R + Send + 'static,
    {
        async move {
            // Placeholder implementation
            if let Some(source) = self.source {
                let session = AudioIsolationSessionImpl { _source: source };
                matcher(Ok(session))
            } else {
                matcher(Err(VoiceError::Configuration(
                    "Missing audio source".to_string(),
                )))
            }
        }
    }
}
