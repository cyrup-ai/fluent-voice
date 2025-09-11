//! Wake word domain objects and traits.

use crate::VoiceError;
use futures::Stream;
use std::path::PathBuf;
use std::pin::Pin;

/// Result type for wake word operations
pub type WakeWordResult<T> = Result<T, VoiceError>;

/// Wake word detection event with timestamp
#[derive(Debug, Clone)]
pub struct WakeWordEvent {
    pub wake_word: String,
    pub confidence: f32,
    pub timestamp_ms: u64,
}

/// Wake word detector trait - core detection functionality
pub trait WakeWordDetector: Send + Sync {
    type Event;
    type Stream: Stream<Item = WakeWordResult<Self::Event>> + Send;

    fn add_wake_word_model(&mut self, model_path: PathBuf) -> WakeWordResult<()>;
    fn process_audio(&mut self, audio_data: &[f32]) -> WakeWordResult<Option<Self::Event>>;
    fn process_samples(&mut self, samples: &[f32]) -> WakeWordResult<Vec<Self::Event>>;
    fn update_config(&mut self, config: String) -> WakeWordResult<()>;
    fn get_config(&self) -> String;
    fn start_detection(&mut self) -> WakeWordResult<Self::Stream>;
}

/// Wake word streaming interface
pub trait WakeWordStream: Stream + Send + Sync {
    type Event;

    fn process_stream(
        &mut self,
        audio_stream: Pin<Box<dyn Stream<Item = Vec<f32>> + Send>>,
    ) -> WakeWordResult<()>;
    fn process_sample_stream(&mut self, samples: Vec<f32>) -> WakeWordResult<Vec<Self::Event>>;
    fn stop(&mut self) -> WakeWordResult<()>;
    fn is_active(&self) -> bool;
}

/// Wake word builder pattern interface
pub trait WakeWordBuilder: Send + Sync {
    type Config;
    type Detector: WakeWordDetector;

    fn model_file(&mut self, path: PathBuf) -> &mut Self;
    fn confidence_threshold(&mut self, threshold: f32) -> &mut Self;
    fn timeout(&mut self, timeout_ms: u64) -> &mut Self;
    fn detect(&mut self) -> WakeWordResult<Self::Detector>;
    fn with_wake_word_model(&mut self, model_path: PathBuf) -> &mut Self;
    fn with_confidence_threshold(&mut self, threshold: f32) -> &mut Self;
    fn with_debug(&mut self, debug: bool) -> &mut Self;
    fn build(&self) -> WakeWordResult<Self::Detector>;
}
