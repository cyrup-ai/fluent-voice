//! Wake word detection traits and types.

use futures_core::Stream;
use std::path::Path;

/// Result type for wake word detection operations.
pub type WakeWordResult<T> = Result<T, crate::voice_error::VoiceError>;

/// Wake word detection event containing the detected wake word and confidence.
#[derive(Debug, Clone)]
pub struct WakeWordEvent {
    /// The wake word that was detected.
    pub word: String,
    /// Confidence score (0.0 to 1.0).
    pub confidence: f32,
    /// Timestamp when the wake word was detected (milliseconds).
    pub timestamp_ms: u64,
}

/// Configuration for wake word detection.
#[derive(Debug, PartialEq, Clone)]
pub struct WakeWordConfig {
    /// Minimum confidence threshold for detection.
    pub confidence_threshold: f32,
    /// Enable debug output.
    pub debug: bool,
}

impl Default for WakeWordConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.5,
            debug: false,
        }
    }
}

/// Core trait for wake word detection functionality.
///
/// This trait defines the interface for wake word detectors that can
/// process audio data and emit wake word detection events.
pub trait WakeWordDetector: Send + Sync {
    /// The type of wake word events this detector emits.
    type Event: Send + Sync + Clone;

    /// Add a wake word model from a file path.
    ///
    /// # Arguments
    /// * `model_path` - Path to the wake word model file
    /// * `wake_word` - The wake word string associated with this model
    ///
    /// # Returns
    /// Result indicating success or failure of model loading.
    fn add_wake_word_model<P: AsRef<Path>>(
        &mut self,
        model_path: P,
        wake_word: String,
    ) -> WakeWordResult<()>;

    /// Process raw audio bytes and return detection events.
    ///
    /// # Arguments
    /// * `audio_data` - Raw audio bytes to process
    ///
    /// # Returns
    /// Stream of wake word detection events.
    fn process_audio(&mut self, audio_data: &[u8]) -> WakeWordResult<Vec<Self::Event>>;

    /// Process audio samples and return detection events.
    ///
    /// # Arguments
    /// * `samples` - Audio samples as f32 values
    ///
    /// # Returns
    /// Stream of wake word detection events.
    fn process_samples(&mut self, samples: &[f32]) -> WakeWordResult<Vec<Self::Event>>;

    /// Update the detector configuration.
    ///
    /// # Arguments
    /// * `config` - New configuration to apply
    fn update_config(&mut self, config: WakeWordConfig) -> WakeWordResult<()>;

    /// Get the current configuration.
    fn get_config(&self) -> &WakeWordConfig;
}

/// Builder trait for configuring wake word detection.
pub trait WakeWordBuilder: Send + Sync {
    /// The concrete wake word detector type this builder creates.
    type Detector: WakeWordDetector;

    /// Add a wake word model from a file path.
    ///
    /// # Arguments
    /// * `model_path` - Path to the wake word model file
    /// * `wake_word` - The wake word string associated with this model
    fn with_wake_word_model<P: AsRef<Path>>(
        self,
        model_path: P,
        wake_word: String,
    ) -> WakeWordResult<Self>
    where
        Self: Sized;

    /// Set the confidence threshold for wake word detection.
    ///
    /// # Arguments
    /// * `threshold` - Confidence threshold (0.0 to 1.0)
    fn with_confidence_threshold(self, threshold: f32) -> Self
    where
        Self: Sized;

    /// Enable or disable debug output.
    ///
    /// # Arguments
    /// * `debug` - Whether to enable debug output
    fn with_debug(self, debug: bool) -> Self
    where
        Self: Sized;

    /// Build the wake word detector.
    ///
    /// # Returns
    /// The configured wake word detector instance.
    fn build(self) -> WakeWordResult<Self::Detector>;
}

/// Stream-based wake word detection trait.
///
/// This trait provides methods for processing continuous audio streams
/// and emitting wake word detection events as they occur.
pub trait WakeWordStream: Send + Sync {
    /// The type of events emitted by this stream processor.
    type Event: Send + Sync + Clone;

    /// Process a continuous audio stream and emit wake word events.
    ///
    /// # Arguments
    /// * `audio_stream` - Stream of audio data chunks
    ///
    /// # Returns
    /// Stream of wake word detection events.
    fn process_stream<S>(
        &mut self,
        audio_stream: S,
    ) -> impl Stream<Item = WakeWordResult<Self::Event>> + Send
    where
        S: Stream<Item = Vec<u8>> + Send + Unpin;

    /// Process a continuous sample stream and emit wake word events.
    ///
    /// # Arguments
    /// * `sample_stream` - Stream of audio sample chunks
    ///
    /// # Returns
    /// Stream of wake word detection events.
    fn process_sample_stream<S>(
        &mut self,
        sample_stream: S,
    ) -> impl Stream<Item = WakeWordResult<Self::Event>> + Send
    where
        S: Stream<Item = Vec<f32>> + Send + Unpin;
}
