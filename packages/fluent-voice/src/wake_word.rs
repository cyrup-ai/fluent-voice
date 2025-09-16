//! Wake word detection traits and types.

use core::future::Future;
use fluent_voice_domain::{VoiceError, WakeWordDetectionResult};
use futures_core::Stream;
use std::time::Duration;

/// Builder trait for wake word detection functionality.
pub trait WakeWordBuilder: Sized + Send {
    /// The configuration type used by this builder.
    type Config: WakeWordConfig;
    /// The detector type produced by this builder.
    type Detector: WakeWordDetector;

    /// Set the wake word model file path.
    fn model_file(self, path: impl Into<String>) -> Self;

    /// Set the confidence threshold.
    fn confidence_threshold(self, threshold: f32) -> Self;

    /// Set the detection timeout.
    fn timeout(self, timeout: Duration) -> Self;

    /// Build the wake word detector with a matcher closure.
    fn detect<F, R>(self, matcher: F) -> impl Future<Output = R> + Send
    where
        F: FnOnce(Result<Self::Detector, VoiceError>) -> R + Send + 'static;
}

/// Configuration trait for wake word detection.
pub trait WakeWordConfig: Send + Sync + Clone {
    /// Get the confidence threshold.
    fn confidence_threshold(&self) -> f32;

    /// Get the detection timeout.
    fn timeout(&self) -> Duration;

    /// Get the model file path.
    fn model_file(&self) -> Option<&str>;
}

/// Default configuration implementation for wake word detection.
#[derive(Debug, Clone)]
pub struct DefaultWakeWordConfig {
    pub confidence_threshold: f32,
    pub timeout: Duration,
    pub model_file: Option<String>,
    pub debug: bool,
}

impl Default for DefaultWakeWordConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.5,
            timeout: Duration::from_secs(30),
            model_file: None,
            debug: false,
        }
    }
}

impl WakeWordConfig for DefaultWakeWordConfig {
    fn confidence_threshold(&self) -> f32 {
        self.confidence_threshold
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }

    fn model_file(&self) -> Option<&str> {
        self.model_file.as_deref()
    }
}

/// Detector trait for wake word detection.
pub trait WakeWordDetector: Send {
    /// The stream type produced by this detector.
    type Stream: WakeWordStream;

    /// Start detection and return a stream of wake word events.
    fn start_detection(self) -> Self::Stream;
}

/// Stream trait for wake word detection events.
pub trait WakeWordStream: Stream<Item = WakeWordDetectionResult> + Send + Unpin {
    /// Stop the detection stream.
    fn stop(&mut self);

    /// Check if the stream is active.
    fn is_active(&self) -> bool;
}

/// Extension trait for wake word builders.
///
/// This trait provides the static method for starting a new wake word detection.
pub trait WakeWordConversationExt {
    /// Begin a new wake word builder.
    fn builder() -> impl WakeWordBuilder;
}
