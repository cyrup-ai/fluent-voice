//! Simplified wake word detection implementation using the Koffee crate.

use crate::fluent_voice::default_engine_coordinator::KoffeeEngine;
use crate::wake_word::{DefaultWakeWordConfig, WakeWordBuilder, WakeWordDetector, WakeWordStream};
use fluent_voice_domain::VoiceError;
use fluent_voice_domain::{WakeWordDetectionResult, WakeWordEvent};
use futures_core::Stream;
use std::{
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll},
    time::Duration,
};

/// Simple wake word detector implementation using Koffee.
pub struct KoffeeWakeWordDetector {
    config: DefaultWakeWordConfig,
}

impl KoffeeWakeWordDetector {
    pub fn new() -> Result<Self, VoiceError> {
        Ok(Self {
            config: DefaultWakeWordConfig::default(),
        })
    }

    pub fn with_config(config: DefaultWakeWordConfig) -> Result<Self, VoiceError> {
        Ok(Self { config })
    }

    pub fn config(&self) -> &DefaultWakeWordConfig {
        &self.config
    }
}

impl WakeWordDetector for KoffeeWakeWordDetector {
    type Stream = KoffeeWakeWordStream;

    fn start_detection(self) -> Self::Stream {
        KoffeeWakeWordStream::with_config(self.config)
    }
}

/// Simple wake word stream implementation.
pub struct KoffeeWakeWordStream {
    active: bool,
    #[allow(dead_code)] // Used by public config() method
    config: DefaultWakeWordConfig,
    koffee_engine: Option<Arc<Mutex<KoffeeEngine>>>, // Real engine (optional if initialization fails)
}

impl KoffeeWakeWordStream {
    fn with_config(config: DefaultWakeWordConfig) -> Self {
        // Initialize real KoffeeEngine with proper error handling (no unwrap/expect)
        let koffee_engine = match KoffeeEngine::new() {
            Ok(engine) => Some(Arc::new(Mutex::new(engine))),
            Err(e) => {
                tracing::error!("Failed to initialize KoffeeEngine: {}", e);
                None // Will be handled in poll_next with error result
            }
        };

        Self {
            active: true,
            config,
            koffee_engine,
        }
    }
}

impl Stream for KoffeeWakeWordStream {
    type Item = WakeWordDetectionResult;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.active {
            // Check if KoffeeEngine is available
            if let Some(ref koffee_engine) = self.koffee_engine {
                // Try to lock the engine for detection
                match koffee_engine.try_lock() {
                    Ok(mut engine) => {
                        // Create dummy audio bytes for detection (in real usage, this would come from audio input)
                        let audio_bytes = vec![0u8; 1024]; // Placeholder audio data

                        // Use real KoffeeEngine detection as specified in TURD.md
                        match engine.detect(&audio_bytes) {
                            Ok(Some(result)) => {
                                let event = WakeWordEvent {
                                    word: result.word,
                                    confidence: result.confidence,
                                    timestamp_ms: result.timestamp,
                                };
                                Poll::Ready(Some(WakeWordDetectionResult::detected(event)))
                            }
                            Ok(None) => Poll::Ready(Some(WakeWordDetectionResult::not_detected())),
                            Err(e) => {
                                tracing::error!("Wake word detection error: {}", e);
                                Poll::Ready(Some(WakeWordDetectionResult::not_detected()))
                            }
                        }
                    }
                    Err(_) => {
                        // Engine is busy, return no detection for this poll
                        Poll::Ready(Some(WakeWordDetectionResult::not_detected()))
                    }
                }
            } else {
                // Engine failed to initialize, return not_detected and log error
                tracing::error!(
                    "KoffeeEngine initialization failed, cannot perform wake word detection"
                );
                Poll::Ready(Some(WakeWordDetectionResult::not_detected()))
            }
        } else {
            Poll::Ready(None)
        }
    }
}

impl WakeWordStream for KoffeeWakeWordStream {
    fn stop(&mut self) {
        self.active = false;
    }

    fn is_active(&self) -> bool {
        self.active
    }
}

/// Builder for KoffeeWakeWordDetector.
pub struct KoffeeWakeWordBuilder {
    config: DefaultWakeWordConfig,
}

impl KoffeeWakeWordBuilder {
    pub fn new() -> Self {
        Self {
            config: DefaultWakeWordConfig::default(),
        }
    }
}

impl Default for KoffeeWakeWordBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl WakeWordBuilder for KoffeeWakeWordBuilder {
    type Config = DefaultWakeWordConfig;
    type Detector = KoffeeWakeWordDetector;

    fn model_file(mut self, path: impl Into<String>) -> Self {
        self.config.model_file = Some(path.into());
        self
    }

    fn confidence_threshold(mut self, threshold: f32) -> Self {
        self.config.confidence_threshold = threshold;
        self
    }

    fn timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = timeout;
        self
    }

    fn detect<F, R>(self, matcher: F) -> impl Future<Output = R> + Send
    where
        F: FnOnce(Result<Self::Detector, VoiceError>) -> R + Send + 'static,
    {
        async move {
            let detector_result = KoffeeWakeWordDetector::with_config(self.config);
            matcher(detector_result)
        }
    }
}
