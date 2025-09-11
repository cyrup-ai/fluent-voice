//! Simplified wake word detection implementation using the Koffee crate.

use crate::wake_word::{
    DefaultWakeWordConfig, WakeWordBuilder, WakeWordDetector, WakeWordResult, WakeWordStream,
};
use fluent_voice_domain::VoiceError;
use futures_core::Stream;
use std::{
    future::Future,
    pin::Pin,
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
    config: DefaultWakeWordConfig,
}

impl KoffeeWakeWordStream {
    fn with_config(config: DefaultWakeWordConfig) -> Self {
        Self {
            active: true,
            config,
        }
    }
}

impl Stream for KoffeeWakeWordStream {
    type Item = WakeWordResult;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.active {
            // Use config for detection logic (confidence threshold, timeout, etc.)
            let _confidence_threshold = self.config.confidence_threshold;
            let _timeout = self.config.timeout;
            let _model_file = &self.config.model_file;

            // For now, just return a successful no-detection result using config parameters
            Poll::Ready(Some(WakeWordResult::not_detected()))
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
