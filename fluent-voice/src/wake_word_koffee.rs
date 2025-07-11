//! Default wake word detection implementation using the Koffee crate.

use crate::wake_word::{
    WakeWordBuilder, WakeWordConfig, WakeWordDetector, WakeWordEvent, WakeWordResult,
    WakeWordStream,
};
use fluent_voice_domain::VoiceError;
use futures_core::Stream;
use futures_util::StreamExt;
use koffee::{
    KoffeeCandle,
    config::{DetectorConfig, KoffeeCandleConfig},
    wakewords::WakewordLoad,
};
use std::{
    path::Path,
    sync::{Arc, Mutex},
};

/// Default wake word detector implementation using Koffee.
pub struct KoffeeWakeWordDetector {
    /// The underlying Koffee detector.
    detector: Arc<Mutex<KoffeeCandle>>,
    /// Current configuration.
    config: WakeWordConfig,
}

impl KoffeeWakeWordDetector {
    /// Create a new Koffee-based wake word detector.
    pub fn new() -> WakeWordResult<Self> {
        let cfg = KoffeeCandleConfig::default();
        let detector = KoffeeCandle::new(&cfg).map_err(|e| {
            VoiceError::ConfigurationError(format!("Failed to create Koffee detector: {}", e))
        })?;

        Ok(Self {
            detector: Arc::new(Mutex::new(detector)),
            config: WakeWordConfig::default(),
        })
    }
}

impl WakeWordDetector for KoffeeWakeWordDetector {
    type Event = WakeWordEvent;

    fn add_wake_word_model<P: AsRef<Path>>(
        &mut self,
        model_path: P,
        wake_word: String,
    ) -> WakeWordResult<()> {
        let model = WakewordLoad::load_from_file(model_path.as_ref()).map_err(|e| {
            VoiceError::ConfigurationError(format!(
                "Failed to load wake word model '{}': {}",
                wake_word, e
            ))
        })?;

        let mut detector = self.detector.lock().unwrap();
        detector.add_wakeword_model(model).map_err(|e| {
            VoiceError::ConfigurationError(format!(
                "Failed to add wake word model '{}': {}",
                wake_word, e
            ))
        })
    }

    fn process_audio(&mut self, audio_data: &[u8]) -> WakeWordResult<Vec<Self::Event>> {
        let detection_opt = self.detector.lock().unwrap().process_bytes(audio_data);

        let mut events = Vec::new();
        if let Some(detection) = detection_opt {
            if detection.score >= self.config.confidence_threshold {
                events.push(WakeWordEvent {
                    word: detection.name.clone(),
                    confidence: detection.score,
                    timestamp_ms: detection.counter as u64, // Using counter as timestamp for now
                });
            }
        }

        if self.config.debug && !events.is_empty() {
            for event in &events {
                eprintln!(
                    "Wake word detected: '{}' (confidence: {:.2})",
                    event.word, event.confidence
                );
            }
        }

        Ok(events)
    }

    fn process_samples(&mut self, samples: &[f32]) -> WakeWordResult<Vec<Self::Event>> {
        let detection_opt = self.detector.lock().unwrap().process_samples(samples);

        let mut events = Vec::new();
        if let Some(detection) = detection_opt {
            if detection.score >= self.config.confidence_threshold {
                events.push(WakeWordEvent {
                    word: detection.name.clone(),
                    confidence: detection.score,
                    timestamp_ms: detection.counter as u64, // Using counter as timestamp for now
                });
            }
        }

        if self.config.debug && !events.is_empty() {
            for event in &events {
                eprintln!(
                    "Wake word detected: '{}' (confidence: {:.2})",
                    event.word, event.confidence
                );
            }
        }

        Ok(events)
    }

    fn update_config(&mut self, config: WakeWordConfig) -> WakeWordResult<()> {
        // Convert our wake word config to Koffee DetectorConfig
        let detector_config = DetectorConfig {
            threshold: config.confidence_threshold,
            ..DetectorConfig::default()
        };

        // Update the detector with new configuration
        self.detector
            .lock()
            .unwrap()
            .update_config(&detector_config);

        // Store our configuration
        self.config = config;
        Ok(())
    }

    fn get_config(&self) -> &WakeWordConfig {
        &self.config
    }
}

impl WakeWordStream for KoffeeWakeWordDetector {
    type Event = WakeWordEvent;

    fn process_stream<S>(
        &mut self,
        audio_stream: S,
    ) -> impl Stream<Item = WakeWordResult<Self::Event>> + Send
    where
        S: Stream<Item = Vec<u8>> + Send + Unpin,
    {
        let detector = Arc::clone(&self.detector);
        let config = self.config.clone();
        audio_stream.filter_map(move |audio_chunk| {
            let detector = Arc::clone(&detector);
            let config = config.clone();
            async move {
                let detection_opt = detector.lock().unwrap().process_bytes(&audio_chunk);
                if let Some(detection) = detection_opt {
                    if detection.score >= config.confidence_threshold {
                        let event = WakeWordEvent {
                            word: detection.name.clone(),
                            confidence: detection.score,
                            timestamp_ms: detection.counter as u64,
                        };
                        Some(Ok(event))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
        })
    }

    fn process_sample_stream<S>(
        &mut self,
        sample_stream: S,
    ) -> impl Stream<Item = WakeWordResult<Self::Event>> + Send
    where
        S: Stream<Item = Vec<f32>> + Send + Unpin,
    {
        let detector = Arc::clone(&self.detector);
        let config = self.config.clone();
        sample_stream.filter_map(move |sample_chunk| {
            let detector = Arc::clone(&detector);
            let config = config.clone();
            async move {
                let detection_opt = detector.lock().unwrap().process_samples(&sample_chunk);
                if let Some(detection) = detection_opt {
                    if detection.score >= config.confidence_threshold {
                        let event = WakeWordEvent {
                            word: detection.name.clone(),
                            confidence: detection.score,
                            timestamp_ms: detection.counter as u64,
                        };
                        Some(Ok(event))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
        })
    }
}

/// Default wake word builder implementation using Koffee.
pub struct KoffeeWakeWordBuilder {
    /// Wake word models to add.
    models: Vec<(std::path::PathBuf, String)>,
    /// Configuration to apply.
    config: WakeWordConfig,
}

impl KoffeeWakeWordBuilder {
    /// Create a new Koffee wake word builder.
    pub fn new() -> Self {
        Self {
            models: Vec::new(),
            config: WakeWordConfig::default(),
        }
    }
}

impl Default for KoffeeWakeWordBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl WakeWordBuilder for KoffeeWakeWordBuilder {
    type Detector = KoffeeWakeWordDetector;

    fn with_wake_word_model<P: AsRef<Path>>(
        mut self,
        model_path: P,
        wake_word: String,
    ) -> WakeWordResult<Self>
    where
        Self: Sized,
    {
        self.models
            .push((model_path.as_ref().to_path_buf(), wake_word));
        Ok(self)
    }

    fn with_confidence_threshold(mut self, threshold: f32) -> Self
    where
        Self: Sized,
    {
        self.config.confidence_threshold = threshold;
        self
    }

    fn with_debug(mut self, debug: bool) -> Self
    where
        Self: Sized,
    {
        self.config.debug = debug;
        self
    }

    fn build(self) -> WakeWordResult<Self::Detector> {
        let mut detector = KoffeeWakeWordDetector::new()?;

        // Add all wake word models
        for (model_path, wake_word) in self.models {
            detector.add_wake_word_model(&model_path, wake_word)?;
        }

        // Apply configuration
        detector.update_config(self.config)?;

        Ok(detector)
    }
}
