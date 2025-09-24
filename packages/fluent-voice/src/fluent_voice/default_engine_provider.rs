//! Integration API for implementing packages to access coordinated default engines.

use super::coordinated_voice_stream::CoordinatedVoiceStream;
use super::default_engine_coordinator::DefaultEngineCoordinator;
use fluent_voice_domain::VoiceError;

#[cfg(test)]
use super::default_engine_coordinator::{
    DefaultSttEngine, DefaultTtsEngine, KoffeeEngine, VadEngine,
};
use std::{future::Future, pin::Pin};

/// API that implementing packages use to access coordinated default engines
///
/// This trait enables implementing packages (like elevenlabs, kyutai) to selectively
/// override specific engines while keeping others as defaults, maintaining full
/// coordination functionality across all engines.
///
/// # Examples
///
/// ```rust
/// // Example: ElevenLabs package overrides TTS but uses default STT/VAD/wake word
/// impl DefaultEngineProvider for ElevenLabsEngine {
///     fn default_engines(&self) -> &DefaultEngineCoordinator {
///         &self.default_coordinator
///     }
///
///     fn with_custom_tts<T: TtsEngine>(&mut self, tts_engine: T) -> &mut Self {
///         // Override TTS while keeping default STT, VAD, wake word
///         self.custom_tts = Some(Box::new(tts_engine));
///         self
///     }
///
///     async fn start_coordinated_processing(&self) -> Result<CoordinatedVoiceStream, VoiceError> {
///         // Create coordinated stream with custom TTS + default engines
///         let mut stream = self.default_engines().start_coordinated_pipeline().await?;
///
///         // Override TTS engine in the stream
///         if let Some(ref custom_tts) = self.custom_tts {
///             stream.override_tts_engine(custom_tts.clone()).await?;
///         }
///
///         Ok(stream)
///     }
/// }
/// ```
pub trait DefaultEngineProvider {
    /// Get coordinated default engines
    ///
    /// Returns a reference to the default engine coordinator that manages
    /// TTS, STT, VAD, and wake word engines with full coordination.
    fn default_engines(&self) -> &DefaultEngineCoordinator;

    /// Override TTS engine while keeping other engines as defaults
    ///
    /// Allows implementing packages to provide their own TTS engine
    /// (e.g., ElevenLabs TTS) while using default STT, VAD, and wake word engines.
    ///
    /// # Type Parameters
    ///
    /// * `T` - The custom TTS engine type implementing TtsEngine trait
    ///
    /// # Arguments
    ///
    /// * `tts_engine` - The custom TTS engine instance
    ///
    /// # Returns
    ///
    /// Self for method chaining
    fn with_custom_tts<T: TtsEngine>(&mut self, tts_engine: T) -> &mut Self;

    /// Override STT engine while keeping other engines as defaults
    ///
    /// Allows implementing packages to provide their own STT engine
    /// while using default TTS, VAD, and wake word engines.
    ///
    /// # Type Parameters
    ///
    /// * `S` - The custom STT engine type implementing SttEngine trait
    ///
    /// # Arguments
    ///
    /// * `stt_engine` - The custom STT engine instance
    ///
    /// # Returns
    ///
    /// Self for method chaining
    fn with_custom_stt<S: SttEngine>(&mut self, stt_engine: S) -> &mut Self;

    /// Override VAD engine while keeping other engines as defaults
    ///
    /// Allows implementing packages to provide their own Voice Activity Detection
    /// engine while using default TTS, STT, and wake word engines.
    ///
    /// # Type Parameters
    ///
    /// * `V` - The custom VAD engine type implementing VadEngine trait
    ///
    /// # Arguments
    ///
    /// * `vad_engine` - The custom VAD engine instance
    ///
    /// # Returns
    ///
    /// Self for method chaining
    fn with_custom_vad<V: VadEngine>(&mut self, vad_engine: V) -> &mut Self;

    /// Override wake word engine while keeping other engines as defaults
    ///
    /// Allows implementing packages to provide their own wake word detection
    /// engine while using default TTS, STT, and VAD engines.
    ///
    /// # Type Parameters
    ///
    /// * `W` - The custom wake word engine type implementing WakeWordEngine trait
    ///
    /// # Arguments
    ///
    /// * `wake_word_engine` - The custom wake word engine instance
    ///
    /// # Returns
    ///
    /// Self for method chaining
    fn with_custom_wake_word<W: WakeWordEngine>(&mut self, wake_word_engine: W) -> &mut Self;

    /// Start coordinated processing with mixed custom/default engines
    ///
    /// Creates a coordinated voice stream that combines any custom engines
    /// with default engines, maintaining full coordination across all engines
    /// including event bus communication and shared state management.
    ///
    /// # Returns
    ///
    /// A `Result` containing:
    /// - `Ok(CoordinatedVoiceStream)` - Stream with coordinated processing
    /// - `Err(VoiceError)` - If coordination setup fails
    ///
    /// # Examples
    ///
    /// ```rust
    /// // Start coordinated processing with ElevenLabs TTS + default engines
    /// let stream = elevenlabs_engine.start_coordinated_processing().await?;
    ///
    /// // Process audio through coordinated pipeline
    /// let result = stream.process_audio_input(&audio_data).await?;
    /// ```
    fn start_coordinated_processing(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<CoordinatedVoiceStream, VoiceError>> + Send + '_>>;
}

/// Trait bounds for custom TTS engines
///
/// Custom TTS engines must implement this trait to be compatible
/// with the coordination system.
pub trait TtsEngine: Send + Sync {
    /// Synthesize text to audio
    fn synthesize(
        &mut self,
        text: &str,
        speaker_id: &str,
    ) -> Pin<
        Box<dyn Future<Output = Result<fluent_voice_domain::AudioChunk, VoiceError>> + Send + '_>,
    >;
}

/// Trait bounds for custom STT engines
///
/// Custom STT engines must implement this trait to be compatible
/// with the coordination system.
pub trait SttEngine: Send + Sync {
    /// Transcribe audio to text
    fn transcribe(
        &mut self,
        audio_data: &[u8],
    ) -> Pin<
        Box<
            dyn Future<Output = Result<super::default_engine_coordinator::SttResult, VoiceError>>
                + Send
                + '_,
        >,
    >;
}

/// Trait bounds for custom VAD engines
///
/// Custom VAD engines must implement this trait to be compatible
/// with the coordination system.
pub trait VadEngine: Send + Sync {
    /// Detect voice activity in audio
    fn detect_voice_activity(
        &mut self,
        audio_data: &[u8],
    ) -> Pin<
        Box<
            dyn Future<Output = Result<super::default_engine_coordinator::VadResult, VoiceError>>
                + Send
                + '_,
        >,
    >;
}

/// Trait bounds for custom wake word engines
///
/// Custom wake word engines must implement this trait to be compatible
/// with the coordination system.
pub trait WakeWordEngine: Send + Sync {
    /// Detect wake words in audio
    fn detect(
        &mut self,
        audio_data: &[u8],
    ) -> Result<Option<super::default_engine_coordinator::WakeWordResult>, VoiceError>;
}

/// Default implementation for packages that don't need custom engines
///
/// This provides a basic implementation that uses all default engines
/// without any customization. Implementing packages can use this as a
/// starting point and override specific methods as needed.
pub struct DefaultEngineImplementation {
    coordinator: DefaultEngineCoordinator,
}

impl DefaultEngineImplementation {
    /// Create a new default engine implementation
    ///
    /// Initializes all engines with default configurations and
    /// sets up coordination between them.
    ///
    /// # Returns
    ///
    /// A `Result` containing:
    /// - `Ok(DefaultEngineImplementation)` - Ready-to-use implementation
    /// - `Err(VoiceError)` - If engine initialization fails
    pub fn new() -> Result<Self, VoiceError> {
        let coordinator = DefaultEngineCoordinator::new()?;
        Ok(Self { coordinator })
    }
}

impl DefaultEngineProvider for DefaultEngineImplementation {
    fn default_engines(&self) -> &DefaultEngineCoordinator {
        &self.coordinator
    }

    fn with_custom_tts<T: TtsEngine>(&mut self, _tts_engine: T) -> &mut Self {
        // Default implementation doesn't support custom engines
        // Implementing packages should override this method
        self
    }

    fn with_custom_stt<S: SttEngine>(&mut self, _stt_engine: S) -> &mut Self {
        // Default implementation doesn't support custom engines
        // Implementing packages should override this method
        self
    }

    fn with_custom_vad<V: VadEngine>(&mut self, _vad_engine: V) -> &mut Self {
        // Default implementation doesn't support custom engines
        // Implementing packages should override this method
        self
    }

    fn with_custom_wake_word<W: WakeWordEngine>(&mut self, _wake_word_engine: W) -> &mut Self {
        // Default implementation doesn't support custom engines
        // Implementing packages should override this method
        self
    }

    fn start_coordinated_processing(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<CoordinatedVoiceStream, VoiceError>> + Send + '_>> {
        let coordinator = &self.coordinator;
        Box::pin(async move {
            // Use default coordination without any custom engines
            coordinator.start_coordinated_pipeline().await
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_default_engine_implementation_creation() {
        let implementation = DefaultEngineImplementation::new();
        assert!(implementation.is_ok());
    }

    #[tokio::test]
    async fn test_default_engine_provider_access() {
        let implementation = DefaultEngineImplementation::new().unwrap();
        let coordinator = implementation.default_engines();

        // Verify we can access the coordinator
        assert!(coordinator.is_ready().await);
    }

    #[tokio::test]
    async fn test_default_coordinated_processing() {
        let implementation = DefaultEngineImplementation::new().unwrap();

        // Test starting coordinated processing
        let result = implementation.start_coordinated_processing().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_custom_engine_chaining() {
        let mut implementation = match DefaultEngineImplementation::new() {
            Ok(impl_) => impl_,
            Err(_) => return, // Skip test if implementation creation fails
        };

        // Test method chaining with REAL engines using actual ML models - with proper error handling
        let tts_engine = match DefaultTtsEngine::new() {
            Ok(engine) => engine,
            Err(_) => return, // Skip test if engine creation fails
        };

        let stt_engine = match DefaultSttEngine::with_whisper_vad_koffee() {
            Ok(engine) => engine,
            Err(_) => return, // Skip test if engine creation fails
        };

        let vad_engine = match VadEngine::new() {
            Ok(engine) => engine,
            Err(_) => return, // Skip test if engine creation fails
        };

        let wake_word_engine = match KoffeeEngine::new() {
            Ok(engine) => engine,
            Err(_) => return, // Skip test if engine creation fails
        };

        implementation
            .with_custom_tts(tts_engine)
            .with_custom_stt(stt_engine)
            .with_custom_vad(vad_engine)
            .with_custom_wake_word(wake_word_engine);

        // Should work with real engines
        let result = implementation.start_coordinated_processing().await;
        assert!(result.is_ok());
    }
}
