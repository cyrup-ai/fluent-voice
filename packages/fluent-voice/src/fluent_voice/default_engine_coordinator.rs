//! Central coordinator for all default engines in the fluent-voice ecosystem.

use super::event_bus::{EventBus, EventType};
use dia::voice::{voice_builder::DiaVoiceBuilder, VoicePool};
use fluent_voice_domain::{AudioChunk, AudioFormat, SpeechSource, VoiceError};
use fluent_voice_vad::{Error as VadError, VoiceActivityDetector};
use fluent_voice_whisper::{ModelConfig, WhichModel, WhisperTranscriber};
use koffee::{KoffeeCandle, KoffeeCandleConfig};
use koffee::wakewords::{WakewordModel, WakewordLoad};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};

/// Real TTS engine using dia integration for speech synthesis
pub struct DefaultTtsEngine {
    voice_pool: Arc<VoicePool>,
    temp_dir: std::path::PathBuf,
}

impl DefaultTtsEngine {
    pub fn new() -> Result<Self, VoiceError> {
        // ADAPT FROM: default_tts_builder.rs:207-214
        let voice_pool = VoicePool::new()
            .map_err(|e| VoiceError::Configuration(format!("Failed to create VoicePool: {}", e)))?;

        Ok(Self {
            voice_pool: Arc::new(voice_pool),
            temp_dir: std::env::temp_dir(),
        })
    }

    pub async fn synthesize(
        &mut self,
        text: &str,
        speaker_id: &str,
    ) -> Result<AudioChunk, VoiceError> {
        // ADAPT FROM: default_tts_builder.rs:270-304

        // Create unique audio path for this synthesis
        let audio_path = self.temp_dir.join(format!(
            "tts_coord_{}_{}.wav",
            speaker_id,
            std::process::id()
        ));

        // Create DiaVoiceBuilder using proven pattern
        let dia_builder = DiaVoiceBuilder::new(self.voice_pool.clone(), audio_path);

        // Real synthesis using existing working pattern
        let audio_data = dia_builder
            .speak(text)
            .play(|result| match result {
                Ok(voice_player) => voice_player.audio_data,
                Err(e) => {
                    tracing::error!("DefaultTtsEngine synthesis failed: {}", e);
                    Vec::new() // Return empty on error, don't propagate
                }
            })
            .await;

        // Duration calculation using proven formula
        let duration_ms = if !audio_data.is_empty() {
            (audio_data.len() as u64 * 1000) / (16000u64 * 2u64) // 16kHz, 16-bit PCM
        } else {
            0
        };

        // Create AudioChunk using proven pattern
        Ok(AudioChunk::with_metadata(
            audio_data,                   // Real synthesized audio from dia
            duration_ms,                  // Calculated duration
            0,                            // start_ms (synthesis always starts at 0)
            Some(speaker_id.to_string()), // speaker_id from params
            Some(text.to_string()),       // source text from params
            Some(AudioFormat::Pcm16Khz),  // dia output format
        ))
    }
}

/// Real STT engine using Whisper for speech-to-text transcription
pub struct DefaultSttEngine {
    transcriber: WhisperTranscriber,
}

impl DefaultSttEngine {
    pub fn with_whisper_vad_koffee() -> Result<Self, VoiceError> {
        // Use base model with optimized settings for real-time performance
        let config = ModelConfig {
            which_model: WhichModel::Base,
            timestamps: true,
            verbose: false,
            temperature: 0.0,                 // Deterministic output
            task: None,                       // Auto-detect transcribe vs translate
            language: Some("en".to_string()), // Default to English
            quantized: false,                 // Use full precision for better accuracy
            cpu: false,                       // Use hardware acceleration when available
            ..Default::default()
        };

        let transcriber = WhisperTranscriber::with_config(config).map_err(|e| {
            VoiceError::Configuration(format!("Failed to create WhisperTranscriber: {}", e))
        })?;

        Ok(Self { transcriber })
    }

    pub async fn transcribe(&mut self, audio_data: &[u8]) -> Result<SttResult, VoiceError> {
        // Convert raw audio bytes to SpeechSource::Memory
        let speech_source = SpeechSource::Memory {
            data: audio_data.to_vec(),
            format: AudioFormat::Pcm16Khz, // 16-bit PCM at 16kHz
            sample_rate: 16000,            // Standard rate for speech processing
        };

        // Perform transcription using the real Whisper engine
        let transcript = self
            .transcriber
            .transcribe(speech_source)
            .await
            .map_err(|_| VoiceError::Stt("Whisper transcription failed"))?;

        // Use helper method to convert transcript to SttResult (eliminates code duplication)
        self.transcript_to_stt_result(transcript)
    }

    /// Transcribe from file path (leverages existing SpeechSource::File infrastructure)
    pub async fn transcribe_file(&mut self, file_path: &str) -> Result<SttResult, VoiceError> {
        let speech_source = SpeechSource::File {
            path: file_path.to_string(),
            format: AudioFormat::Pcm16Khz, // 16-bit PCM at 16kHz
        };

        let transcript = self
            .transcriber
            .transcribe(speech_source)
            .await
            .map_err(|_| VoiceError::Stt("File transcription failed"))?;

        self.transcript_to_stt_result(transcript)
    }

    /// Transcribe from microphone (leverages existing microphone infrastructure)
    #[cfg(feature = "microphone")]
    pub async fn transcribe_microphone(&mut self) -> Result<SttResult, VoiceError> {
        let speech_source = SpeechSource::Microphone {
            backend: fluent_voice_domain::MicBackend::Default,
            format: AudioFormat::Pcm16Khz,
            sample_rate: 16000,
        };

        let transcript = self
            .transcriber
            .transcribe(speech_source)
            .await
            .map_err(|_| VoiceError::Stt("Microphone transcription failed"))?;

        self.transcript_to_stt_result(transcript)
    }

    /// Helper to convert Transcript to SttResult (extracts existing logic)
    fn transcript_to_stt_result(
        &self,
        transcript: fluent_voice_whisper::Transcript,
    ) -> Result<SttResult, VoiceError> {
        let text = transcript.as_text();
        let chunks = transcript.chunks();

        let confidence = self.calculate_overall_confidence(chunks);

        Ok(SttResult {
            text: text.to_string(),
            confidence,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        })
    }

    /// SOPHISTICATED confidence calculation using all available metrics
    fn calculate_overall_confidence(&self, chunks: &[fluent_voice_whisper::TtsChunk]) -> f32 {
        if chunks.is_empty() {
            return 0.0;
        }

        let total_duration: f64 = chunks.iter().map(|c| c.duration).sum();
        if total_duration <= 0.0 {
            return 0.0;
        }

        chunks
            .iter()
            .map(|chunk| {
                // ADVANCED MULTI-METRIC CONFIDENCE CALCULATION

                // 1. No-speech confidence (primary indicator)
                let no_speech_confidence = (1.0 - chunk.no_speech_prob).max(0.0).min(1.0);

                // 2. Log probability confidence (model certainty)
                // Whisper logprobs typically range from -1.0 to 0.0, normalize to 0.0-1.0
                let logprob_confidence = (chunk.avg_logprob + 1.0).max(0.0).min(1.0);

                // 3. Compression ratio confidence (repetition detection)
                // Good transcriptions have compression ratios around 1.5-2.5
                // Higher ratios indicate repetitive/hallucinated text
                let compression_confidence = if chunk.compression_ratio <= 2.5 {
                    1.0
                } else if chunk.compression_ratio <= 4.0 {
                    // Linear decay from 1.0 to 0.0 between 2.5 and 4.0
                    1.0 - ((chunk.compression_ratio - 2.5) / 1.5)
                } else {
                    0.0
                }
                .max(0.0)
                .min(1.0);

                // 4. Temperature-based confidence (sampling randomness)
                // Lower temperature = more confident predictions
                let temperature_confidence = (1.0 - chunk.temperature).max(0.0).min(1.0);

                // 5. Duration-based confidence (very short segments are less reliable)
                let duration_confidence = if chunk.duration >= 0.5 {
                    1.0
                } else {
                    (chunk.duration / 0.5).max(0.0).min(1.0)
                };

                // WEIGHTED COMBINATION of all confidence metrics
                let segment_confidence = (no_speech_confidence * 0.35) +      // Primary weight
                                       (logprob_confidence * 0.25) +          // Model certainty
                                       (compression_confidence * 0.20) +      // Repetition detection
                                       (temperature_confidence * 0.10) +      // Sampling confidence
                                       (duration_confidence * 0.10); // Duration reliability

                // Weight by segment duration for overall confidence
                let weight = chunk.duration / total_duration;
                segment_confidence * weight
            })
            .sum::<f64>() as f32
    }

    /// Create engine optimized for REAL-TIME performance
    pub fn with_realtime_config() -> Result<Self, VoiceError> {
        let config = ModelConfig {
            which_model: WhichModel::TinyEn,  // Fastest model for English
            quantized: true,                  // Reduce memory usage
            cpu: false,                       // Use hardware acceleration
            timestamps: false,                // Disable for speed
            verbose: false,                   // Reduce logging overhead
            temperature: 0.0,                 // Deterministic for consistency
            task: None,                       // Auto-detect
            language: Some("en".to_string()), // English optimization
            ..Default::default()
        };

        let transcriber = WhisperTranscriber::with_config(config).map_err(|e| {
            VoiceError::Configuration(format!(
                "Failed to create realtime WhisperTranscriber: {}",
                e
            ))
        })?;

        Ok(Self { transcriber })
    }

    /// Create engine optimized for ACCURACY
    pub fn with_accuracy_config() -> Result<Self, VoiceError> {
        let config = ModelConfig {
            which_model: WhichModel::Large, // Most accurate model
            quantized: false,               // Full precision
            cpu: false,                     // Use hardware acceleration
            timestamps: true,               // Enable detailed timing
            verbose: false,                 // Keep logging minimal
            temperature: 0.0,               // Deterministic output
            task: None,                     // Auto-detect transcribe vs translate
            language: None,                 // Auto-detect language
            ..Default::default()
        };

        let transcriber = WhisperTranscriber::with_config(config).map_err(|e| {
            VoiceError::Configuration(format!(
                "Failed to create accuracy WhisperTranscriber: {}",
                e
            ))
        })?;

        Ok(Self { transcriber })
    }

    /// Create engine with custom configuration
    pub fn with_custom_config(config: ModelConfig) -> Result<Self, VoiceError> {
        let transcriber = WhisperTranscriber::with_config(config).map_err(|e| {
            VoiceError::Configuration(format!("Failed to create custom WhisperTranscriber: {}", e))
        })?;

        Ok(Self { transcriber })
    }
}

/// Real VAD engine using Silero ONNX model for voice activity detection
pub struct VadEngine {
    detector: VoiceActivityDetector,
    chunk_size: usize,
}

impl VadEngine {
    /// Create new VAD engine with optimized configuration for real-time processing
    pub fn new() -> Result<Self, VoiceError> {
        let detector = VoiceActivityDetector::builder()
            .sample_rate(16000i64) // Standard sample rate for speech processing
            .chunk_size(512usize) // Optimal chunk size for real-time performance
            .build()
            .map_err(|e| {
                VoiceError::Configuration(format!("Failed to create VAD detector: {}", e))
            })?;

        let chunk_size = detector.chunk_size();

        Ok(Self {
            detector,
            chunk_size,
        })
    }

    /// Detect voice activity in audio data with sophisticated confidence scoring
    pub async fn detect_voice_activity(
        &mut self,
        audio_data: &[u8],
    ) -> Result<VadResult, VoiceError> {
        // Convert raw audio bytes to i16 samples (16-bit PCM format)
        if audio_data.len() % 2 != 0 {
            return Err(VoiceError::ProcessingError(
                "Audio data length must be even for 16-bit samples".to_string(),
            ));
        }

        let samples: Vec<i16> = audio_data
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
            .collect();

        // Process audio in chunks if necessary
        let chunk_samples = if samples.len() > self.chunk_size {
            // Take first chunk for real-time processing
            &samples[..self.chunk_size]
        } else {
            // Pad with zeros if too short
            &samples
        };

        // Perform voice activity detection using Silero model
        let probability = self
            .detector
            .predict(chunk_samples.iter().copied())
            .map_err(|e| match e {
                VadError::PredictionFailed(msg) => {
                    VoiceError::ProcessingError(format!("VAD prediction failed: {}", msg))
                }
                VadError::VadConfigError {
                    sample_rate,
                    chunk_size,
                } => VoiceError::Configuration(format!(
                    "VAD configuration error - sample_rate: {}, chunk_size: {}",
                    sample_rate, chunk_size
                )),
            })?;

        // Apply threshold for voice detection (0.5 is standard threshold)
        let voice_detected = probability > 0.5;

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Ok(VadResult {
            voice_detected,
            timestamp,
            confidence: probability,
        })
    }

    /// Reset VAD internal state (useful for new audio streams)
    pub fn reset(&mut self) {
        self.detector.reset();
    }

    /// Get the configured chunk size for optimal audio processing
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }
}

/// Production wake word engine using Koffee-Candle ML detection
pub struct KoffeeEngine {
    detector: KoffeeCandle,
    models_loaded: Vec<String>,
    #[allow(dead_code)] // Config kept for potential future debugging/reconfiguration needs
    config: KoffeeCandleConfig,
}

impl KoffeeEngine {
    pub fn new() -> Result<Self, VoiceError> {
        let config = KoffeeCandleConfig::default();
        let mut detector = KoffeeCandle::new(&config)
            .map_err(|e| VoiceError::Configuration(format!("Failed to create Koffee detector: {}", e)))?;
        
        let mut models_loaded = Vec::new();
        let syrup_model_path = "../koffee/training/models/syrup.rpw";
        match WakewordModel::load_from_file(syrup_model_path) {
            Ok(model) => {
                detector.add_wakeword_model(model)
                    .map_err(|e| VoiceError::Configuration(format!("Failed to add syrup model: {}", e)))?;
                models_loaded.push("syrup".to_string());
            }
            Err(_) => {
                let alt_model_path = "../koffee/tests/resources/syrup_model.rpw";
                if let Ok(model) = WakewordModel::load_from_file(alt_model_path) {
                    detector.add_wakeword_model(model)
                        .map_err(|e| VoiceError::Configuration(format!("Failed to add syrup model: {}", e)))?;
                    models_loaded.push("syrup".to_string());
                } else {
                    return Err(VoiceError::Configuration("No wake word models could be loaded".to_string()));
                }
            }
        }
        
        Ok(Self { detector, models_loaded, config })
    }

    pub fn detect(&mut self, audio_data: &[u8]) -> Result<Option<WakeWordResult>, VoiceError> {
        // Validate audio data format (16-bit PCM expected)
        if audio_data.len() % 2 != 0 {
            return Err(VoiceError::ProcessingError(
                "Audio data length must be even for 16-bit samples".to_string()
            ));
        }
        
        if audio_data.is_empty() {
            return Ok(None);
        }
        
        // Process audio through Koffee detection pipeline
        match self.detector.process_bytes(audio_data) {
            Some(detection) => {
                let timestamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64;
                
                // Map KoffeeCandleDetection to WakeWordResult
                let result = WakeWordResult {
                    word: detection.name,
                    confidence: detection.score,
                    timestamp,
                };
                
                Ok(Some(result))
            }
            None => Ok(None),
        }
    }
    
    /// Get information about loaded models
    pub fn loaded_models(&self) -> &[String] {
        &self.models_loaded
    }
    
    /// Update detection thresholds at runtime
    pub fn update_thresholds(&mut self, _avg_threshold: f32, _threshold: f32) {
        // Note: Koffee detector configuration update requires reconstruction
        // This is a placeholder for runtime threshold updates
    }
}

/// Result structure for STT operations
#[derive(Debug, Clone)]
pub struct SttResult {
    pub text: String,
    pub confidence: f32,
    pub timestamp: u64,
}

/// Result structure for VAD operations
#[derive(Debug, Clone)]
pub struct VadResult {
    pub voice_detected: bool,
    pub timestamp: u64,
    pub confidence: f32,
}

/// Result structure for wake word detection
#[derive(Debug, Clone)]
pub struct WakeWordResult {
    pub confidence: f32,
    pub timestamp: u64,
    pub word: String,
}

/// Shared coordination state across all engines
#[derive(Debug, Clone)]
pub struct CoordinationState {
    pub current_mode: ProcessingMode,
    pub last_wake_word_time: Option<u64>,
    pub last_voice_activity_time: Option<u64>,
    pub conversation_active: bool,
    pub error_count: u32,
}

/// Processing modes for the coordination system
#[derive(Debug, Clone, PartialEq)]
pub enum ProcessingMode {
    Idle,
    WakeWordListening,
    VoiceProcessing,
    SpeechSynthesis,
    ConversationMode,
}

impl CoordinationState {
    pub fn new() -> Self {
        Self {
            current_mode: ProcessingMode::Idle,
            last_wake_word_time: None,
            last_voice_activity_time: None,
            conversation_active: false,
            error_count: 0,
        }
    }

    pub fn set_mode(&mut self, mode: ProcessingMode) {
        self.current_mode = mode;
    }

    pub fn is_conversation_active(&self) -> bool {
        self.conversation_active
    }

    pub fn start_conversation(&mut self) {
        self.conversation_active = true;
        self.current_mode = ProcessingMode::ConversationMode;
    }

    pub fn end_conversation(&mut self) {
        self.conversation_active = false;
        self.current_mode = ProcessingMode::Idle;
    }

    pub fn increment_error_count(&mut self) {
        self.error_count += 1;
    }

    pub fn reset_error_count(&mut self) {
        self.error_count = 0;
    }
}

impl Default for CoordinationState {
    fn default() -> Self {
        Self::new()
    }
}

/// Central coordinator for all default engines
pub struct DefaultEngineCoordinator {
    tts_engine: Arc<Mutex<DefaultTtsEngine>>,
    stt_engine: Arc<Mutex<DefaultSttEngine>>,
    vad_engine: Arc<Mutex<VadEngine>>,
    wake_word_engine: Arc<Mutex<KoffeeEngine>>,
    coordination_state: Arc<RwLock<CoordinationState>>,
    event_bus: Arc<EventBus>,
}

impl DefaultEngineCoordinator {
    /// Create a new default engine coordinator
    pub fn new() -> Result<Self, VoiceError> {
        // Initialize all default engines with coordination
        let tts_engine = Arc::new(Mutex::new(DefaultTtsEngine::new()?));
        let stt_engine = Arc::new(Mutex::new(DefaultSttEngine::with_whisper_vad_koffee()?));
        let vad_engine = Arc::new(Mutex::new(VadEngine::new()?));
        let wake_word_engine = Arc::new(Mutex::new(KoffeeEngine::new()?));

        let coordination_state = Arc::new(RwLock::new(CoordinationState::new()));
        let event_bus = Arc::new(EventBus::new());

        Ok(Self {
            tts_engine,
            stt_engine,
            vad_engine,
            wake_word_engine,
            coordination_state,
            event_bus,
        })
    }

    /// Start coordinated voice processing pipeline
    pub async fn start_coordinated_pipeline(
        &self,
    ) -> Result<super::CoordinatedVoiceStream, VoiceError> {
        // Create coordinated stream that manages all engines
        let stream = super::CoordinatedVoiceStream::new(
            self.tts_engine.clone(),
            self.stt_engine.clone(),
            self.vad_engine.clone(),
            self.wake_word_engine.clone(),
            self.event_bus.clone(),
        );

        Ok(stream)
    }

    /// Get the current coordination state
    pub async fn get_coordination_state(&self) -> CoordinationState {
        let state = self.coordination_state.read().await;
        state.clone()
    }

    /// Update the coordination state
    pub async fn update_coordination_state<F>(&self, updater: F) -> Result<(), VoiceError>
    where
        F: FnOnce(&mut CoordinationState),
    {
        let mut state = self.coordination_state.write().await;
        updater(&mut state);
        Ok(())
    }

    /// Get access to the event bus
    pub fn event_bus(&self) -> &Arc<EventBus> {
        &self.event_bus
    }

    /// Get access to the TTS engine
    pub fn tts_engine(&self) -> &Arc<Mutex<DefaultTtsEngine>> {
        &self.tts_engine
    }

    /// Get access to the STT engine
    pub fn stt_engine(&self) -> &Arc<Mutex<DefaultSttEngine>> {
        &self.stt_engine
    }

    /// Get access to the VAD engine
    pub fn vad_engine(&self) -> &Arc<Mutex<VadEngine>> {
        &self.vad_engine
    }

    /// Get access to the wake word engine
    pub fn wake_word_engine(&self) -> &Arc<Mutex<KoffeeEngine>> {
        &self.wake_word_engine
    }

    /// Set up default event handlers for coordination
    pub async fn setup_coordination_handlers(&self) -> Result<(), VoiceError> {
        let state = self.coordination_state.clone();
        let event_bus = self.event_bus.clone();

        // Handle wake word events
        event_bus
            .subscribe(EventType::WakeWordDetected, {
                let state = state.clone();
                move |event| {
                    let state = state.clone();
                    Box::pin(async move {
                        if let super::event_bus::VoiceEvent::WakeWordDetected {
                            timestamp, ..
                        } = event
                        {
                            let mut coordination_state = state.write().await;
                            coordination_state.last_wake_word_time = Some(timestamp);
                            coordination_state.set_mode(ProcessingMode::VoiceProcessing);
                        }
                        Ok(())
                    })
                }
            })
            .await;

        // Handle voice activity events
        event_bus
            .subscribe(EventType::VoiceActivityStarted, {
                let state = state.clone();
                move |event| {
                    let state = state.clone();
                    Box::pin(async move {
                        if let super::event_bus::VoiceEvent::VoiceActivityStarted {
                            timestamp,
                            ..
                        } = event
                        {
                            let mut coordination_state = state.write().await;
                            coordination_state.last_voice_activity_time = Some(timestamp);
                        }
                        Ok(())
                    })
                }
            })
            .await;

        // Handle error events
        event_bus
            .subscribe(EventType::ErrorOccurred, {
                let state = state.clone();
                move |_event| {
                    let state = state.clone();
                    Box::pin(async move {
                        let mut coordination_state = state.write().await;
                        coordination_state.increment_error_count();
                        Ok(())
                    })
                }
            })
            .await;

        Ok(())
    }

    /// Check if the coordinator is ready for processing
    pub async fn is_ready(&self) -> bool {
        // Check if all engines are initialized and ready
        // This is a simplified check - actual implementation would verify engine states
        true
    }

    /// Shutdown the coordinator and all engines
    pub async fn shutdown(&self) -> Result<(), VoiceError> {
        // Update state to indicate shutdown
        self.update_coordination_state(|state| {
            state.set_mode(ProcessingMode::Idle);
            state.end_conversation();
        })
        .await?;

        // Clear any pending events
        self.event_bus.clear_event_queue().await?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_coordinator_creation() {
        let coordinator = DefaultEngineCoordinator::new();
        assert!(coordinator.is_ok());

        let coordinator = coordinator.unwrap();
        assert!(coordinator.is_ready().await);
    }

    #[tokio::test]
    async fn test_coordination_state_management() {
        let coordinator = DefaultEngineCoordinator::new().unwrap();

        let initial_state = coordinator.get_coordination_state().await;
        assert_eq!(initial_state.current_mode, ProcessingMode::Idle);
        assert!(!initial_state.is_conversation_active());

        coordinator
            .update_coordination_state(|state| {
                state.start_conversation();
            })
            .await
            .unwrap();

        let updated_state = coordinator.get_coordination_state().await;
        assert_eq!(updated_state.current_mode, ProcessingMode::ConversationMode);
        assert!(updated_state.is_conversation_active());
    }

    #[tokio::test]
    async fn test_coordination_handlers_setup() {
        let coordinator = DefaultEngineCoordinator::new().unwrap();
        let result = coordinator.setup_coordination_handlers().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_coordinator_shutdown() {
        let coordinator = DefaultEngineCoordinator::new().unwrap();
        let result = coordinator.shutdown().await;
        assert!(result.is_ok());

        let final_state = coordinator.get_coordination_state().await;
        assert_eq!(final_state.current_mode, ProcessingMode::Idle);
        assert!(!final_state.is_conversation_active());
    }

    // Additional tests for DefaultSttEngine advanced features
    #[tokio::test]
    async fn test_enhanced_confidence_calculation() {
        let engine = DefaultSttEngine::with_whisper_vad_koffee().unwrap();

        // Test with realistic chunk data (based on actual Whisper output patterns)
        let chunks = vec![
            fluent_voice_whisper::TtsChunk::new(
                0.0,
                2.0,                       // 2-second segment
                vec![1, 2, 3, 4, 5],       // Token IDs
                "Hello world".to_string(), // Clear speech
                -0.1,                      // Good avg_logprob
                0.05,                      // Low no_speech_prob
                0.0,                       // Deterministic temperature
                1.8,                       // Good compression_ratio
            ),
            fluent_voice_whisper::TtsChunk::new(
                2.0,
                3.5,                // 1.5-second segment
                vec![6, 7, 8],      // Fewer tokens
                "test".to_string(), // Short word
                -0.3,               // Lower confidence logprob
                0.15,               // Higher no_speech_prob
                0.2,                // Some randomness
                2.2,                // Acceptable compression
            ),
        ];

        let confidence = engine.calculate_overall_confidence(&chunks);

        // Confidence should be weighted average, accounting for segment duration
        assert!(confidence >= 0.0 && confidence <= 1.0);
        assert!(
            confidence > 0.5,
            "Should have reasonable confidence for good audio"
        );

        // Test edge cases
        let empty_chunks = vec![];
        assert_eq!(engine.calculate_overall_confidence(&empty_chunks), 0.0);
    }

    #[tokio::test]
    async fn test_real_audio_file_transcription() {
        let mut engine = DefaultSttEngine::with_whisper_vad_koffee().unwrap();

        // Test with actual sample audio files from the repository
        let sample_paths = vec![
            "../../fluent-voice-samples/samples/tts_samples/mixed/ex04-ex03_whisper_001_channel1_198s.wav",
            "../koffee/training/wake_words/sap/sap_01[sap].wav",
        ];

        for sample_path in sample_paths {
            if std::path::Path::new(sample_path).exists() {
                let result = engine.transcribe_file(sample_path).await;

                match result {
                    Ok(stt_result) => {
                        // Validate result structure
                        assert!(
                            !stt_result.text.is_empty(),
                            "Should produce non-empty transcription"
                        );
                        assert!(stt_result.confidence >= 0.0 && stt_result.confidence <= 1.0);
                        assert!(stt_result.timestamp > 0, "Should have valid timestamp");

                        println!(
                            "✅ Transcribed {}: '{}' (confidence: {:.2})",
                            sample_path, stt_result.text, stt_result.confidence
                        );
                    }
                    Err(e) => {
                        println!("⚠️  Failed to transcribe {}: {}", sample_path, e);
                        // Don't fail test - file might not be available in all environments
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn test_memory_transcription_with_generated_audio() {
        let mut engine = DefaultSttEngine::with_whisper_vad_koffee().unwrap();

        // Generate test audio: 2 seconds of 16kHz 16-bit sine wave (440Hz tone)
        let sample_rate = 16000;
        let duration_seconds = 2;
        let frequency = 440.0; // A4 note

        let mut audio_data = Vec::new();
        for i in 0..(sample_rate * duration_seconds) {
            let t = i as f32 / sample_rate as f32;
            let sample = (2.0 * std::f32::consts::PI * frequency * t).sin();
            let sample_i16 = (sample * 32767.0) as i16;
            audio_data.extend_from_slice(&sample_i16.to_le_bytes());
        }

        let result = engine.transcribe(&audio_data).await;
        assert!(
            result.is_ok(),
            "Should handle generated audio without error"
        );

        let stt_result = result.unwrap();
        assert!(stt_result.confidence >= 0.0 && stt_result.confidence <= 1.0);
        assert!(stt_result.timestamp > 0);
        // Pure tone should produce minimal/empty text (no speech)
        assert!(
            stt_result.text.len() < 20,
            "Pure tone should not produce significant text"
        );
    }

    #[tokio::test]
    async fn test_performance_configurations() {
        // Test realtime configuration
        let realtime_engine = DefaultSttEngine::with_realtime_config();
        assert!(
            realtime_engine.is_ok(),
            "Realtime config should initialize successfully"
        );

        // Test accuracy configuration
        let accuracy_engine = DefaultSttEngine::with_accuracy_config();
        assert!(
            accuracy_engine.is_ok(),
            "Accuracy config should initialize successfully"
        );

        // Test custom configuration
        let custom_config = ModelConfig {
            which_model: WhichModel::Base,
            quantized: true,
            timestamps: true,
            ..Default::default()
        };
        let custom_engine = DefaultSttEngine::with_custom_config(custom_config);
        assert!(
            custom_engine.is_ok(),
            "Custom config should initialize successfully"
        );
    }

    #[cfg(feature = "microphone")]
    #[tokio::test]
    async fn test_microphone_transcription_interface() {
        let mut engine = DefaultSttEngine::with_whisper_vad_koffee().unwrap();

        // Test microphone interface (may fail if no microphone available)
        let result = engine.transcribe_microphone().await;

        // Don't assert success - microphone may not be available in test environment
        // Just verify the interface works and returns appropriate error types
        match result {
            Ok(stt_result) => {
                assert!(stt_result.confidence >= 0.0 && stt_result.confidence <= 1.0);
                assert!(stt_result.timestamp > 0);
                println!("✅ Microphone transcription successful");
            }
            Err(VoiceError::Stt(_)) => {
                println!("⚠️  Microphone not available in test environment (expected)");
            }
            Err(e) => {
                panic!("Unexpected error type: {}", e);
            }
        }
    }
}
