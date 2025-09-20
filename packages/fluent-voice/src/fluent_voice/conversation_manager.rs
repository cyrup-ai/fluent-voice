//! Advanced conversation management with coordinated default engines.

use super::coordinated_voice_stream::CoordinatedVoiceStream;
use super::default_engine_coordinator::{DefaultEngineCoordinator, VadEngine};
use fluent_voice_domain::VoiceError;
use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Manages complex conversation flows with all default engines
pub struct ConversationManager {
    coordinator: DefaultEngineCoordinator,
    conversation_state: Arc<RwLock<ConversationState>>,
    turn_detection: TurnDetectionEngine,
}

/// State information for an active conversation
#[derive(Debug, Clone)]
pub struct ConversationState {
    pub current_speaker: Option<String>,
    pub conversation_history: Vec<ConversationTurn>,
    pub active_wake_words: HashSet<String>,
    pub vad_sensitivity: f32,
    pub turn_detection_enabled: bool,
}

/// Represents a single turn in a conversation
#[derive(Debug, Clone)]
pub struct ConversationTurn {
    pub speaker_id: String,
    pub transcript: String,
    pub audio_data: Vec<u8>,
    pub timestamp: u64,
    pub confidence: f32,
}

/// Engine for detecting conversation turns and speaker changes
#[derive(Debug, Clone)]
pub struct TurnDetectionEngine {
    sensitivity: f32,
    minimum_pause_duration_ms: u64,
    #[allow(dead_code)]
    speaker_change_threshold: f32,
    last_voice_activity_timestamp: Option<u64>,
}

/// Configuration for turn detection behavior
#[derive(Debug, Clone)]
pub struct TurnDetectionConfig {
    pub sensitivity: f32,
    pub minimum_pause_duration_ms: u64,
    pub speaker_change_threshold: f32,
    pub enable_speaker_identification: bool,
}

/// Stream that handles conversation-level processing
pub struct ConversationStream {
    voice_stream: CoordinatedVoiceStream,
    conversation_state: Arc<RwLock<ConversationState>>,
    turn_detection: TurnDetectionEngine,
}

/// Results from conversation processing
#[derive(Debug, Clone)]
pub enum ConversationResult {
    /// A conversation turn was completed
    TurnCompleted(ConversationTurn),
    /// Speaker change was detected
    SpeakerChanged {
        previous_speaker: Option<String>,
        new_speaker: String,
    },
    /// Conversation was paused due to silence
    ConversationPaused { last_activity_timestamp: u64 },
    /// Conversation was resumed after pause
    ConversationResumed { resume_timestamp: u64 },
    /// Error occurred during conversation processing
    ConversationError { error: String },
}

impl Default for TurnDetectionConfig {
    fn default() -> Self {
        Self {
            sensitivity: 0.5,
            minimum_pause_duration_ms: 1000, // 1 second
            speaker_change_threshold: 0.7,
            enable_speaker_identification: false,
        }
    }
}

impl TurnDetectionEngine {
    /// Create a new turn detection engine with default configuration
    pub fn new() -> Self {
        let config = TurnDetectionConfig::default();
        Self::with_config(config)
    }

    /// Create a new turn detection engine with custom configuration
    pub fn with_config(config: TurnDetectionConfig) -> Self {
        Self {
            sensitivity: config.sensitivity,
            minimum_pause_duration_ms: config.minimum_pause_duration_ms,
            speaker_change_threshold: config.speaker_change_threshold,
            last_voice_activity_timestamp: None,
        }
    }

    /// Configure the turn detection engine to work with VAD
    pub async fn configure_with_vad(
        &mut self,
        _vad_engine: &Arc<tokio::sync::Mutex<VadEngine>>,
    ) -> Result<(), VoiceError> {
        // Configuration logic for integrating with VAD engine
        // This would set up callbacks and sensitivity based on VAD performance
        Ok(())
    }

    /// Detect if a conversation turn has occurred
    pub fn detect_turn(&mut self, current_timestamp: u64, voice_detected: bool) -> bool {
        if voice_detected {
            self.last_voice_activity_timestamp = Some(current_timestamp);
            false // No turn during active speech
        } else {
            // Check if enough silence has passed to indicate a turn
            if let Some(last_activity) = self.last_voice_activity_timestamp {
                let silence_duration = current_timestamp - last_activity;
                silence_duration >= self.minimum_pause_duration_ms
            } else {
                false
            }
        }
    }

    /// Update the sensitivity of turn detection
    pub fn set_sensitivity(&mut self, sensitivity: f32) {
        self.sensitivity = sensitivity.clamp(0.0, 1.0);
    }

    /// Get the current sensitivity setting
    pub fn get_sensitivity(&self) -> f32 {
        self.sensitivity
    }

    /// Set the minimum pause duration for turn detection
    pub fn set_minimum_pause_duration(&mut self, duration_ms: u64) {
        self.minimum_pause_duration_ms = duration_ms;
    }

    /// Get the current minimum pause duration
    pub fn get_minimum_pause_duration(&self) -> u64 {
        self.minimum_pause_duration_ms
    }
}

impl Default for TurnDetectionEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ConversationState {
    /// Create a new conversation state
    pub fn new() -> Self {
        Self {
            current_speaker: None,
            conversation_history: Vec::new(),
            active_wake_words: HashSet::new(),
            vad_sensitivity: 0.5,
            turn_detection_enabled: true,
        }
    }

    /// Add a new turn to the conversation history
    pub fn add_turn(&mut self, turn: ConversationTurn) {
        self.current_speaker = Some(turn.speaker_id.clone());
        self.conversation_history.push(turn);
    }

    /// Get the last N turns from the conversation history
    pub fn get_recent_turns(&self, count: usize) -> Vec<ConversationTurn> {
        let start_index = if self.conversation_history.len() > count {
            self.conversation_history.len() - count
        } else {
            0
        };
        self.conversation_history[start_index..].to_vec()
    }

    /// Clear the conversation history
    pub fn clear_history(&mut self) {
        self.conversation_history.clear();
        self.current_speaker = None;
    }

    /// Get the total number of turns in the conversation
    pub fn turn_count(&self) -> usize {
        self.conversation_history.len()
    }

    /// Check if a specific wake word is active
    pub fn is_wake_word_active(&self, wake_word: &str) -> bool {
        self.active_wake_words.contains(wake_word)
    }

    /// Add a wake word to the active set
    pub fn add_wake_word(&mut self, wake_word: String) {
        self.active_wake_words.insert(wake_word);
    }

    /// Remove a wake word from the active set
    pub fn remove_wake_word(&mut self, wake_word: &str) {
        self.active_wake_words.remove(wake_word);
    }

    /// Set the VAD sensitivity for the conversation
    pub fn set_vad_sensitivity(&mut self, sensitivity: f32) {
        self.vad_sensitivity = sensitivity.clamp(0.0, 1.0);
    }

    /// Enable or disable turn detection
    pub fn set_turn_detection_enabled(&mut self, enabled: bool) {
        self.turn_detection_enabled = enabled;
    }
}

impl Default for ConversationState {
    fn default() -> Self {
        Self::new()
    }
}

impl ConversationManager {
    /// Create a new conversation manager
    pub fn new(coordinator: DefaultEngineCoordinator) -> Self {
        Self {
            coordinator,
            conversation_state: Arc::new(RwLock::new(ConversationState::new())),
            turn_detection: TurnDetectionEngine::new(),
        }
    }

    /// Create a conversation manager with custom turn detection configuration
    pub fn with_turn_detection_config(
        coordinator: DefaultEngineCoordinator,
        turn_config: TurnDetectionConfig,
    ) -> Self {
        Self {
            coordinator,
            conversation_state: Arc::new(RwLock::new(ConversationState::new())),
            turn_detection: TurnDetectionEngine::with_config(turn_config),
        }
    }

    /// Start a new conversation
    pub async fn start_conversation(&mut self) -> Result<ConversationStream, VoiceError> {
        // Start coordinated pipeline
        let voice_stream = self.coordinator.start_coordinated_pipeline().await?;

        // Set up turn detection
        self.turn_detection
            .configure_with_vad(self.coordinator.vad_engine())
            .await?;

        // Clear any previous conversation state
        {
            let mut state = self.conversation_state.write().await;
            state.clear_history();
        }

        // Create conversation stream with all engines coordinated
        let conversation_stream = ConversationStream::new(
            voice_stream,
            self.conversation_state.clone(),
            self.turn_detection.clone(),
        );

        Ok(conversation_stream)
    }

    /// Get the current conversation state
    pub async fn get_conversation_state(&self) -> ConversationState {
        let state = self.conversation_state.read().await;
        state.clone()
    }

    /// Update the conversation state
    pub async fn update_conversation_state<F>(&self, updater: F) -> Result<(), VoiceError>
    where
        F: FnOnce(&mut ConversationState),
    {
        let mut state = self.conversation_state.write().await;
        updater(&mut state);
        Ok(())
    }

    /// Configure turn detection settings
    pub fn configure_turn_detection(&mut self, config: TurnDetectionConfig) {
        self.turn_detection = TurnDetectionEngine::with_config(config);
    }

    /// Get access to the underlying coordinator
    pub fn coordinator(&self) -> &DefaultEngineCoordinator {
        &self.coordinator
    }

    /// End the current conversation and clean up resources
    pub async fn end_conversation(&self) -> Result<(), VoiceError> {
        let mut state = self.conversation_state.write().await;
        state.clear_history();
        Ok(())
    }
}

impl ConversationStream {
    /// Create a new conversation stream
    pub fn new(
        voice_stream: CoordinatedVoiceStream,
        conversation_state: Arc<RwLock<ConversationState>>,
        turn_detection: TurnDetectionEngine,
    ) -> Self {
        Self {
            voice_stream,
            conversation_state,
            turn_detection,
        }
    }

    /// Process audio input through the conversation pipeline
    pub async fn process_conversation_audio(
        &mut self,
        audio_data: &[u8],
        current_timestamp: u64,
    ) -> Result<ConversationResult, VoiceError> {
        // Process through the coordinated voice stream first
        let pipeline_result = self.voice_stream.process_audio_input(audio_data).await?;

        match pipeline_result {
            super::coordinated_voice_stream::PipelineResult::SpeechTranscribed(stt_result) => {
                // Create a conversation turn
                let turn = ConversationTurn {
                    speaker_id: "unknown_speaker".to_string(), // Would be determined by speaker identification
                    transcript: stt_result.text,
                    audio_data: audio_data.to_vec(),
                    timestamp: stt_result.timestamp,
                    confidence: stt_result.confidence,
                };

                // Add turn to conversation state
                {
                    let mut state = self.conversation_state.write().await;
                    state.add_turn(turn.clone());
                }

                Ok(ConversationResult::TurnCompleted(turn))
            }
            super::coordinated_voice_stream::PipelineResult::VoiceActivityDetected(vad_result) => {
                // Check for turn detection
                let turn_detected = self
                    .turn_detection
                    .detect_turn(current_timestamp, vad_result.voice_detected);

                if turn_detected {
                    Ok(ConversationResult::ConversationPaused {
                        last_activity_timestamp: vad_result.timestamp,
                    })
                } else if vad_result.voice_detected {
                    Ok(ConversationResult::ConversationResumed {
                        resume_timestamp: vad_result.timestamp,
                    })
                } else {
                    Ok(ConversationResult::ConversationPaused {
                        last_activity_timestamp: vad_result.timestamp,
                    })
                }
            }
            super::coordinated_voice_stream::PipelineResult::WakeWordDetected(_) => {
                Ok(ConversationResult::ConversationResumed {
                    resume_timestamp: current_timestamp,
                })
            }
            super::coordinated_voice_stream::PipelineResult::NoVoiceDetected => {
                Ok(ConversationResult::ConversationPaused {
                    last_activity_timestamp: current_timestamp,
                })
            }
            super::coordinated_voice_stream::PipelineResult::ProcessingError(error) => {
                Ok(ConversationResult::ConversationError { error })
            }
        }
    }

    /// Generate a response in the conversation
    pub async fn generate_conversation_response(
        &self,
        text: &str,
        speaker_id: &str,
    ) -> Result<fluent_voice_domain::AudioChunk, VoiceError> {
        self.voice_stream
            .generate_coordinated_response(text, speaker_id)
            .await
    }

    /// Get access to the underlying voice stream
    pub fn voice_stream(&self) -> &CoordinatedVoiceStream {
        &self.voice_stream
    }

    /// Get the current conversation state
    pub async fn get_conversation_state(&self) -> ConversationState {
        let state = self.conversation_state.read().await;
        state.clone()
    }

    /// End the conversation stream
    pub async fn end_conversation(&self) -> Result<(), VoiceError> {
        self.voice_stream.shutdown().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_conversation_manager_creation() {
        let coordinator = DefaultEngineCoordinator::new().unwrap();
        let manager = ConversationManager::new(coordinator);

        let state = manager.get_conversation_state().await;
        assert_eq!(state.turn_count(), 0);
        assert!(state.current_speaker.is_none());
    }

    #[tokio::test]
    async fn test_conversation_state_management() {
        let coordinator = DefaultEngineCoordinator::new().unwrap();
        let manager = ConversationManager::new(coordinator);

        manager
            .update_conversation_state(|state| {
                let turn = ConversationTurn {
                    speaker_id: "test_speaker".to_string(),
                    transcript: "Hello world".to_string(),
                    audio_data: vec![1, 2, 3],
                    timestamp: 12345,
                    confidence: 0.95,
                };
                state.add_turn(turn);
            })
            .await
            .unwrap();

        let state = manager.get_conversation_state().await;
        assert_eq!(state.turn_count(), 1);
        assert_eq!(state.current_speaker, Some("test_speaker".to_string()));
    }

    #[tokio::test]
    async fn test_turn_detection_engine() {
        let mut turn_engine = TurnDetectionEngine::new();

        // Test initial state
        assert_eq!(turn_engine.get_sensitivity(), 0.5);
        assert_eq!(turn_engine.get_minimum_pause_duration(), 1000);

        // Test sensitivity adjustment
        turn_engine.set_sensitivity(0.8);
        assert_eq!(turn_engine.get_sensitivity(), 0.8);

        // Test pause duration adjustment
        turn_engine.set_minimum_pause_duration(2000);
        assert_eq!(turn_engine.get_minimum_pause_duration(), 2000);
    }

    #[tokio::test]
    async fn test_conversation_turn_detection() {
        let mut turn_engine = TurnDetectionEngine::new();

        // Simulate voice activity
        assert!(!turn_engine.detect_turn(1000, true)); // Voice active, no turn
        assert!(!turn_engine.detect_turn(1500, false)); // Short silence, no turn
        assert!(turn_engine.detect_turn(2500, false)); // Long silence, turn detected
    }

    #[tokio::test]
    async fn test_conversation_stream_creation() {
        let coordinator = DefaultEngineCoordinator::new().unwrap();
        let mut manager = ConversationManager::new(coordinator);

        let stream_result = manager.start_conversation().await;
        assert!(stream_result.is_ok());
    }
}
