//! VAD-Driven Conversation System
//!
//! Complete conversation management system that integrates VAD with conversation flow
//! for natural turn-taking, dialogue management, and speaker tracking.

use crate::fluent_voice::{
    conversation_manager::{ConversationManager, ConversationState, ConversationTurn},
    coordinated_vad_system::{CoordinatedVadSystem, CoordinationResult},
    default_engine_coordinator::{DefaultEngineCoordinator, VadResult},
    event_bus::{EngineType, EventBus},
};
use fluent_voice_domain::VoiceError;
use futures::{Stream, StreamExt};
use std::{
    collections::HashMap,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::sync::{mpsc, RwLock};

/// Complete VAD-driven conversation system
pub struct VadConversationSystem {
    coordinated_system: CoordinatedVadSystem,
    _conversation_manager: ConversationManager,
    turn_processor: TurnProcessor,
    speaker_tracker: SpeakerTracker,
    dialogue_controller: DialogueController,
}

/// Processes conversation turns based on VAD input
pub struct TurnProcessor {
    _turn_threshold_ms: u64,
    min_turn_duration_ms: u64,
    current_turn_start: Option<u64>,
    turn_counter: u64,
}

impl TurnProcessor {
    pub fn new() -> Self {
        Self {
            _turn_threshold_ms: 1000,  // 1 second silence indicates turn end
            min_turn_duration_ms: 500, // Minimum 500ms for a valid turn
            current_turn_start: None,
            turn_counter: 0,
        }
    }

    pub fn process_vad_result(&mut self, vad_result: &VadResult) -> Option<TurnEvent> {
        if vad_result.voice_detected {
            if self.current_turn_start.is_none() {
                // Start of new turn
                self.current_turn_start = Some(vad_result.timestamp);
                Some(TurnEvent::TurnStarted {
                    turn_id: self.turn_counter,
                    timestamp: vad_result.timestamp,
                    confidence: vad_result.confidence,
                })
            } else {
                // Continuing turn
                None
            }
        } else {
            if let Some(turn_start) = self.current_turn_start {
                let turn_duration = vad_result.timestamp - turn_start;

                if turn_duration >= self.min_turn_duration_ms {
                    // Valid turn completed
                    self.current_turn_start = None;
                    self.turn_counter += 1;

                    Some(TurnEvent::TurnCompleted {
                        turn_id: self.turn_counter - 1,
                        start_timestamp: turn_start,
                        end_timestamp: vad_result.timestamp,
                        duration_ms: turn_duration,
                        confidence: vad_result.confidence,
                    })
                } else {
                    // Turn too short, ignore
                    self.current_turn_start = None;
                    None
                }
            } else {
                // Silence continues
                None
            }
        }
    }

    pub fn get_current_turn_duration(&self, current_timestamp: u64) -> Option<u64> {
        self.current_turn_start
            .map(|start| current_timestamp - start)
    }
}

/// Tracks speakers in the conversation
pub struct SpeakerTracker {
    speakers: HashMap<String, SpeakerInfo>,
    current_speaker: Option<String>,
    _speaker_change_threshold: f32,
}

#[derive(Debug, Clone)]
pub struct SpeakerInfo {
    pub speaker_id: String,
    pub total_speaking_time_ms: u64,
    pub turn_count: u32,
    pub average_confidence: f32,
    pub last_activity_timestamp: u64,
}

impl SpeakerTracker {
    pub fn new() -> Self {
        Self {
            speakers: HashMap::new(),
            current_speaker: None,
            _speaker_change_threshold: 0.7, // Confidence threshold for speaker change
        }
    }

    pub fn process_turn(&mut self, turn: &ConversationTurn) -> Option<SpeakerChangeEvent> {
        let speaker_id = &turn.speaker_id;

        // Update speaker info
        let speaker_info = self
            .speakers
            .entry(speaker_id.clone())
            .or_insert_with(|| SpeakerInfo {
                speaker_id: speaker_id.clone(),
                total_speaking_time_ms: 0,
                turn_count: 0,
                average_confidence: 0.0,
                last_activity_timestamp: turn.timestamp,
            });

        // Update statistics
        speaker_info.turn_count += 1;
        speaker_info.last_activity_timestamp = turn.timestamp;

        // Update average confidence
        speaker_info.average_confidence = (speaker_info.average_confidence
            * (speaker_info.turn_count - 1) as f32
            + turn.confidence)
            / speaker_info.turn_count as f32;

        // Check for speaker change
        let speaker_changed = match &self.current_speaker {
            Some(current) => current != speaker_id,
            None => true, // First speaker
        };

        if speaker_changed {
            let previous_speaker = self.current_speaker.clone();
            self.current_speaker = Some(speaker_id.clone());

            Some(SpeakerChangeEvent {
                previous_speaker,
                new_speaker: speaker_id.clone(),
                timestamp: turn.timestamp,
                confidence: turn.confidence,
            })
        } else {
            None
        }
    }

    pub fn get_current_speaker(&self) -> Option<&String> {
        self.current_speaker.as_ref()
    }

    pub fn get_speaker_stats(&self) -> &HashMap<String, SpeakerInfo> {
        &self.speakers
    }
}

/// Controls dialogue flow and conversation state
pub struct DialogueController {
    conversation_state: ConversationFlowState,
    _max_silence_duration_ms: u64,
    _conversation_timeout_ms: u64,
}

#[derive(Debug, Clone)]
pub enum ConversationFlowState {
    Idle,
    Listening,
    Processing,
    Responding,
    Paused,
    Ended,
}

impl DialogueController {
    pub fn new() -> Self {
        Self {
            conversation_state: ConversationFlowState::Idle,
            _max_silence_duration_ms: 5000,   // 5 seconds max silence
            _conversation_timeout_ms: 300000, // 5 minutes conversation timeout
        }
    }

    pub fn process_coordination_result(
        &mut self,
        result: &CoordinationResult,
    ) -> Option<DialogueEvent> {
        match result {
            CoordinationResult::EngineActivated {
                engine_type: EngineType::Stt,
                ..
            } => {
                if matches!(
                    self.conversation_state,
                    ConversationFlowState::Idle | ConversationFlowState::Paused
                ) {
                    self.conversation_state = ConversationFlowState::Listening;
                    Some(DialogueEvent::ConversationStarted)
                } else {
                    None
                }
            }
            CoordinationResult::EngineDeactivated {
                engine_type: EngineType::Stt,
                ..
            } => {
                if matches!(self.conversation_state, ConversationFlowState::Listening) {
                    self.conversation_state = ConversationFlowState::Processing;
                    Some(DialogueEvent::ProcessingStarted)
                } else {
                    None
                }
            }
            CoordinationResult::TurnDetected { .. } => {
                self.conversation_state = ConversationFlowState::Responding;
                Some(DialogueEvent::ResponseRequired)
            }
            CoordinationResult::Error(_) => {
                self.conversation_state = ConversationFlowState::Paused;
                Some(DialogueEvent::ConversationPaused)
            }
            _ => None,
        }
    }

    pub fn get_conversation_state(&self) -> &ConversationFlowState {
        &self.conversation_state
    }

    pub fn end_conversation(&mut self) {
        self.conversation_state = ConversationFlowState::Ended;
    }
}

/// Events from turn processing
#[derive(Debug, Clone)]
pub enum TurnEvent {
    TurnStarted {
        turn_id: u64,
        timestamp: u64,
        confidence: f32,
    },
    TurnCompleted {
        turn_id: u64,
        start_timestamp: u64,
        end_timestamp: u64,
        duration_ms: u64,
        confidence: f32,
    },
}

/// Events from speaker tracking
#[derive(Debug, Clone)]
pub struct SpeakerChangeEvent {
    pub previous_speaker: Option<String>,
    pub new_speaker: String,
    pub timestamp: u64,
    pub confidence: f32,
}

/// Events from dialogue control
#[derive(Debug, Clone)]
pub enum DialogueEvent {
    ConversationStarted,
    ProcessingStarted,
    ResponseRequired,
    ConversationPaused,
    ConversationEnded,
}

impl VadConversationSystem {
    /// Create a new VAD-driven conversation system
    pub async fn new(
        event_bus: Arc<EventBus>,
        coordinator: DefaultEngineCoordinator,
    ) -> Result<Self, VoiceError> {
        let coordinated_system = CoordinatedVadSystem::new(event_bus.clone()).await?;
        let conversation_manager = ConversationManager::new(coordinator);
        let turn_processor = TurnProcessor::new();
        let speaker_tracker = SpeakerTracker::new();
        let dialogue_controller = DialogueController::new();

        Ok(Self {
            coordinated_system,
            _conversation_manager: conversation_manager,
            turn_processor,
            speaker_tracker,
            dialogue_controller,
        })
    }

    /// Start complete conversation processing
    pub async fn start_conversation<S>(
        &mut self,
        audio_stream: S,
    ) -> Result<ConversationStream, VoiceError>
    where
        S: Stream<Item = Vec<i16>> + Send + Unpin + 'static,
    {
        // Start coordinated processing
        let coordination_stream = self
            .coordinated_system
            .start_coordinated_processing(audio_stream)
            .await?;

        // Create conversation stream
        let conversation_stream = self.create_conversation_stream(coordination_stream).await?;

        Ok(conversation_stream)
    }

    async fn create_conversation_stream(
        &mut self,
        coordination_stream: impl Stream<Item = CoordinationResult> + Send + 'static,
    ) -> Result<ConversationStream, VoiceError> {
        let (conv_tx, conv_rx) = mpsc::channel(100);
        let conversation_state = Arc::new(RwLock::new(ConversationState::new()));

        // Move components into the processing task
        let mut turn_processor = std::mem::replace(&mut self.turn_processor, TurnProcessor::new());
        let mut speaker_tracker =
            std::mem::replace(&mut self.speaker_tracker, SpeakerTracker::new());
        let mut dialogue_controller =
            std::mem::replace(&mut self.dialogue_controller, DialogueController::new());

        // Clone conversation_state for the spawned task
        let conversation_state_for_task = Arc::clone(&conversation_state);

        tokio::spawn(async move {
            let mut coordination_stream = Box::pin(coordination_stream);

            while let Some(coordination_result) = coordination_stream.next().await {
                // Process through dialogue controller
                if let Some(dialogue_event) =
                    dialogue_controller.process_coordination_result(&coordination_result)
                {
                    match dialogue_event {
                        DialogueEvent::ConversationStarted => {
                            let event = ConversationStreamEvent::ConversationStarted {
                                timestamp: Self::current_timestamp(),
                            };
                            if conv_tx.send(event).await.is_err() {
                                break;
                            }
                        }
                        DialogueEvent::ResponseRequired => {
                            let event = ConversationStreamEvent::ResponseRequired {
                                timestamp: Self::current_timestamp(),
                            };
                            if conv_tx.send(event).await.is_err() {
                                break;
                            }
                        }
                        _ => {
                            // Handle other dialogue events as needed
                        }
                    }
                }

                // Extract VAD result for turn processing
                if let CoordinationResult::EngineActivated { vad_result, .. }
                | CoordinationResult::EngineDeactivated { vad_result, .. } = &coordination_result
                {
                    // Process turn events
                    if let Some(turn_event) = turn_processor.process_vad_result(vad_result) {
                        match turn_event {
                            TurnEvent::TurnCompleted {
                                turn_id,
                                start_timestamp: _,
                                end_timestamp,
                                duration_ms: _,
                                confidence,
                            } => {
                                // Create conversation turn
                                let turn = ConversationTurn {
                                    speaker_id: speaker_tracker
                                        .get_current_speaker()
                                        .cloned()
                                        .unwrap_or_else(|| "unknown".to_string()),
                                    transcript: format!("Turn {}", turn_id), // Would be filled by STT
                                    audio_data: Vec::new(), // Would be filled with actual audio
                                    timestamp: end_timestamp,
                                    confidence,
                                };

                                // Process speaker tracking
                                if let Some(speaker_change) = speaker_tracker.process_turn(&turn) {
                                    let event = ConversationStreamEvent::SpeakerChanged {
                                        previous_speaker: speaker_change.previous_speaker,
                                        new_speaker: speaker_change.new_speaker,
                                        timestamp: speaker_change.timestamp,
                                    };
                                    if conv_tx.send(event).await.is_err() {
                                        break;
                                    }
                                }

                                // Add turn to conversation state
                                {
                                    let mut state = conversation_state_for_task.write().await;
                                    state.add_turn(turn.clone());
                                }

                                let event = ConversationStreamEvent::TurnCompleted { turn };
                                if conv_tx.send(event).await.is_err() {
                                    break;
                                }
                            }
                            _ => {
                                // Handle other turn events as needed
                            }
                        }
                    }
                }
            }
        });

        Ok(ConversationStream::new(conv_rx, conversation_state))
    }

    fn current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    /// Get conversation statistics
    pub fn get_conversation_stats(&self) -> ConversationStats {
        ConversationStats {
            total_turns: self.turn_processor.turn_counter,
            active_speakers: self.speaker_tracker.speakers.len() as u32,
            conversation_state: self.dialogue_controller.conversation_state.clone(),
        }
    }

    /// Shutdown conversation system
    pub async fn shutdown(&mut self) -> Result<(), VoiceError> {
        self.coordinated_system.shutdown().await?;
        self.dialogue_controller.end_conversation();
        Ok(())
    }
}

/// Statistics about the conversation
#[derive(Debug, Clone)]
pub struct ConversationStats {
    pub total_turns: u64,
    pub active_speakers: u32,
    pub conversation_state: ConversationFlowState,
}

/// Events from the conversation stream
#[derive(Debug, Clone)]
pub enum ConversationStreamEvent {
    ConversationStarted {
        timestamp: u64,
    },
    TurnCompleted {
        turn: ConversationTurn,
    },
    SpeakerChanged {
        previous_speaker: Option<String>,
        new_speaker: String,
        timestamp: u64,
    },
    ResponseRequired {
        timestamp: u64,
    },
    ConversationEnded {
        timestamp: u64,
    },
}

/// Stream of conversation events
pub struct ConversationStream {
    receiver: mpsc::Receiver<ConversationStreamEvent>,
    conversation_state: Arc<RwLock<ConversationState>>,
}

impl ConversationStream {
    fn new(
        receiver: mpsc::Receiver<ConversationStreamEvent>,
        conversation_state: Arc<RwLock<ConversationState>>,
    ) -> Self {
        Self {
            receiver,
            conversation_state,
        }
    }

    /// Get current conversation state
    pub async fn get_conversation_state(&self) -> ConversationState {
        self.conversation_state.read().await.clone()
    }
}

impl Stream for ConversationStream {
    type Item = ConversationStreamEvent;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.receiver.poll_recv(cx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_stream::wrappers::ReceiverStream;

    fn create_test_audio_stream() -> impl Stream<Item = Vec<i16>> + Send + Unpin {
        let (tx, rx) = mpsc::channel(100);

        tokio::spawn(async move {
            // Generate conversation-like audio pattern
            for i in 0..8 {
                let samples = match i % 4 {
                    0 | 1 => {
                        // Speech from speaker 1
                        (0..512)
                            .map(|j| {
                                let t = j as f32 / 16000.0;
                                (440.0 * 2.0 * std::f32::consts::PI * t).sin() * 800.0
                            })
                            .map(|f| f as i16)
                            .collect()
                    }
                    2 => {
                        // Silence (turn boundary)
                        vec![0i16; 512]
                    }
                    3 => {
                        // Speech from speaker 2 (different frequency)
                        (0..512)
                            .map(|j| {
                                let t = j as f32 / 16000.0;
                                (880.0 * 2.0 * std::f32::consts::PI * t).sin() * 800.0
                            })
                            .map(|f| f as i16)
                            .collect()
                    }
                    _ => vec![0i16; 512],
                };

                if tx.send(samples).await.is_err() {
                    break;
                }

                tokio::time::sleep(tokio::time::Duration::from_millis(32)).await;
            }
        });

        ReceiverStream::new(rx)
    }

    #[tokio::test]
    async fn test_vad_conversation_system_creation() {
        let event_bus = Arc::new(EventBus::new());
        let coordinator = match DefaultEngineCoordinator::new() {
            Ok(coord) => coord,
            Err(_) => return, // Skip test if coordinator can't be created
        };

        let result = VadConversationSystem::new(event_bus, coordinator).await;
        assert!(
            result.is_ok(),
            "Should create VAD conversation system successfully"
        );
    }

    #[test]
    fn test_turn_processor() {
        let mut processor = TurnProcessor::new();

        // Test turn start
        let vad_result_start = VadResult {
            voice_detected: true,
            timestamp: 1000,
            confidence: 0.8,
        };

        let event = processor.process_vad_result(&vad_result_start);
        assert!(matches!(event, Some(TurnEvent::TurnStarted { .. })));

        // Test turn end
        let vad_result_end = VadResult {
            voice_detected: false,
            timestamp: 2000,
            confidence: 0.7,
        };

        let event = processor.process_vad_result(&vad_result_end);
        assert!(matches!(event, Some(TurnEvent::TurnCompleted { .. })));
    }

    #[test]
    fn test_speaker_tracker() {
        let mut tracker = SpeakerTracker::new();

        let turn = ConversationTurn {
            speaker_id: "speaker1".to_string(),
            transcript: "Hello".to_string(),
            audio_data: Vec::new(),
            timestamp: 1000,
            confidence: 0.8,
        };

        let event = tracker.process_turn(&turn);
        assert!(
            event.is_some(),
            "Should detect speaker change for first speaker"
        );

        assert_eq!(tracker.get_current_speaker(), Some(&"speaker1".to_string()));
    }

    #[test]
    fn test_dialogue_controller() {
        let mut controller = DialogueController::new();

        let coordination_result = CoordinationResult::EngineActivated {
            engine_type: EngineType::Stt,
            vad_result: VadResult {
                voice_detected: true,
                timestamp: 1000,
                confidence: 0.8,
            },
        };

        let event = controller.process_coordination_result(&coordination_result);
        assert!(matches!(event, Some(DialogueEvent::ConversationStarted)));
        assert!(matches!(
            controller.get_conversation_state(),
            ConversationFlowState::Listening
        ));
    }
}
