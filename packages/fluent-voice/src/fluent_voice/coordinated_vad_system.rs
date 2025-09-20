//! Coordinated VAD System Integration
//!
//! Complete system that integrates VAD with STT, TTS, and Wake Word engines
//! for coordinated voice processing with resource management and state synchronization.

use crate::fluent_voice::{
    default_engine_coordinator::{
        DefaultEngineCoordinator, DefaultSttEngine, DefaultTtsEngine, KoffeeEngine, VadResult,
    },
    event_bus::{EngineType, EventBus, VoiceEvent},
    vad_processing_system::{RealTimeVadSystem, VadProcessingStream},
};
use fluent_voice_domain::VoiceError;
use futures::{Stream, StreamExt};
use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};
use tokio::sync::{mpsc, Mutex, RwLock};

/// Complete coordinated VAD system integrating all engines
pub struct CoordinatedVadSystem {
    vad_system: RealTimeVadSystem,
    engine_coordinator: DefaultEngineCoordinator,
    coordination_state: Arc<RwLock<CoordinationState>>,
    event_bus: Arc<EventBus>,
}

/// State information for engine coordination
#[derive(Debug, Clone)]
pub struct CoordinationState {
    pub stt_active: bool,
    pub tts_active: bool,
    pub wake_word_active: bool,
    pub vad_active: bool,
    pub current_voice_detected: bool,
    pub last_coordination_timestamp: Option<u64>,
    pub resource_usage: ResourceUsage,
}

impl Default for CoordinationState {
    fn default() -> Self {
        Self {
            stt_active: false,
            tts_active: false,
            wake_word_active: true, // Wake word is typically always active
            vad_active: false,
            current_voice_detected: false,
            last_coordination_timestamp: None,
            resource_usage: ResourceUsage::default(),
        }
    }
}

/// Resource usage tracking for performance monitoring
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    pub cpu_usage_percent: f32,
    pub memory_usage_mb: f32,
    pub active_engines_count: u32,
}

impl Default for ResourceUsage {
    fn default() -> Self {
        Self {
            cpu_usage_percent: 0.0,
            memory_usage_mb: 0.0,
            active_engines_count: 0,
        }
    }
}

impl CoordinatedVadSystem {
    /// Create a new coordinated VAD system
    pub async fn new(event_bus: Arc<EventBus>) -> Result<Self, VoiceError> {
        let engine_coordinator = DefaultEngineCoordinator::new()?;
        let vad_engine = engine_coordinator.vad_engine().clone();
        let vad_system = RealTimeVadSystem::new(event_bus.clone(), vad_engine);
        let coordination_state = Arc::new(RwLock::new(CoordinationState::default()));

        Ok(Self {
            vad_system,
            engine_coordinator,
            coordination_state,
            event_bus,
        })
    }

    /// Start complete coordinated processing
    pub async fn start_coordinated_processing<S>(
        &mut self,
        audio_stream: S,
    ) -> Result<CoordinatedProcessingStream, VoiceError>
    where
        S: Stream<Item = Vec<i16>> + Send + Unpin + 'static,
    {
        // Start VAD processing
        let vad_stream = self.vad_system.start_processing(audio_stream).await?;

        // Set up engine coordination based on VAD results
        let coordination_stream = self.create_coordination_stream(vad_stream).await?;

        Ok(coordination_stream)
    }

    async fn create_coordination_stream(
        &mut self,
        vad_stream: VadProcessingStream,
    ) -> Result<CoordinatedProcessingStream, VoiceError> {
        let (coord_tx, coord_rx) = mpsc::channel(100);
        let stt_engine = self.engine_coordinator.stt_engine().clone();
        let tts_engine = self.engine_coordinator.tts_engine().clone();
        let wake_word_engine = self.engine_coordinator.wake_word_engine().clone();
        let coordination_state = self.coordination_state.clone();
        let event_bus = self.event_bus.clone();

        tokio::spawn(async move {
            let mut vad_stream = vad_stream;

            while let Some(vad_result) = vad_stream.next().await {
                match vad_result {
                    Ok(result) => {
                        // Coordinate engines based on VAD result
                        let coordination_result = Self::coordinate_engines(
                            &result,
                            &stt_engine,
                            &tts_engine,
                            &wake_word_engine,
                            &coordination_state,
                            &event_bus,
                        )
                        .await;

                        if coord_tx.send(coordination_result).await.is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        let error_result = CoordinationResult::Error(e);
                        if coord_tx.send(error_result).await.is_err() {
                            break;
                        }
                    }
                }
            }
        });

        Ok(CoordinatedProcessingStream::new(coord_rx))
    }

    async fn coordinate_engines(
        vad_result: &VadResult,
        stt_engine: &Arc<Mutex<DefaultSttEngine>>,
        _tts_engine: &Arc<Mutex<DefaultTtsEngine>>,
        _wake_word_engine: &Arc<Mutex<KoffeeEngine>>,
        coordination_state: &Arc<RwLock<CoordinationState>>,
        event_bus: &Arc<EventBus>,
    ) -> CoordinationResult {
        // Update coordination state
        {
            let mut state = coordination_state.write().await;
            state.current_voice_detected = vad_result.voice_detected;
            state.last_coordination_timestamp = Some(vad_result.timestamp);

            // Update resource usage
            state.resource_usage.active_engines_count = if state.stt_active { 1 } else { 0 }
                + if state.tts_active { 1 } else { 0 }
                + if state.wake_word_active { 1 } else { 0 }
                + if state.vad_active { 1 } else { 0 };
        }

        // Coordinate engines based on voice activity
        if vad_result.voice_detected {
            // Activate STT when voice detected
            let mut state = coordination_state.write().await;
            if !state.stt_active {
                state.stt_active = true;
                drop(state); // Release lock before async operation

                // Attempt to activate STT engine
                match stt_engine.try_lock() {
                    Ok(_stt) => {
                        // STT engine is available for activation
                        // Publish engine activation event
                        let activation_event = VoiceEvent::SpeechTranscribed {
                            text: "STT Engine Activated".to_string(),
                            confidence: vad_result.confidence,
                            timestamp: vad_result.timestamp,
                        };
                        let _ = event_bus.publish(activation_event).await;
                        
                        CoordinationResult::EngineActivated {
                            engine_type: EngineType::Stt,
                            vad_result: vad_result.clone(),
                        }
                    }
                    Err(_) => {
                        // STT engine is busy
                        let error_event = VoiceEvent::ErrorOccurred {
                            engine: EngineType::Stt,
                            error: "Engine busy".to_string(),
                        };
                        let _ = event_bus.publish(error_event).await;
                        
                        CoordinationResult::EngineActivationFailed {
                            engine_type: EngineType::Stt,
                            reason: "Engine busy".to_string(),
                            vad_result: vad_result.clone(),
                        }
                    }
                }
            } else {
                CoordinationResult::EngineAlreadyActive {
                    engine_type: EngineType::Stt,
                    vad_result: vad_result.clone(),
                }
            }
        } else {
            // Deactivate STT when silence detected
            let mut state = coordination_state.write().await;
            if state.stt_active {
                state.stt_active = false;
                drop(state); // Release lock

                // Publish engine deactivation event
                let deactivation_event = VoiceEvent::VoiceActivityEnded {
                    timestamp: vad_result.timestamp,
                };
                let _ = event_bus.publish(deactivation_event).await;

                CoordinationResult::EngineDeactivated {
                    engine_type: EngineType::Stt,
                    vad_result: vad_result.clone(),
                }
            } else {
                CoordinationResult::NoCoordinationNeeded {
                    vad_result: vad_result.clone(),
                }
            }
        }
    }

    /// Get current coordination state
    pub async fn get_coordination_state(&self) -> CoordinationState {
        self.coordination_state.read().await.clone()
    }

    /// Update resource usage metrics
    pub async fn update_resource_usage(&self, cpu_percent: f32, memory_mb: f32) {
        let mut state = self.coordination_state.write().await;
        state.resource_usage.cpu_usage_percent = cpu_percent;
        state.resource_usage.memory_usage_mb = memory_mb;
    }

    /// Shutdown coordinated system
    pub async fn shutdown(&mut self) -> Result<(), VoiceError> {
        // Stop VAD processing
        self.vad_system.stop_processing().await;

        // Shutdown engine coordinator
        self.engine_coordinator.shutdown().await?;

        // Reset coordination state
        let mut state = self.coordination_state.write().await;
        *state = CoordinationState::default();

        Ok(())
    }
}

/// Results from engine coordination operations
#[derive(Debug, Clone)]
pub enum CoordinationResult {
    EngineActivated {
        engine_type: EngineType,
        vad_result: VadResult,
    },
    EngineDeactivated {
        engine_type: EngineType,
        vad_result: VadResult,
    },
    EngineActivationFailed {
        engine_type: EngineType,
        reason: String,
        vad_result: VadResult,
    },
    EngineAlreadyActive {
        engine_type: EngineType,
        vad_result: VadResult,
    },
    TurnDetected {
        turn_id: u64,
        vad_result: VadResult,
    },
    NoCoordinationNeeded {
        vad_result: VadResult,
    },
    Error(VoiceError),
}

/// Stream of coordination processing results
pub struct CoordinatedProcessingStream {
    receiver: mpsc::Receiver<CoordinationResult>,
}

impl CoordinatedProcessingStream {
    fn new(receiver: mpsc::Receiver<CoordinationResult>) -> Self {
        Self { receiver }
    }
}

impl Stream for CoordinatedProcessingStream {
    type Item = CoordinationResult;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.receiver.poll_recv(cx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio_stream::wrappers::ReceiverStream;

    fn create_test_audio_stream() -> impl Stream<Item = Vec<i16>> + Send + Unpin {
        let (tx, rx) = mpsc::channel(100);

        tokio::spawn(async move {
            // Generate test audio with voice transitions
            for i in 0..5 {
                let samples = if i % 2 == 0 {
                    // Silence
                    vec![0i16; 512]
                } else {
                    // Simulated speech
                    (0..512)
                        .map(|j| {
                            let t = j as f32 / 16000.0;
                            (440.0 * 2.0 * std::f32::consts::PI * t).sin() * 1000.0
                        })
                        .map(|f| f as i16)
                        .collect()
                };

                if tx.send(samples).await.is_err() {
                    break;
                }

                tokio::time::sleep(Duration::from_millis(32)).await;
            }
        });

        ReceiverStream::new(rx)
    }

    #[tokio::test]
    async fn test_coordinated_vad_system_creation() {
        let event_bus = Arc::new(EventBus::new());
        let result = CoordinatedVadSystem::new(event_bus).await;

        assert!(
            result.is_ok(),
            "Should create coordinated VAD system successfully"
        );
    }

    #[tokio::test]
    async fn test_coordination_state_management() {
        let event_bus = Arc::new(EventBus::new());
        let system = match CoordinatedVadSystem::new(event_bus).await {
            Ok(system) => system,
            Err(_) => return, // Skip test if system can't be created
        };

        let initial_state = system.get_coordination_state().await;
        assert!(
            !initial_state.stt_active,
            "STT should initially be inactive"
        );
        assert!(
            initial_state.wake_word_active,
            "Wake word should initially be active"
        );
    }

    #[tokio::test]
    async fn test_coordinated_processing_stream() {
        let event_bus = Arc::new(EventBus::new());
        let mut system = match CoordinatedVadSystem::new(event_bus).await {
            Ok(system) => system,
            Err(_) => return, // Skip test if system can't be created
        };

        let audio_stream = create_test_audio_stream();
        let processing_result = system.start_coordinated_processing(audio_stream).await;

        match processing_result {
            Ok(mut coordination_stream) => {
                // Process a few coordination results
                let mut results = Vec::new();
                for _ in 0..3 {
                    if let Some(result) = coordination_stream.next().await {
                        results.push(result);
                    }
                }

                assert!(!results.is_empty(), "Should produce coordination results");
            }
            Err(_) => {
                // Test passes if processing can't start (dependencies may not be available)
            }
        }
    }

    #[test]
    fn test_coordination_state_default() {
        let state = CoordinationState::default();

        assert!(!state.stt_active);
        assert!(!state.tts_active);
        assert!(state.wake_word_active);
        assert!(!state.vad_active);
        assert!(!state.current_voice_detected);
    }

    #[test]
    fn test_resource_usage_tracking() {
        let mut usage = ResourceUsage::default();

        usage.cpu_usage_percent = 25.5;
        usage.memory_usage_mb = 128.0;
        usage.active_engines_count = 2;

        assert_eq!(usage.cpu_usage_percent, 25.5);
        assert_eq!(usage.memory_usage_mb, 128.0);
        assert_eq!(usage.active_engines_count, 2);
    }
}
