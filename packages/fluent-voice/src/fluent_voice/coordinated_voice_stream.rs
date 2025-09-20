//! Coordinated voice processing pipeline that manages all default engines.

use super::default_engine_coordinator::{
    DefaultSttEngine, DefaultTtsEngine, KoffeeEngine, SttResult, VadEngine, VadResult,
    WakeWordResult,
};
use super::event_bus::{EventBus, VoiceEvent};
use fluent_voice_domain::{AudioChunk, VoiceError};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};

/// Pipeline processing results
#[derive(Debug, Clone)]
pub enum PipelineResult {
    /// Speech was successfully transcribed
    SpeechTranscribed(SttResult),
    /// No voice activity was detected in the audio
    NoVoiceDetected,
    /// Wake word was detected
    WakeWordDetected(WakeWordResult),
    /// Voice activity was detected but no transcription yet
    VoiceActivityDetected(VadResult),
    /// Error occurred during processing
    ProcessingError(String),
}

/// Current state of the pipeline processing
#[derive(Debug, Clone, PartialEq)]
pub enum PipelineState {
    /// Pipeline is idle and waiting for input
    Idle,
    /// Currently listening for wake words
    ListeningForWakeWord,
    /// Processing voice activity
    ProcessingVoiceActivity,
    /// Transcribing speech
    TranscribingSpeech,
    /// Synthesizing response
    SynthesizingResponse,
    /// Error state - pipeline needs reset
    Error,
}

impl Default for PipelineState {
    fn default() -> Self {
        PipelineState::Idle
    }
}

/// Coordinated stream that manages all default engines
pub struct CoordinatedVoiceStream {
    tts_engine: Arc<Mutex<DefaultTtsEngine>>,
    stt_engine: Arc<Mutex<DefaultSttEngine>>,
    vad_engine: Arc<Mutex<VadEngine>>,
    wake_word_engine: Arc<Mutex<KoffeeEngine>>,
    event_bus: Arc<EventBus>,
    pipeline_state: Arc<RwLock<PipelineState>>,
}

impl CoordinatedVoiceStream {
    /// Create a new coordinated voice stream
    pub fn new(
        tts_engine: Arc<Mutex<DefaultTtsEngine>>,
        stt_engine: Arc<Mutex<DefaultSttEngine>>,
        vad_engine: Arc<Mutex<VadEngine>>,
        wake_word_engine: Arc<Mutex<KoffeeEngine>>,
        event_bus: Arc<EventBus>,
    ) -> Self {
        Self {
            tts_engine,
            stt_engine,
            vad_engine,
            wake_word_engine,
            event_bus,
            pipeline_state: Arc::new(RwLock::new(PipelineState::Idle)),
        }
    }

    /// Process audio input through coordinated pipeline
    pub async fn process_audio_input(
        &self,
        audio_data: &[u8],
    ) -> Result<PipelineResult, VoiceError> {
        // Update pipeline state to processing
        {
            let mut state = self.pipeline_state.write().await;
            if *state == PipelineState::Idle {
                *state = PipelineState::ListeningForWakeWord;
            }
        }

        // 1. Wake word detection
        {
            let mut wake_word_engine = self.wake_word_engine.lock().await;
            if let Some(wake_word_result) = wake_word_engine.detect(audio_data)? {
                // Publish wake word detected event
                self.event_bus
                    .publish(VoiceEvent::WakeWordDetected {
                        confidence: wake_word_result.confidence,
                        timestamp: wake_word_result.timestamp,
                    })
                    .await?;

                // Update pipeline state
                {
                    let mut state = self.pipeline_state.write().await;
                    *state = PipelineState::ProcessingVoiceActivity;
                }

                return Ok(PipelineResult::WakeWordDetected(wake_word_result));
            }
        }

        // 2. Voice activity detection
        {
            let mut vad_engine = self.vad_engine.lock().await;
            let vad_result = vad_engine.detect_voice_activity(audio_data).await?;

            if vad_result.voice_detected {
                // Publish voice activity started event
                self.event_bus
                    .publish(VoiceEvent::VoiceActivityStarted {
                        timestamp: vad_result.timestamp,
                    })
                    .await?;

                // Update pipeline state
                {
                    let mut state = self.pipeline_state.write().await;
                    *state = PipelineState::TranscribingSpeech;
                }

                // 3. Speech-to-text when voice detected
                let mut stt_engine = self.stt_engine.lock().await;
                let stt_result = stt_engine.transcribe(audio_data).await?;

                // Publish speech transcribed event
                self.event_bus
                    .publish(VoiceEvent::SpeechTranscribed {
                        text: stt_result.text.clone(),
                        confidence: stt_result.confidence,
                        timestamp: stt_result.timestamp,
                    })
                    .await?;

                // Reset pipeline state to idle after successful transcription
                {
                    let mut state = self.pipeline_state.write().await;
                    *state = PipelineState::Idle;
                }

                return Ok(PipelineResult::SpeechTranscribed(stt_result));
            } else {
                // Publish voice activity ended event if no voice detected
                self.event_bus
                    .publish(VoiceEvent::VoiceActivityEnded {
                        timestamp: vad_result.timestamp,
                    })
                    .await?;

                return Ok(PipelineResult::VoiceActivityDetected(vad_result));
            }
        }
    }

    /// Generate coordinated TTS response
    pub async fn generate_coordinated_response(
        &self,
        text: &str,
        speaker_id: &str,
    ) -> Result<AudioChunk, VoiceError> {
        // Update pipeline state
        {
            let mut state = self.pipeline_state.write().await;
            *state = PipelineState::SynthesizingResponse;
        }

        // Publish synthesis start event
        self.event_bus
            .publish(VoiceEvent::SynthesisStarted {
                text: text.to_string(),
                speaker_id: speaker_id.to_string(),
            })
            .await?;

        // Generate TTS using default engine
        let audio_chunk = {
            let mut tts_engine = self.tts_engine.lock().await;
            tts_engine.synthesize(text, speaker_id).await?
        };

        // Publish synthesis completion event
        self.event_bus
            .publish(VoiceEvent::SynthesisCompleted {
                audio_data: audio_chunk.data().to_vec(),
                duration_ms: audio_chunk.duration_ms(),
            })
            .await?;

        // Reset pipeline state
        {
            let mut state = self.pipeline_state.write().await;
            *state = PipelineState::Idle;
        }

        Ok(audio_chunk)
    }

    /// Get the current pipeline state
    pub async fn get_pipeline_state(&self) -> PipelineState {
        let state = self.pipeline_state.read().await;
        state.clone()
    }

    /// Reset the pipeline state to idle
    pub async fn reset_pipeline(&self) -> Result<(), VoiceError> {
        let mut state = self.pipeline_state.write().await;
        *state = PipelineState::Idle;
        Ok(())
    }

    /// Check if the pipeline is currently processing
    pub async fn is_processing(&self) -> bool {
        let state = self.pipeline_state.read().await;
        !matches!(*state, PipelineState::Idle)
    }

    /// Set the pipeline to error state
    pub async fn set_error_state(&self, error_message: &str) -> Result<(), VoiceError> {
        {
            let mut state = self.pipeline_state.write().await;
            *state = PipelineState::Error;
        }

        // Publish error event
        self.event_bus
            .publish(VoiceEvent::ErrorOccurred {
                engine: super::event_bus::EngineType::Stt, // Default to STT for generic errors
                error: error_message.to_string(),
            })
            .await?;

        Ok(())
    }

    /// Process a batch of audio samples for continuous processing
    pub async fn process_audio_batch(
        &self,
        audio_samples: &[&[u8]],
    ) -> Result<Vec<PipelineResult>, VoiceError> {
        let mut results = Vec::new();

        for audio_data in audio_samples {
            match self.process_audio_input(audio_data).await {
                Ok(result) => results.push(result),
                Err(e) => {
                    // Set error state and break on first error
                    self.set_error_state(&e.to_string()).await?;
                    return Err(e);
                }
            }
        }

        Ok(results)
    }

    /// Get access to the event bus for external event handling
    pub fn event_bus(&self) -> &Arc<EventBus> {
        &self.event_bus
    }

    /// Get access to individual engines for direct control if needed
    pub fn tts_engine(&self) -> &Arc<Mutex<DefaultTtsEngine>> {
        &self.tts_engine
    }

    pub fn stt_engine(&self) -> &Arc<Mutex<DefaultSttEngine>> {
        &self.stt_engine
    }

    pub fn vad_engine(&self) -> &Arc<Mutex<VadEngine>> {
        &self.vad_engine
    }

    pub fn wake_word_engine(&self) -> &Arc<Mutex<KoffeeEngine>> {
        &self.wake_word_engine
    }

    /// Stop all processing and clean up resources
    pub async fn shutdown(&self) -> Result<(), VoiceError> {
        // Reset pipeline state
        self.reset_pipeline().await?;

        // Clear any pending events
        self.event_bus.clear_event_queue().await?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fluent_voice::default_engine_coordinator::DefaultEngineCoordinator;

    #[tokio::test]
    async fn test_coordinated_stream_creation() {
        let coordinator = DefaultEngineCoordinator::new().unwrap();
        let stream = coordinator.start_coordinated_pipeline().await;
        assert!(stream.is_ok());
    }

    #[tokio::test]
    async fn test_pipeline_state_management() {
        let coordinator = DefaultEngineCoordinator::new().unwrap();
        let stream = coordinator.start_coordinated_pipeline().await.unwrap();

        let initial_state = stream.get_pipeline_state().await;
        assert_eq!(initial_state, PipelineState::Idle);

        stream.reset_pipeline().await.unwrap();
        let reset_state = stream.get_pipeline_state().await;
        assert_eq!(reset_state, PipelineState::Idle);
    }

    #[tokio::test]
    async fn test_error_state_handling() {
        let coordinator = DefaultEngineCoordinator::new().unwrap();
        let stream = coordinator.start_coordinated_pipeline().await.unwrap();

        stream.set_error_state("Test error").await.unwrap();
        let error_state = stream.get_pipeline_state().await;
        assert_eq!(error_state, PipelineState::Error);
    }

    #[tokio::test]
    async fn test_processing_state_check() {
        let coordinator = DefaultEngineCoordinator::new().unwrap();
        let stream = coordinator.start_coordinated_pipeline().await.unwrap();

        assert!(!stream.is_processing().await);

        stream.set_error_state("Test").await.unwrap();
        assert!(stream.is_processing().await);

        stream.reset_pipeline().await.unwrap();
        assert!(!stream.is_processing().await);
    }

    #[tokio::test]
    async fn test_stream_shutdown() {
        let coordinator = DefaultEngineCoordinator::new().unwrap();
        let stream = coordinator.start_coordinated_pipeline().await.unwrap();

        let result = stream.shutdown().await;
        assert!(result.is_ok());

        let final_state = stream.get_pipeline_state().await;
        assert_eq!(final_state, PipelineState::Idle);
    }
}
