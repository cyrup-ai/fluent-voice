use egui_wgpu::RenderState;
use livekit::webrtc::prelude::{RtcAudioTrack, RtcVideoTrack};
use tokio::sync::broadcast;

use crate::speech_animator::SpeechAnimator;

// Re-export necessary types from livekit
pub use livekit::{ConnectionState, Room, RoomEvent, RoomOptions, SimulateScenario};

/// Production-ready TTS system using fluent-voice with Dia Voice integration
pub struct FluentVoiceTts {
    speech_animator: Option<SpeechAnimator>,
    current_voice_id: Option<String>,
    cancellation_tx: broadcast::Sender<()>,
    active_task: Option<tokio::task::JoinHandle<()>>,
}

impl FluentVoiceTts {
    pub fn new() -> Result<Self, TtsError> {
        let (cancellation_tx, _) = broadcast::channel(1);
        Ok(Self {
            speech_animator: None,
            current_voice_id: None,
            cancellation_tx,
            active_task: None,
        })
    }

    pub fn initialize_speech_animator(
        &mut self,
        rt_handle: &tokio::runtime::Handle,
        render_state: RenderState,
        audio_track: RtcAudioTrack,
        video_track: RtcVideoTrack,
    ) {
        self.speech_animator = Some(SpeechAnimator::new(
            rt_handle,
            render_state,
            audio_track,
            video_track,
        ));
    }

    /// Synthesize speech using the fluent-voice API with Dia Voice integration
    pub async fn speak(&mut self, text: String) -> Result<(), TtsError> {
        // Import dia voice components for real TTS synthesis
        use dia::voice::{Conversation, DiaSpeaker, DiaSpeakerBuilder, VoicePool};
        use std::sync::Arc;

        // Create speaker with current voice ID
        let speaker_id = self.current_voice_id.as_deref().unwrap_or("Default");
        
        // Create speaker with specified voice ID using proper builder pattern
        let speaker_builder = DiaSpeakerBuilder::new(speaker_id.to_string());
        let speaker = DiaSpeaker::from_builder(speaker_builder)
            .map_err(|e| TtsError::ConfigurationError(format!("Failed to create speaker: {}", e)))?;
        let pool = Arc::new(VoicePool::new().map_err(|e| TtsError::ConfigurationError(format!("Failed to create voice pool: {}", e)))?);

        // Cancel any existing synthesis
        if let Some(task) = self.active_task.take() {
            task.abort();
        }

        // Create conversation with the text and speaker
        let conversation = Conversation::new_sync(text.clone(), speaker, pool)
            .map_err(|e| TtsError::SynthesisError(format!("Failed to create conversation: {}", e)))?;

        // Create cancellation receiver for this synthesis
        let mut cancellation_rx = self.cancellation_tx.subscribe();
        let text_clone = text.clone();

        // Spawn cancellable synthesis task
        let task = tokio::spawn(async move {
            tokio::select! {
                result = conversation.play(|result| result) => {
                    match result {
                        Ok(player) => {
                            match player.play().await {
                                Ok(_) => {
                                    tracing::info!("Speech synthesis completed for text: {}", text_clone);
                                }
                                Err(e) => {
                                    tracing::error!("Audio playback failed: {}", e);
                                }
                            }
                        }
                        Err(e) => {
                            tracing::error!("TTS generation failed: {}", e);
                        }
                    }
                }
                _ = cancellation_rx.recv() => {
                    tracing::info!("Speech synthesis cancelled for text: {}", text_clone);
                }
            }
        });

        self.active_task = Some(task);
        Ok(())
    }

    pub async fn stop(&mut self) -> Result<(), TtsError> {
        // Cancel any active synthesis task
        if let Some(task) = self.active_task.take() {
            task.abort();
            let _ = task.await; // Wait for cleanup, ignore cancellation errors
        }

        // Send cancellation signal to any listening synthesis operations
        let _ = self.cancellation_tx.send(());

        tracing::info!("TTS stopped and all synthesis tasks cancelled");
        Ok(())
    }

    pub async fn set_voice(&mut self, voice_id: String) -> Result<(), TtsError> {
        // Stop any ongoing synthesis when changing voice
        self.stop().await?;
        
        self.current_voice_id = Some(voice_id.clone());
        tracing::info!("Voice set to: {}", voice_id);
        Ok(())
    }

    pub fn update_animation(&self) {
        if let Some(animator) = &self.speech_animator {
            animator.update_animation();
        }
    }

    pub fn render(&self, ui: &mut egui::Ui) {
        if let Some(animator) = &self.speech_animator {
            animator.render(ui);
        }
    }
}

#[derive(Debug)]
pub enum TtsError {
    SynthesisError(String),
    ConfigurationError(String),
}

impl std::fmt::Display for TtsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TtsError::SynthesisError(msg) => write!(f, "TTS synthesis error: {}", msg),
            TtsError::ConfigurationError(msg) => write!(f, "TTS configuration error: {}", msg),
        }
    }
}

impl std::error::Error for TtsError {}
