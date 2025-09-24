use egui_wgpu::RenderState;
use livekit::webrtc::prelude::{RtcAudioTrack, RtcVideoTrack};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::broadcast;

use crate::speech_animator::SpeechAnimator;

// Re-export necessary types from livekit
pub use livekit::{ConnectionState, Room, RoomEvent, RoomOptions, SimulateScenario};

/// Production-ready TTS system using fluent-voice with Dia Voice integration
pub struct FluentVoiceTts {
    speech_animator: Option<SpeechAnimator>,
    current_voice_path: Option<PathBuf>,
    cancellation_tx: broadcast::Sender<()>,
    active_task: Option<tokio::task::JoinHandle<()>>,
    pool_initialized: bool,
}

impl FluentVoiceTts {
    pub fn new() -> Result<Self, TtsError> {
        let (cancellation_tx, _) = broadcast::channel(1);
        Ok(Self {
            speech_animator: None,
            current_voice_path: None,
            cancellation_tx,
            active_task: None,
            pool_initialized: false,
        })
    }

    /// Initialize the global voice pool for dia voice synthesis
    fn ensure_pool_initialized(&mut self) -> Result<(), TtsError> {
        if self.pool_initialized {
            return Ok(());
        }

        // Import dia voice pool initialization
        use candle_core::Device;
        use dia::voice::pool::init_global_pool;

        // Create cache directory for voice data
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(std::env::temp_dir)
            .join("fluent-voice-animator");

        // Create cache directory if it doesn't exist
        std::fs::create_dir_all(&cache_dir).map_err(|e| {
            TtsError::ConfigurationError(format!("Failed to create cache directory: {}", e))
        })?;

        // Determine best available device for voice synthesis
        let device = if candle_core::utils::cuda_is_available() {
            Device::new_cuda(0).unwrap_or(Device::Cpu)
        } else if candle_core::utils::metal_is_available() {
            Device::new_metal(0).unwrap_or(Device::Cpu)
        } else {
            Device::Cpu
        };

        // Initialize global pool
        init_global_pool(cache_dir, device).map_err(|e| {
            TtsError::ConfigurationError(format!("Failed to initialize voice pool: {}", e))
        })?;

        self.pool_initialized = true;
        tracing::info!("Dia voice pool initialized successfully");
        Ok(())
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
        // Ensure voice pool is initialized
        self.ensure_pool_initialized()?;

        // Cancel any existing synthesis
        if let Some(task) = self.active_task.take() {
            task.abort();
        }

        // Create cancellation receiver for this synthesis
        let _cancellation_rx = self.cancellation_tx.subscribe();
        let text_clone = text.clone();
        let voice_path = self.current_voice_path.clone();

        // Spawn synthesis task on blocking thread to avoid Send requirements
        let task = tokio::task::spawn_blocking(move || {
            // Use block_on to run the async synthesis in the blocking context
            let rt = tokio::runtime::Handle::current();
            rt.block_on(async move {
                match Self::synthesize_speech_internal(text_clone.clone(), voice_path).await {
                    Ok(player) => match player.play().await {
                        Ok(_) => {
                            tracing::info!("Speech synthesis completed for text: {}", text_clone);
                        }
                        Err(e) => {
                            tracing::error!("Audio playback failed: {}", e);
                        }
                    },
                    Err(e) => {
                        tracing::error!("TTS generation failed: {}", e);
                    }
                }
            })
        });

        self.active_task = Some(task);
        Ok(())
    }

    /// Internal synthesis method using proper dia fluent API
    async fn synthesize_speech_internal(
        text: String,
        voice_path: Option<PathBuf>,
    ) -> Result<dia::voice::VoicePlayer, TtsError> {
        use dia::voice::{Conversation, DiaSpeaker, VoicePool};

        if let Some(path) = voice_path {
            // Use voice cloning with specified audio file
            let player = DiaSpeaker::clone(path)
                .speak(text)
                .execute()
                .map_err(|e| TtsError::SynthesisError(format!("Voice synthesis failed: {}", e)))?;
            Ok(player)
        } else {
            // Use default speaker for synthesis
            let speaker = DiaSpeaker::default();
            let pool = Arc::new(VoicePool::new().map_err(|e| {
                TtsError::ConfigurationError(format!("Failed to create voice pool: {}", e))
            })?);

            let conversation = Conversation::new(text, speaker, pool).await.map_err(|e| {
                TtsError::SynthesisError(format!("Failed to create conversation: {}", e))
            })?;

            // Use the play method with proper error handling
            let result = conversation.play(|result| result).await;
            result.map_err(|e| TtsError::SynthesisError(format!("Speech generation failed: {}", e)))
        }
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

    pub async fn set_voice_from_file(&mut self, voice_path: PathBuf) -> Result<(), TtsError> {
        // Stop any ongoing synthesis when changing voice
        self.stop().await?;

        // Validate voice file exists
        if !voice_path.exists() {
            return Err(TtsError::ConfigurationError(format!(
                "Voice file does not exist: {}",
                voice_path.display()
            )));
        }

        self.current_voice_path = Some(voice_path.clone());
        tracing::info!("Voice set to file: {}", voice_path.display());
        Ok(())
    }

    pub async fn use_default_voice(&mut self) -> Result<(), TtsError> {
        // Stop any ongoing synthesis when changing voice
        self.stop().await?;

        self.current_voice_path = None;
        tracing::info!("Using default voice for TTS");
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
