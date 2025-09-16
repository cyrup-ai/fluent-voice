use futures::Future;
use parking_lot::Mutex;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::mpsc;

use crate::audio_visualizer::AudioVisualizer;
use crate::speech_animator::SpeechAnimator;
use crate::video_renderer::VideoRenderer;

use egui_wgpu::RenderState;

// Re-export necessary types from livekit
pub use livekit::{ConnectionState, Room, RoomEvent, RoomOptions, SimulateScenario};

// Re-export WebRTC types from livekit
pub use livekit::webrtc::prelude::{RtcAudioTrack, RtcVideoTrack};

pub struct CykoTTS {
    operations: Arc<Mutex<Vec<Arc<dyn TtsOperation>>>>,
    runtime: tokio::runtime::Runtime,
    speech_animator: Option<SpeechAnimator>,
}

impl CykoTTS {
    pub fn new() -> Result<Self, TtsError> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|_| TtsError::RuntimeInitializationFailed)?;

        Ok(Self {
            operations: Arc::new(Mutex::new(Vec::new())),
            runtime,
            speech_animator: None,
        })
    }

    pub fn initialize_speech_animator(
        &mut self,
        render_state: RenderState,
        audio_track: RtcAudioTrack,
        video_track: RtcVideoTrack,
    ) {
        self.speech_animator = Some(SpeechAnimator::new(
            self.runtime.handle(),
            render_state,
            audio_track,
            video_track,
        ));
    }

    pub async fn speak(&self, text: String) -> Result<(), TtsError> {
        let (tx, mut rx) = mpsc::channel(1);
        let op = Arc::new(SpeakOperation {
            text,
            _tx: tx.clone(),
        });
        self.operations.lock().push(op.clone());

        self.runtime.spawn(async move {
            let result = op.perform().await;
            if let Err(e) = result {
                let _ = tx.send(Err(e)).await;
            } else {
                let _ = tx.send(Ok(())).await;
            }
        });

        match rx.recv().await {
            Some(result) => result,
            None => Err(TtsError::OperationFailed),
        }
    }

    pub async fn stop(&self) -> Result<(), TtsError> {
        let (tx, mut rx) = mpsc::channel(1);
        let op = Arc::new(StopOperation { _tx: tx.clone() });
        self.operations.lock().push(op.clone());

        self.runtime.spawn(async move {
            let result = op.perform().await;
            if let Err(e) = result {
                let _ = tx.send(Err(e)).await;
            } else {
                let _ = tx.send(Ok(())).await;
            }
        });

        match rx.recv().await {
            Some(result) => result,
            None => Err(TtsError::OperationFailed),
        }
    }

    pub async fn set_voice(&self, voice: String) -> Result<(), TtsError> {
        let (tx, mut rx) = mpsc::channel(1);
        let op = Arc::new(SetVoiceOperation {
            voice,
            _tx: tx.clone(),
        });
        self.operations.lock().push(op.clone());

        self.runtime.spawn(async move {
            let result = op.perform().await;
            if let Err(e) = result {
                let _ = tx.send(Err(e)).await;
            } else {
                let _ = tx.send(Ok(())).await;
            }
        });

        match rx.recv().await {
            Some(result) => result,
            None => Err(TtsError::OperationFailed),
        }
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

trait TtsOperation: Send + Sync + 'static {
    fn perform<'a>(&'a self) -> Pin<Box<dyn Future<Output = Result<(), TtsError>> + Send + 'a>>;
}

struct SpeakOperation {
    text: String,
    _tx: mpsc::Sender<Result<(), TtsError>>,
}

impl TtsOperation for SpeakOperation {
    fn perform<'a>(&'a self) -> Pin<Box<dyn Future<Output = Result<(), TtsError>> + Send + 'a>> {
        Box::pin(async move {
            // Implement TTS logic here
            println!("Speaking: {}", self.text);
            // No need to send the result here, it will be sent by the caller
            Ok(())
        })
    }
}

struct StopOperation {
    _tx: mpsc::Sender<Result<(), TtsError>>,
}

impl TtsOperation for StopOperation {
    fn perform<'a>(&'a self) -> Pin<Box<dyn Future<Output = Result<(), TtsError>> + Send + 'a>> {
        Box::pin(async move {
            // Implement stop logic here
            println!("Stopping TTS");
            // No need to send the result here, it will be sent by the caller
            Ok(())
        })
    }
}

struct SetVoiceOperation {
    voice: String,
    _tx: mpsc::Sender<Result<(), TtsError>>,
}

impl TtsOperation for SetVoiceOperation {
    fn perform<'a>(&'a self) -> Pin<Box<dyn Future<Output = Result<(), TtsError>> + Send + 'a>> {
        Box::pin(async move {
            // Implement voice setting logic here
            println!("Setting voice to: {}", self.voice);
            // No need to send the result here, it will be sent by the caller
            Ok(())
        })
    }
}

#[derive(Debug)]
pub enum TtsError {
    OperationFailed,
    RuntimeInitializationFailed,
    // Add other error variants as needed
}

// Public API
impl CykoTTS {
    pub fn speak_async(&self, text: String) -> impl Future<Output = Result<(), TtsError>> + Send {
        self.speak(text)
    }

    pub fn stop_async(&self) -> impl Future<Output = Result<(), TtsError>> + Send {
        self.stop()
    }

    pub fn set_voice_async(
        &self,
        voice: String,
    ) -> impl Future<Output = Result<(), TtsError>> + Send {
        self.set_voice(voice)
    }
}

// Example usage
pub async fn example_usage() -> Result<(), TtsError> {
    let tts = CykoTTS::new()?;

    // Note: You would need to initialize the speech animator with actual RenderState and tracks
    // tts.initialize_speech_animator(render_state, audio_track, video_track);

    tts.set_voice_async("en-US-female-1".to_string()).await?;
    tts.speak_async("Hello, world!".to_string()).await?;
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
    tts.stop_async().await?;

    // In your main rendering loop:
    // tts.update_animation();
    // let mut ui = egui::Ui::new(...);
    // tts.render(&mut ui);

    Ok(())
}
