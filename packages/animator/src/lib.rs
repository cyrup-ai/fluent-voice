#![recursion_limit = "256"]

#[cfg(all(
    not(feature = "microphone"),
    not(feature = "encodec"),
    not(feature = "mimi"),
    not(feature = "snac")
))]
compile_error!("At least one audio feature must be enabled: microphone, encodec, mimi, or snac");

pub mod app;
pub mod audio_visualizer;
pub mod audioio;
pub mod cfg;
pub mod display;
pub mod input;
pub mod livekit_audio_player;
pub mod music;
pub mod oscillator;
pub mod speech_animator;
pub mod tts;
pub mod video_renderer;
pub mod visualizer_config;
pub mod visualizer_error;

pub use audio_visualizer::AudioVisualizer;
pub use livekit_audio_player::LiveKitAudioPlayer;
pub use video_renderer::VideoRenderer;
pub use visualizer_config::{ConnectionQuality, RoomVisualizerConfig};
pub use visualizer_error::{ErrorSeverity, ErrorState, VisualizerError};
