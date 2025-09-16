//! Error types and handling for room visualizer
//!
//! Comprehensive error handling following VoiceError pattern from packages/domain/src/voice_error.rs

use std::time::Instant;
use thiserror::Error;

/// Comprehensive error types for room visualizer operations
/// Based on VoiceError pattern from packages/domain/src/voice_error.rs
#[derive(Debug, Clone, Error)]
pub enum VisualizerError {
    #[error("connection: {0}")]
    Connection(String),

    #[error("configuration: {0}")]
    Configuration(String),

    #[error("livekit: {0}")]
    LiveKit(String),

    #[error("audio processing: {0}")]
    AudioProcessing(String),

    #[error("video rendering: {0}")]
    VideoRendering(String),

    #[error("participant management: {0}")]
    ParticipantManagement(String),
}

impl VisualizerError {
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            Self::Connection(_) => ErrorSeverity::Critical,
            Self::LiveKit(_) => ErrorSeverity::High,
            Self::Configuration(_) => ErrorSeverity::Medium,
            Self::AudioProcessing(_) => ErrorSeverity::Low,
            Self::VideoRendering(_) => ErrorSeverity::Low,
            Self::ParticipantManagement(_) => ErrorSeverity::Medium,
        }
    }

    /// Get appropriate color for UI display based on severity
    pub fn color(&self) -> egui::Color32 {
        self.severity().color()
    }

    /// Get icon for UI display based on error type
    pub fn icon(&self) -> &'static str {
        match self {
            Self::Connection(_) => "🔌",
            Self::Configuration(_) => "⚙️",
            Self::LiveKit(_) => "📡",
            Self::AudioProcessing(_) => "🔊",
            Self::VideoRendering(_) => "🎥",
            Self::ParticipantManagement(_) => "👥",
        }
    }
}

#[derive(Debug, Clone)]
pub enum ErrorSeverity {
    Critical, // Red, blocks functionality
    High,     // Orange, major features impacted
    Medium,   // Yellow, minor features impacted
    Low,      // Blue, cosmetic issues
}

impl ErrorSeverity {
    pub fn color(&self) -> egui::Color32 {
        match self {
            Self::Critical => egui::Color32::RED,
            Self::High => egui::Color32::from_rgb(255, 165, 0), // Orange
            Self::Medium => egui::Color32::YELLOW,
            Self::Low => egui::Color32::LIGHT_BLUE,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Critical => "Critical",
            Self::High => "High",
            Self::Medium => "Medium",
            Self::Low => "Low",
        }
    }
}

/// Error state for UI display
#[derive(Debug, Clone)]
pub struct ErrorState {
    pub error: VisualizerError,
    pub timestamp: Instant,
    pub dismissed: bool,
}

impl ErrorState {
    pub fn new(error: VisualizerError) -> Self {
        Self {
            error,
            timestamp: Instant::now(),
            dismissed: false,
        }
    }

    pub fn elapsed(&self) -> std::time::Duration {
        self.timestamp.elapsed()
    }

    pub fn dismiss(&mut self) {
        self.dismissed = true;
    }

    pub fn is_expired(&self, timeout: std::time::Duration) -> bool {
        self.dismissed || self.elapsed() > timeout
    }
}

/// Helper methods for common error scenarios
impl VisualizerError {
    pub fn missing_room_url() -> Self {
        Self::Configuration("Room URL is required".to_string())
    }

    pub fn missing_credentials() -> Self {
        Self::Configuration("API key and secret are required".to_string())
    }

    pub fn connection_failed(reason: impl Into<String>) -> Self {
        Self::Connection(format!("Failed to connect: {}", reason.into()))
    }

    pub fn audio_player_failed(participant: impl Into<String>) -> Self {
        Self::AudioProcessing(format!(
            "Failed to create audio player for {}",
            participant.into()
        ))
    }

    pub fn video_renderer_failed(participant: impl Into<String>) -> Self {
        Self::VideoRendering(format!(
            "Failed to create video renderer for {}",
            participant.into()
        ))
    }

    pub fn participant_cleanup_failed(participant: impl Into<String>) -> Self {
        Self::ParticipantManagement(format!(
            "Failed to cleanup participant {}",
            participant.into()
        ))
    }
}
