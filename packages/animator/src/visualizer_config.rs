//! Configuration management for room visualizer
//!
//! Provides configurable parameters for production-quality room visualization,
//! replacing hardcoded values with user-adjustable settings.

use std::time::Duration;

/// Configuration for room visualizer behavior
/// Based on AudioVisualizerConfig pattern from src/audio_visualizer.rs
#[derive(Debug, Clone)]
pub struct RoomVisualizerConfig {
    /// Threshold for speaking detection (0.0 = always speaking, 1.0 = never speaking)
    pub speaking_threshold: f32,
    /// Interval for connection quality updates
    pub connection_quality_update_interval: Duration,
    /// How long to display error messages in UI
    pub error_display_timeout: Duration,
    /// Auto-cleanup timeout for disconnected participants
    pub auto_cleanup_timeout: Duration,
    /// Smoothing factor for speaking detection (reduces flickering)
    pub speaking_smoothing: f32,
}

impl Default for RoomVisualizerConfig {
    fn default() -> Self {
        Self {
            speaking_threshold: 0.01, // Current hardcoded value
            connection_quality_update_interval: Duration::from_secs(5),
            error_display_timeout: Duration::from_secs(10),
            auto_cleanup_timeout: Duration::from_secs(300),
            speaking_smoothing: 0.2, // Same as AudioVisualizerConfig::smoothing_factor
        }
    }
}

impl RoomVisualizerConfig {
    /// Create configuration with custom speaking threshold
    pub fn with_speaking_threshold(mut self, threshold: f32) -> Self {
        self.speaking_threshold = threshold.clamp(0.001, 0.1);
        self
    }

    /// Create configuration with custom quality update interval
    pub fn with_quality_interval(mut self, interval: Duration) -> Self {
        self.connection_quality_update_interval = interval;
        self
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.speaking_threshold < 0.001 || self.speaking_threshold > 0.1 {
            return Err("Speaking threshold must be between 0.001 and 0.1");
        }

        if self.connection_quality_update_interval.as_secs() < 1 {
            return Err("Quality update interval must be at least 1 second");
        }

        if self.error_display_timeout.as_secs() < 3 {
            return Err("Error display timeout must be at least 3 seconds");
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum ConnectionQuality {
    Excellent, // >90% audio quality, consistent amplitude
    Good,      // >70% audio quality, stable connection
    Fair,      // >40% audio quality, some fluctuation
    Poor,      // <40% audio quality, unstable
    Unknown,   // No data available
}

impl ConnectionQuality {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Excellent => "Excellent",
            Self::Good => "Good",
            Self::Fair => "Fair",
            Self::Poor => "Poor",
            Self::Unknown => "Unknown",
        }
    }

    pub fn color(&self) -> egui::Color32 {
        match self {
            Self::Excellent => egui::Color32::from_rgb(0, 255, 0), // Green
            Self::Good => egui::Color32::from_rgb(144, 238, 144),  // Light Green
            Self::Fair => egui::Color32::from_rgb(255, 255, 0),    // Yellow
            Self::Poor => egui::Color32::from_rgb(255, 0, 0),      // Red
            Self::Unknown => egui::Color32::GRAY,                  // Gray
        }
    }

    /// Calculate connection quality based on audio statistics
    pub fn from_audio_stats(
        current_amplitude: f32,
        average_amplitude: f32,
        peak_amplitude: f32,
        is_muted: bool,
    ) -> Self {
        if is_muted {
            return Self::Good; // Muted is expected, not a quality issue
        }

        // Quality heuristics based on audio metrics
        let stability = if peak_amplitude > 0.0 {
            (average_amplitude / peak_amplitude).min(1.0)
        } else {
            0.0
        };

        match (current_amplitude, stability) {
            (a, s) if a > 0.1 && s > 0.8 => Self::Excellent,
            (a, s) if a > 0.05 && s > 0.6 => Self::Good,
            (a, s) if a > 0.01 && s > 0.4 => Self::Fair,
            _ => Self::Poor,
        }
    }
}
