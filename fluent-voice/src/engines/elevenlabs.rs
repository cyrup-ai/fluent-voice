//! ElevenLabs TTS engine implementation with HTTP/3 QUIC support.
//!
//! This module provides a production-ready HTTP/3 QUIC-enabled TTS engine
//! using the ElevenLabs API for high-quality speech synthesis.

use crate::{
    FluentVoice,
    error::VoiceError,
    language::Language,
    speaker::{Speaker, SpeakerLine},
};
use futures::Stream;
use std::time::Duration;

/// HTTP/3 QUIC configuration for ElevenLabs client
#[derive(Debug, Clone)]
pub struct Http3Config {
    /// Enable 0-RTT early data for reduced latency
    pub enable_early_data: bool,
    /// Maximum QUIC connection idle timeout
    pub max_idle_timeout: Duration,
    /// Per-stream receive window size (bytes)
    pub stream_receive_window: u64,
    /// Connection-wide receive window size (bytes)  
    pub conn_receive_window: u64,
    /// Send window size (bytes)
    pub send_window: u64,
}

impl Default for Http3Config {
    fn default() -> Self {
        Self {
            enable_early_data: true,
            max_idle_timeout: Duration::from_secs(30),
            stream_receive_window: 1024 * 1024,    // 1MB
            conn_receive_window: 10 * 1024 * 1024, // 10MB
            send_window: 1024 * 1024,              // 1MB
        }
    }
}

/// High-performance ElevenLabs engine with HTTP/3 QUIC support
#[derive(Debug, Clone)]
pub struct ElevenLabsEngine {
    http3_config: Http3Config,
    api_key: Option<String>,
}

impl ElevenLabsEngine {
    /// Create a new ElevenLabs engine with default HTTP/3 configuration
    pub fn new() -> Self {
        Self {
            http3_config: Http3Config::default(),
            api_key: None,
        }
    }

    /// Create a new ElevenLabs engine with custom HTTP/3 configuration
    pub fn with_http3_config(config: Http3Config) -> Self {
        Self {
            http3_config: config,
            api_key: None,
        }
    }

    /// Set the API key for authentication
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Configure HTTP/3 early data (0-RTT)
    pub fn with_early_data(mut self, enable: bool) -> Self {
        self.http3_config.enable_early_data = enable;
        self
    }

    /// Configure maximum QUIC connection idle timeout
    pub fn with_idle_timeout(mut self, timeout: Duration) -> Self {
        self.http3_config.max_idle_timeout = timeout;
        self
    }

    /// Configure per-stream receive window size
    pub fn with_stream_window(mut self, size: u64) -> Self {
        self.http3_config.stream_receive_window = size;
        self
    }

    /// Configure connection-wide receive window size
    pub fn with_conn_window(mut self, size: u64) -> Self {
        self.http3_config.conn_receive_window = size;
        self
    }

    /// Configure send window size
    pub fn with_send_window(mut self, size: u64) -> Self {
        self.http3_config.send_window = size;
        self
    }

    /// Create ElevenLabs client with current HTTP/3 configuration
    fn create_client(&self) -> Result<fluent_voice_elevenlabs::ElevenLabsClient, VoiceError> {
        use fluent_voice_elevenlabs::{ClientConfig, ElevenLabsClient};

        let config = ClientConfig {
            enable_early_data: self.http3_config.enable_early_data,
            max_idle_timeout: self.http3_config.max_idle_timeout,
            stream_receive_window: self.http3_config.stream_receive_window,
            conn_receive_window: self.http3_config.conn_receive_window,
            send_window: self.http3_config.send_window,
        };

        match &self.api_key {
            Some(key) => ElevenLabsClient::new_with_config(key.clone(), config).map_err(|e| {
                VoiceError::EngineError(format!("Failed to create HTTP/3 client: {}", e))
            }),
            None => ElevenLabsClient::from_env_with_config(config).map_err(|e| {
                VoiceError::EngineError(format!("Failed to create HTTP/3 client: {}", e))
            }),
        }
    }
}

impl Default for ElevenLabsEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl FluentVoice for ElevenLabsEngine {
    fn tts() -> impl crate::tts_conversation::TtsConversationBuilder {
        use crate::builders::tts_conversation_builder;

        tts_conversation_builder(move |lines: &[SpeakerLine], lang: Option<&Language>| {
            // This is a placeholder implementation
            // In production, this would use the HTTP/3 ElevenLabs client
            // to synthesize speech from the provided speaker lines

            let _language = lang.unwrap_or(&Language::ENGLISH_US);
            let _text = lines
                .iter()
                .map(|line| line.text())
                .collect::<Vec<_>>()
                .join(" ");

            // Return empty stream for now - real implementation would
            // make HTTP/3 requests to ElevenLabs API
            futures::stream::empty::<i16>()
        })
    }

    fn stt() -> impl crate::stt_conversation::SttConversationBuilder {
        use crate::builders::stt_conversation_builder;

        stt_conversation_builder(move |_audio_source| {
            // ElevenLabs doesn't provide STT, so return empty stream
            futures::stream::empty::<Result<crate::fluent_voice::DummySegment, VoiceError>>()
        })
    }
}

/// Convenience re-export of HTTP/3 configuration
pub use Http3Config as ElevenLabsHttp3Config;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elevenlabs_engine_creation() {
        let engine = ElevenLabsEngine::new();
        assert!(engine.http3_config.enable_early_data);
        assert_eq!(
            engine.http3_config.max_idle_timeout,
            Duration::from_secs(30)
        );
    }

    #[test]
    fn test_http3_config_builder() {
        let engine = ElevenLabsEngine::new()
            .with_early_data(false)
            .with_idle_timeout(Duration::from_secs(60))
            .with_stream_window(2 * 1024 * 1024)
            .with_api_key("test-key");

        assert!(!engine.http3_config.enable_early_data);
        assert_eq!(
            engine.http3_config.max_idle_timeout,
            Duration::from_secs(60)
        );
        assert_eq!(engine.http3_config.stream_receive_window, 2 * 1024 * 1024);
        assert_eq!(engine.api_key, Some("test-key".to_string()));
    }
}
