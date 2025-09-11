//! STT conversation domain objects.

use crate::{transcription::TranscriptionStream, voice_error::VoiceError};
use core::future::Future;

/// Engine-specific STT session object.
///
/// This trait represents a configured speech-to-text session that is
/// ready to produce a transcript stream. Engine implementations provide
/// concrete types that implement this trait.
pub trait SttConversation: Send {
    /// The transcript stream type that will be produced.
    type Stream: TranscriptionStream;

    /// Convert this session into a transcript stream.
    ///
    /// This method consumes the session and returns the underlying
    /// transcript stream that yields recognition results.
    fn into_stream(self) -> Self::Stream;

    /// Collect all transcript segments into a complete transcript.
    ///
    /// This method is used when you want to process the entire
    /// transcript at once rather than streaming.
    fn collect(self) -> impl Future<Output = Result<String, VoiceError>> + Send
    where
        Self: Sized,
    {
        async move {
            use crate::transcription::TranscriptionSegment;
            use futures::StreamExt;

            let mut stream = self.into_stream();
            let mut text = String::new();

            while let Some(result) = stream.next().await {
                match result {
                    Ok(segment) => {
                        if !text.is_empty() {
                            text.push(' ');
                        }
                        text.push_str(segment.text());
                    }
                    Err(e) => return Err(e),
                }
            }

            Ok(text)
        }
    }
}

/// Default STT conversation implementation.
///
/// This is a domain object that holds the result of a configured STT session.
#[derive(Debug, Clone)]
pub struct SttConversationImpl {
    /// The transcript text result.
    pub text: String,
    /// Session configuration that was used.
    pub config: SttConfig,
}

impl SttConversationImpl {
    /// Create a new STT conversation with the given text and config.
    pub fn new(text: String, config: SttConfig) -> Self {
        Self { text, config }
    }
}

/// STT configuration value object.
///
/// This holds all the configuration parameters for an STT session.
#[derive(Debug, Clone)]
pub struct SttConfig {
    /// Language hint for recognition.
    pub language: Option<String>,
    /// VAD mode setting.
    pub vad_mode: Option<String>,
    /// Noise reduction level.
    pub noise_reduction: Option<String>,
    /// Whether diarization is enabled.
    pub diarization_enabled: bool,
    /// Whether word timestamps are enabled.
    pub word_timestamps_enabled: bool,
    /// Whether punctuation is enabled.
    pub punctuation_enabled: bool,
}

impl Default for SttConfig {
    fn default() -> Self {
        Self {
            language: None,
            vad_mode: None,
            noise_reduction: None,
            diarization_enabled: false,
            word_timestamps_enabled: false,
            punctuation_enabled: true,
        }
    }
}
