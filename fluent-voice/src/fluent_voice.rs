//! Unified entry point trait for TTS and STT operations.

use crate::{
    stt_conversation::SttConversationBuilder, transcript::TranscriptSegment,
    tts_conversation::TtsConversationBuilder,
};

/// Unified entry point for Text-to-Speech and Speech-to-Text operations.
///
/// This trait provides the main entry points for the fluent voice API, allowing
/// users to start TTS or STT operations with a consistent interface.
///
/// # Examples
///
/// ```ignore
/// use fluent_voice::prelude::*;
///
/// // TTS usage
/// let audio = MyEngine::tts()
///     .with_speaker(
///         Speaker::named("Alice")
///             .speak("Hello, world!")
///             .build()
///     )
///     .synthesize(|conversation| {
///         Ok => conversation.into_stream(),
///         Err(e) => Err(e),
///     })
///     .await?;
///
/// // STT microphone usage
/// let stream = MyEngine::stt()
///     .with_microphone("default")
///     .listen(|conversation| {
///         Ok => conversation.into_stream(),
///         Err(e) => Err(e),
///     })
///     .await?;
///
/// // STT file transcription usage
/// let transcript = MyEngine::stt()
///     .transcribe("audio.wav")
///     .emit(|transcript| {
///         Ok => transcript.into_stream(),
///         Err(e) => Err(e),
///     })
///     .await?;
/// ```
pub trait FluentVoice {
    /// Begin a new TTS conversation builder.
    ///
    /// This method returns a conversation builder that can be used to configure
    /// speakers, voice settings, and other TTS parameters before synthesis.
    ///
    /// # Returns
    ///
    /// A new TTS conversation builder instance.
    fn tts() -> impl TtsConversationBuilder;

    /// Begin a new STT conversation builder.
    ///
    /// This method returns a conversation builder that can be used to configure
    /// audio sources, language hints, VAD settings, and other recognition
    /// parameters before starting transcription.
    ///
    /// # Returns
    ///
    /// A new STT conversation builder instance.
    fn stt() -> impl SttConversationBuilder;
}

/// Default implementation entry point for FluentVoice
pub struct FluentVoiceImpl;

impl FluentVoice for FluentVoiceImpl {
    fn tts() -> impl TtsConversationBuilder {
        // Create a default implementation that returns an empty audio stream
        crate::builders::tts_conversation_builder(|_lines, _lang| {
            // Return an empty stream of i16 audio samples
            futures::stream::empty::<i16>()
        })
    }

    fn stt() -> impl SttConversationBuilder {
        // Create a default implementation that returns an empty transcript stream
        crate::builders::stt_conversation_builder(
            |_source,
             _vad,
             _noise,
             _lang,
             _diarization,
             _word_timestamps,
             _timestamps_granularity,
             _punctuation| {
                // Return an empty stream of transcript segments
                futures::stream::empty::<Result<DummySegment, crate::voice_error::VoiceError>>()
            },
        )
    }
}

/// Dummy transcript segment for default implementation
#[derive(Debug, Clone)]
pub struct DummySegment {
    start_ms: u32,
    end_ms: u32,
    text: String,
    speaker_id: Option<String>,
}

impl TranscriptSegment for DummySegment {
    fn start_ms(&self) -> u32 {
        self.start_ms
    }

    fn end_ms(&self) -> u32 {
        self.end_ms
    }

    fn text(&self) -> &str {
        &self.text
    }

    fn speaker_id(&self) -> Option<&str> {
        self.speaker_id.as_deref()
    }
}

/// Implementation of TtsConversationExt for FluentVoiceImpl
impl crate::tts_conversation::TtsConversationExt for FluentVoiceImpl {
    fn builder() -> impl TtsConversationBuilder {
        Self::tts()
    }
}

/// Implementation of SttConversationExt for FluentVoiceImpl
impl crate::stt_conversation::SttConversationExt for FluentVoiceImpl {
    fn builder() -> impl SttConversationBuilder {
        Self::stt()
    }
}
