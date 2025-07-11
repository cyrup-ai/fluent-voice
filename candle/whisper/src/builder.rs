//! Whisper STT Engine using the fluent-voice builder-for-builders macro system.
//!
//! This module uses the `stt_engine!` macro to generate a complete STT engine
//! implementation with polymorphic builders for both microphone and file transcription.
//!
//! # Example Usage
//!
//! ```no_run
//! use fluent_voice::prelude::*;
//! use whisper::WhisperEngine;
//!
//! // Microphone transcription (live audio)
//! let stream = WhisperEngine::conversation()
//!     .with_microphone("default")  // -> MicrophoneBuilder
//!     .vad_mode(VadMode::Accurate)
//!     .language_hint(Language("en-US"))
//!     .listen(|conversation| {
//!         Ok => conversation.into_stream(),
//!         Err(e) => Err(e),
//!     })
//!     .await?;
//!
//! // File transcription (batch processing)
//! let transcript = WhisperEngine::conversation()
//!     .transcribe("audio.wav")  // -> TranscriptionBuilder
//!     .with_progress("{file} :: {percent}%")
//!     .emit(|transcript| {
//!         Ok => transcript.into_stream(),
//!         Err(e) => Err(e),
//!     })
//!     .await?;
//!
//! // Convenience methods for file transcription
//! let text_stream = WhisperEngine::conversation()
//!     .transcribe("audio.wav")
//!     .as_text();  // Stream<Item = String>
//!
//! let full_transcript = WhisperEngine::conversation()
//!     .transcribe("audio.wav")
//!     .collect()
//!     .await?;
//! ```

use fluent_voice_macros::stt_engine;
use fluent_voice_domain::*;

// Re-export the types the macro will use
use crate::{stream::WhisperStream, transcript::Transcript, types::TtsChunk};

// Generate complete Whisper STT engine with polymorphic builders
stt_engine!(
    engine = WhisperEngine,
    segment = TtsChunk,
    stream = WhisperStream,
    transcript = Transcript,
    /// Whisper local STT engine implementation with polymorphic builders.
    ///
    /// This engine provides both microphone capture and file transcription
    /// capabilities through type-safe polymorphic builders. The macro generates
    /// complete implementations of all fluent-voice traits including:
    ///
    /// - `SttConversationBuilder` - base builder with polymorphic branching
    /// - `MicrophoneBuilder` - live audio capture with `listen()` terminal method
    /// - `TranscriptionBuilder` - file processing with `emit()`, `collect()`, `as_text()` methods
    /// - `SttEngine` and `SttConversationExt` - engine registration traits
    /// - `FluentVoice` - unified entry point trait
    ///
    /// All heavy lifting is off-loaded to `spawn_blocking` workers so the public
    /// API remains lazy - no CPU work starts until streams are first polled.
);

// The macro generates the following concrete types:
// - `WhisperEngine` - the main engine struct
// - `SttBuilder` - base builder implementing SttConversationBuilder
// - `MicBuilder` - microphone builder implementing MicrophoneBuilder
// - `TransBuilder` - transcription builder implementing TranscriptionBuilder
// - `Session` - conversation object implementing SttConversation

// TODO: Replace the macro-generated todo!() implementations with actual Whisper inference:
//
// 1. Session::transcribe_inner() - Use existing whisper.rs inference engine
// 2. MicBuilder microphone capture - Implement real-time audio streaming
// 3. TransBuilder file processing - Use existing pcm_decode.rs and whisper.rs
// 4. Error handling - Map Whisper errors to VoiceError
//
// The macro provides the complete trait scaffolding, just need to fill in the actual
// Whisper inference implementation in the generated todo!() methods.

/// Whisper transcription builder for file-based transcription
pub struct WhisperBuilder {
    path: String,
}

/// Create a new transcription builder for the given audio file path
pub fn transcribe<P: Into<String>>(path: P) -> WhisperBuilder {
    WhisperBuilder {
        path: path.into(),
    }
}
