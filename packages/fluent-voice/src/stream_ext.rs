//! Stream extension traits for TTS synthesis chunk processing
//!
//! Provides the `.on_chunk()` functionality for processing synthesis chunks
//! using cyrup_sugars streaming patterns.

use cyrup_sugars::prelude::MessageChunk;
use cyrup_sugars::{AsyncStream, StreamExt as CyrupStreamExt};
use fluent_voice_domain::VoiceError;
use fluent_voice_domain::{AudioChunk, SynthesisChunk};
use futures_core::Stream;

/// Default error handler for audio streams
///
/// This function provides the default error handling behavior that logs errors
/// with env_logger and returns BadAudioStreamSegment::new(e) for audio stream errors.
pub fn default_audio_chunk_error_handler(result: Result<AudioChunk, VoiceError>) -> AudioChunk {
    match result {
        Ok(chunk) => chunk,
        Err(e) => MessageChunk::bad_chunk(e.to_string()),
    }
}

/// Default error handler for transcript streams
///
/// This function provides the default error handling behavior that logs errors
/// with env_logger and returns empty string for transcript errors.
pub fn default_transcript_error_handler(result: Result<String, VoiceError>) -> String {
    match result {
        Ok(text) => text,
        Err(e) => {
            log::error!("Transcript error: {}", e);
            format!("[ERROR: {}]", e.to_string())
        }
    }
}

/// Extension trait for TTS synthesis streams that provides chunk processing capabilities
///
/// This trait extends AsyncStream<SynthesisChunk> with methods for processing
/// synthesis chunks using the cyrup_sugars patterns.
pub trait TtsStreamExt<T>: Sized + 'static {
    /// Process each synthesis chunk with the provided function.
    ///
    /// This method enables the README.md syntax:
    /// ```ignore
    /// .on_chunk(|synthesis_chunk| {
    ///     Ok => synthesis_chunk.into(),
    ///     Err(e) => AudioChunk::error(e),
    /// })
    /// ```
    ///
    /// The closure receives a `Result<AudioChunk, VoiceError>` and must return
    /// an `AudioChunk`. This allows for transformation and error handling
    /// while maintaining the streaming flow.
    fn on_chunk<F>(self, f: F) -> AsyncStream<AudioChunk>
    where
        F: FnMut(Result<AudioChunk, VoiceError>) -> AudioChunk + Send + 'static;

    /// Process each synthesis chunk with Result-based transformation.
    ///
    /// Similar to `on_chunk` but allows the function to return a Result,
    /// which will be wrapped in SynthesisChunk for continued processing.
    fn on_chunk_result<F>(self, f: F) -> AsyncStream<SynthesisChunk>
    where
        F: FnMut(Result<AudioChunk, VoiceError>) -> Result<AudioChunk, VoiceError> + Send + 'static;

    /// Filter chunks based on a predicate function.
    ///
    /// Only chunks that pass the predicate will be included in the output stream.
    fn filter_chunks<F>(self, f: F) -> AsyncStream<T>
    where
        F: FnMut(&T) -> bool + Send + 'static,
        T: Clone + Send + 'static + cyrup_sugars::NotResult;

    /// Transform chunks to a different type.
    ///
    /// Maps each chunk to a new type using the provided function.
    fn map_chunks<U, F>(self, f: F) -> AsyncStream<U>
    where
        F: FnMut(T) -> U + Send + 'static,
        U: Send + 'static + cyrup_sugars::NotResult;

    /// Tap into the stream for side effects without modifying the chunks.
    ///
    /// Useful for logging, metrics, or other observability operations.
    fn tap_chunks<F>(self, f: F) -> AsyncStream<T>
    where
        F: FnMut(&T) + Send + 'static,
        T: Clone + Send + 'static + cyrup_sugars::NotResult;

    /// Apply default error handling using the built-in error handler.
    ///
    /// This is a convenience method that applies the default error handling
    /// pattern that logs errors with env_logger and returns BadAudioStreamSegment::new(e).
    fn with_default_error_handling(self) -> AsyncStream<AudioChunk>;

    /// Apply custom error handling using a user-provided error handler.
    ///
    /// This method allows users to provide custom error handling logic that
    /// will be applied to stream errors.
    fn on_result<F>(self, error_handler: F) -> AsyncStream<AudioChunk>
    where
        F: FnMut(VoiceError) -> AudioChunk + Send + 'static;
}

impl TtsStreamExt<SynthesisChunk> for AsyncStream<SynthesisChunk> {
    fn on_chunk<F>(self, mut f: F) -> AsyncStream<AudioChunk>
    where
        F: FnMut(Result<AudioChunk, VoiceError>) -> AudioChunk + Send + 'static,
    {
        CyrupStreamExt::on_chunk(self, move |result| match result {
            Ok(synthesis_chunk) => f(Ok(synthesis_chunk.into_chunk())),
            Err(e) => {
                // Convert Box<dyn Error> to VoiceError
                let voice_error = VoiceError::ProcessingError(e.to_string());
                f(Err(voice_error))
            }
        })
    }

    fn on_chunk_result<F>(self, mut f: F) -> AsyncStream<SynthesisChunk>
    where
        F: FnMut(Result<AudioChunk, VoiceError>) -> Result<AudioChunk, VoiceError> + Send + 'static,
    {
        CyrupStreamExt::on_chunk(self, move |result| match result {
            Ok(synthesis_chunk) => {
                let inner = synthesis_chunk.into_chunk();
                let transformed = f(Ok(inner));
                SynthesisChunk::from(transformed)
            }
            Err(_) => {
                let transformed = f(Err(VoiceError::ProcessingError(
                    "Stream processing error".to_string(),
                )));
                SynthesisChunk::from(transformed)
            }
        })
    }

    fn filter_chunks<F>(self, f: F) -> AsyncStream<SynthesisChunk>
    where
        F: FnMut(&SynthesisChunk) -> bool + Send + 'static,
    {
        self.filter_stream(f)
    }

    fn map_chunks<U, F>(self, f: F) -> AsyncStream<U>
    where
        F: FnMut(SynthesisChunk) -> U + Send + 'static,
        U: Send + 'static + cyrup_sugars::NotResult,
    {
        self.map_stream(f)
    }

    fn tap_chunks<F>(self, f: F) -> AsyncStream<SynthesisChunk>
    where
        F: FnMut(&SynthesisChunk) + Send + 'static,
    {
        self.tap_each(f)
    }

    fn with_default_error_handling(self) -> AsyncStream<AudioChunk> {
        TtsStreamExt::on_chunk(self, default_audio_chunk_error_handler)
    }

    fn on_result<F>(self, mut error_handler: F) -> AsyncStream<AudioChunk>
    where
        F: FnMut(VoiceError) -> AudioChunk + Send + 'static,
    {
        TtsStreamExt::on_chunk(self, move |result| match result {
            Ok(chunk) => chunk,
            Err(e) => error_handler(e),
        })
    }
}

impl TtsStreamExt<AudioChunk> for AsyncStream<AudioChunk> {
    fn on_chunk<F>(self, mut f: F) -> AsyncStream<AudioChunk>
    where
        F: FnMut(Result<AudioChunk, VoiceError>) -> AudioChunk + Send + 'static,
    {
        CyrupStreamExt::on_chunk(self, move |result| match result {
            Ok(audio_chunk) => f(Ok(audio_chunk)),
            Err(_) => f(Err(VoiceError::ProcessingError(
                "Stream processing error".to_string(),
            ))),
        })
    }

    fn on_chunk_result<F>(self, mut f: F) -> AsyncStream<SynthesisChunk>
    where
        F: FnMut(Result<AudioChunk, VoiceError>) -> Result<AudioChunk, VoiceError> + Send + 'static,
    {
        CyrupStreamExt::on_chunk(self, move |result| match result {
            Ok(audio_chunk) => {
                let transformed = f(Ok(audio_chunk));
                SynthesisChunk::from(transformed)
            }
            Err(_) => {
                let transformed = f(Err(VoiceError::ProcessingError(
                    "Stream processing error".to_string(),
                )));
                SynthesisChunk::from(transformed)
            }
        })
    }

    fn filter_chunks<F>(self, f: F) -> AsyncStream<AudioChunk>
    where
        F: FnMut(&AudioChunk) -> bool + Send + 'static,
    {
        self.filter_stream(f)
    }

    fn map_chunks<U, F>(self, f: F) -> AsyncStream<U>
    where
        F: FnMut(AudioChunk) -> U + Send + 'static,
        U: Send + 'static + cyrup_sugars::NotResult,
    {
        self.map_stream(f)
    }

    fn tap_chunks<F>(self, f: F) -> AsyncStream<AudioChunk>
    where
        F: FnMut(&AudioChunk) + Send + 'static,
    {
        self.tap_each(f)
    }

    fn with_default_error_handling(self) -> AsyncStream<AudioChunk> {
        TtsStreamExt::on_chunk(self, default_audio_chunk_error_handler)
    }

    fn on_result<F>(self, mut error_handler: F) -> AsyncStream<AudioChunk>
    where
        F: FnMut(VoiceError) -> AudioChunk + Send + 'static,
    {
        TtsStreamExt::on_chunk(self, move |result| match result {
            Ok(chunk) => chunk,
            Err(e) => error_handler(e),
        })
    }
}

/// Extension trait for STT transcript streams that provides chunk processing capabilities
///
/// This trait extends AsyncStream<String> with methods for processing
/// transcript chunks using the cyrup_sugars patterns.
pub trait SttStreamExt<T>: Sized + 'static {
    /// Process each transcript chunk with the provided function.
    ///
    /// This method enables the on_chunk syntax for transcript processing:
    /// ```ignore
    /// .on_chunk(|transcript_result| {
    ///     match transcript_result {
    ///         Ok(text) => text,
    ///         Err(e) => String::new(), // Default error handling
    ///     }
    /// })
    /// ```
    ///
    /// The closure receives a `Result<String, VoiceError>` and must return
    /// a `String`. This allows for transformation and error handling
    /// while maintaining the streaming flow.
    fn on_chunk<F>(self, f: F) -> AsyncStream<String>
    where
        F: FnMut(Result<String, VoiceError>) -> String + Send + 'static;

    /// Process each transcript chunk with Result-based transformation.
    ///
    /// Similar to `on_chunk` but allows the function to return a Result,
    /// which will be handled by the default error handling.
    fn on_chunk_result<F>(self, f: F) -> AsyncStream<String>
    where
        F: FnMut(Result<String, VoiceError>) -> Result<String, VoiceError> + Send + 'static;

    /// Filter chunks based on a predicate function.
    ///
    /// Only chunks that pass the predicate will be included in the output stream.
    fn filter_chunks<F>(self, f: F) -> AsyncStream<T>
    where
        F: FnMut(&T) -> bool + Send + 'static,
        T: Clone + Send + 'static + cyrup_sugars::NotResult;

    /// Transform chunks to a different type.
    ///
    /// Maps each chunk to a new type using the provided function.
    fn map_chunks<U, F>(self, f: F) -> AsyncStream<U>
    where
        F: FnMut(T) -> U + Send + 'static,
        U: Send + 'static + cyrup_sugars::NotResult;

    /// Tap into the stream for side effects without modifying the chunks.
    ///
    /// Useful for logging, metrics, or other observability operations.
    fn tap_chunks<F>(self, f: F) -> AsyncStream<T>
    where
        F: FnMut(&T) + Send + 'static,
        T: Clone + Send + 'static + cyrup_sugars::NotResult;

    /// Apply default error handling using the built-in error handler.
    ///
    /// This is a convenience method that applies the default error handling
    /// pattern that logs errors with env_logger and returns empty string.
    fn with_default_error_handling(self) -> AsyncStream<String>;

    /// Apply custom error handling using a user-provided error handler.
    ///
    /// This method allows users to provide custom error handling logic that
    /// will be applied to stream errors.
    fn on_result<F>(self, error_handler: F) -> AsyncStream<String>
    where
        F: FnMut(VoiceError) -> String + Send + 'static;
}

impl SttStreamExt<String> for AsyncStream<String> {
    fn on_chunk<F>(self, mut f: F) -> AsyncStream<String>
    where
        F: FnMut(Result<String, VoiceError>) -> String + Send + 'static,
    {
        CyrupStreamExt::on_chunk(self, move |result| match result {
            Ok(text) => f(Ok(text)),
            Err(_) => f(Err(VoiceError::ProcessingError(
                "Stream processing error".to_string(),
            ))),
        })
    }

    fn on_chunk_result<F>(self, mut f: F) -> AsyncStream<String>
    where
        F: FnMut(Result<String, VoiceError>) -> Result<String, VoiceError> + Send + 'static,
    {
        CyrupStreamExt::on_chunk(self, move |result| match result {
            Ok(text) => {
                let transformed = f(Ok(text));
                match transformed {
                    Ok(t) => t,
                    Err(e) => {
                        log::error!("Transcript processing error: {}", e);
                        String::new()
                    }
                }
            }
            Err(_) => {
                let transformed = f(Err(VoiceError::ProcessingError(
                    "Stream processing error".to_string(),
                )));
                match transformed {
                    Ok(t) => t,
                    Err(e) => {
                        log::error!("Transcript processing error: {}", e);
                        String::new()
                    }
                }
            }
        })
    }

    fn filter_chunks<F>(self, f: F) -> AsyncStream<String>
    where
        F: FnMut(&String) -> bool + Send + 'static,
    {
        self.filter_stream(f)
    }

    fn map_chunks<U, F>(self, f: F) -> AsyncStream<U>
    where
        F: FnMut(String) -> U + Send + 'static,
        U: Send + 'static + cyrup_sugars::NotResult,
    {
        self.map_stream(f)
    }

    fn tap_chunks<F>(self, f: F) -> AsyncStream<String>
    where
        F: FnMut(&String) + Send + 'static,
    {
        self.tap_each(f)
    }

    fn with_default_error_handling(self) -> AsyncStream<String> {
        SttStreamExt::on_chunk(self, default_transcript_error_handler)
    }

    fn on_result<F>(self, mut error_handler: F) -> AsyncStream<String>
    where
        F: FnMut(VoiceError) -> String + Send + 'static,
    {
        SttStreamExt::on_chunk(self, move |result| match result {
            Ok(text) => text,
            Err(e) => error_handler(e),
        })
    }
}

/// Macro to support the specific syntax shown in README.md
///
/// Enables pattern matching syntax like:
/// ```ignore
/// on_chunk!(|synthesis_chunk| {
///     Ok => synthesis_chunk.into(),
///     Err(e) => AudioChunk::error(e),
/// })
/// ```
#[macro_export]
macro_rules! on_chunk {
    (|$param:ident| { Ok => $ok_expr:expr, Err($err:pat) => $err_expr:expr $(,)? }) => {
        move |$param: Result<$crate::audio_chunk::AudioChunk, fluent_voice_domain::VoiceError>| {
            match $param {
                Ok(chunk) => {
                    let $param = chunk;
                    $ok_expr
                }
                Err($err) => $err_expr,
            }
        }
    };
}

/// Helper function to convert TranscriptStream to Stream<String>
///
/// This function converts a transcript stream (which yields Result<TranscriptionSegment, VoiceError>)
/// to a stream of strings for direct consumption.
pub fn transcript_stream_to_string_stream<S, T>(
    stream: S,
) -> impl Stream<Item = String> + Send + Unpin
where
    S: Stream<Item = Result<T, VoiceError>> + Send + Unpin + 'static,
    T: fluent_voice_domain::transcription::TranscriptionSegment,
{
    use futures_util::StreamExt;

    stream.map(|result| match result {
        Ok(segment) => segment.text().to_string(),
        Err(e) => {
            log::error!("Transcript error: {}", e);
            String::new()
        }
    })
}
