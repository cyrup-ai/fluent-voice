//! Stream extension traits for TTS synthesis chunk processing
//!
//! Provides the `.on_chunk()` functionality for processing synthesis chunks
//! using cyrup_sugars streaming patterns.

use crate::audio_chunk::{AudioChunk, SynthesisChunk};
use cyrup_sugars::{AsyncStream, AsyncTask, StreamExt as CyrupStreamExt};
use fluent_voice_domain::VoiceError;

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
}

impl TtsStreamExt<SynthesisChunk> for AsyncStream<SynthesisChunk> {
    fn on_chunk<F>(self, mut f: F) -> AsyncStream<AudioChunk>
    where
        F: FnMut(Result<AudioChunk, VoiceError>) -> AudioChunk + Send + 'static,
    {
        self.on_chunk(move |result| match result {
            Ok(synthesis_chunk) => f(synthesis_chunk.into_inner()),
            Err(_) => {
                // This shouldn't happen with SynthesisChunk since it wraps Results,
                // but we handle it gracefully
                f(Err(VoiceError::unknown("Stream processing error")))
            }
        })
    }

    fn on_chunk_result<F>(self, mut f: F) -> AsyncStream<SynthesisChunk>
    where
        F: FnMut(Result<AudioChunk, VoiceError>) -> Result<AudioChunk, VoiceError> + Send + 'static,
    {
        self.on_chunk(move |result| match result {
            Ok(synthesis_chunk) => {
                let inner = synthesis_chunk.into_inner();
                let transformed = f(inner);
                SynthesisChunk::from(transformed)
            }
            Err(_) => {
                let transformed = f(Err(VoiceError::unknown("Stream processing error")));
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
}

impl TtsStreamExt<AudioChunk> for AsyncStream<AudioChunk> {
    fn on_chunk<F>(self, mut f: F) -> AsyncStream<AudioChunk>
    where
        F: FnMut(Result<AudioChunk, VoiceError>) -> AudioChunk + Send + 'static,
    {
        self.on_chunk(move |result| match result {
            Ok(audio_chunk) => f(Ok(audio_chunk)),
            Err(_) => f(Err(VoiceError::unknown("Stream processing error"))),
        })
    }

    fn on_chunk_result<F>(self, mut f: F) -> AsyncStream<SynthesisChunk>
    where
        F: FnMut(Result<AudioChunk, VoiceError>) -> Result<AudioChunk, VoiceError> + Send + 'static,
    {
        self.on_chunk(move |result| match result {
            Ok(audio_chunk) => {
                let transformed = f(Ok(audio_chunk));
                SynthesisChunk::from(transformed)
            }
            Err(_) => {
                let transformed = f(Err(VoiceError::unknown("Stream processing error")));
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


