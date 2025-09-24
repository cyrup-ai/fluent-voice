//! Finished-transcript abstraction used by the fluent Whisper builder.
//!
//! A `Transcript` is little more than a typed wrapper around
//! `Vec<TtsChunk>` that offers ergonomic helpers without leaking the
//! internal chunk representation to every downstream crate.
//!
//! Down-stream code obtains a `Transcript` via the builder chain:
//
//! ```rust,no_run
//! use cyterm::whisper::{builder::transcribe, WhisperStreamExt};
//!
//! let transcript = transcribe("file.wav")
//!     .emit()        // -> WhisperStream<TtsChunk>
//!     .collect()     // -> Transcript
//!     .await;
//!
//! println!("{}", transcript.as_text());
//! ```
//!
//! The actual `collect()` implementation resides in the builder; this
//! module focuses only on the data type and its helpers.

use crate::types::TtsChunk;
use std::ops::Index;

/// A completed speech-to-text result containing every decoded chunk.
///
/// The struct is intentionally **not** `pub` on its fields; use the
/// provided helper methods or iterate through `chunks()` if you need
/// fine-grained access.
#[derive(Debug, Clone, Default)]
pub struct Transcript {
    chunks: Vec<TtsChunk>,
}

impl Transcript {
    /* ----------------------------------------------------------------
    Public helpers
    ---------------------------------------------------------------- */

    /// Human-readable text with **no** line breaks inserted.
    ///
    /// Concatenates `chunk.text` in chronological order.
    pub fn as_text(&self) -> String {
        self.chunks
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join("")
    }

    /// Borrow the underlying slice of chunks.
    pub fn chunks(&self) -> &[TtsChunk] {
        &self.chunks
    }

    /// Returns the number of chunks in the transcript.
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    /// Returns true if the transcript contains no chunks.
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Returns an iterator over the chunks.
    pub fn iter(&self) -> impl Iterator<Item = &TtsChunk> {
        self.chunks.iter()
    }

    /* ----------------------------------------------------------------
    Internal helpers – used by the builder / stream collector.
    ---------------------------------------------------------------- */

    /// Create an empty transcript.
    #[allow(dead_code)] // Library code - used by fluent-voice builders
    #[inline]
    pub fn new(chunks: Vec<TtsChunk>) -> Self {
        Self { chunks }
    }

    /// Create an empty transcript.
    #[inline]
    pub fn empty() -> Self {
        Self { chunks: Vec::new() }
    }

    /// Push a single chunk; used while collecting the stream.
    #[inline]
    pub(crate) fn push(&mut self, chunk: TtsChunk) {
        self.chunks.push(chunk);
    }

    /// Consume and return the inner `Vec<TtsChunk>`.
    #[allow(dead_code)] // Library code - used by fluent-voice builders
    #[inline]
    pub(crate) fn into_inner(self) -> Vec<TtsChunk> {
        self.chunks
    }
}

impl Index<usize> for Transcript {
    type Output = TtsChunk;

    fn index(&self, index: usize) -> &Self::Output {
        &self.chunks[index]
    }
}
