//! Fluent, ergonomic interface for launching a Whisper transcription.
//!
//! # Example
//! ```no_run
//! use cyterm::whisper::{Whisper, WhisperStreamExt};
//! # async fn demo() -> anyhow::Result<()> {
//! let text = Whisper::transcribe("./assets/audio.mp3")
//!     .with_progress("{file} :: {percent}%")
//!     .emit()                               // => WhisperStream<TtsChunk>
//!     .collect_with(|result| match result { // => String
//!         Ok(t)  => t.as_text(),
//!         Err(e) => format!("transcription failed: {e}"),
//!     })
//!     .await;
//! # Ok(()) }
//! ```
//!
//! Internally the heavy lifting is off-loaded to a `spawn_blocking`
//! worker so that the public `emit()` remains **synchronous** and lazy
//! — no CPU work starts until the returned stream is first polled.
//!
//! ⚠ **Implementation note:** the real Whisper decoding is still a stub.
//! Replace `internal::decode_file_to_stream` with actual ASR logic.

#![cfg(feature = "tokio")]

use futures::{Stream, StreamExt};
use tokio::sync::mpsc;

use crate::{
    async_task::AsyncStream,
    whisper::{WhisperStream, pcm_decode, transcript::Transcript, types::TtsChunk},
};

/// Zero-sized helper used purely for its associated `transcribe()`
/// function.  Mirrors the familiar `std::fs::File::open` style API.
pub struct Whisper;

impl Whisper {
    /// Begin building a transcription job from a file path.
    pub fn transcribe<P: Into<String>>(path: P) -> WhisperBuilder {
        WhisperBuilder {
            source: path.into(),
            progress_template: None,
        }
    }
}

/// Builder returned by `Whisper::transcribe`.
#[derive(Debug, Clone)]
pub struct WhisperBuilder {
    source: String,
    progress_template: Option<String>,
}

impl WhisperBuilder {
    /// Attach a progress message template.  Use `{file}` and
    /// `{percent}` placeholders.
    pub fn with_progress<S: Into<String>>(mut self, template: S) -> Self {
        self.progress_template = Some(template.into());
        self
    }

    /// Produce a *lazy* `WhisperStream<TtsChunk>`.
    ///
    /// No decoding occurs until the caller begins to poll the stream.
    pub fn emit(self) -> WhisperStream {
        let (tx, rx) = mpsc::unbounded_channel::<TtsChunk>();
        let file = self.source.clone();
        let tmpl = self.progress_template.clone();

        tokio::spawn(async move {
            // Off-load CPU to a dedicated blocking thread.
            if let Err(e) = tokio::task::spawn_blocking(move || {
                internal::decode_file_to_stream(&file, tmpl.as_deref(), tx)
            })
            .await
            {
                eprintln!("whisper worker panicked: {e}");
            }
        });

        AsyncStream::new(rx)
    }

    /// Drain the stream and gather a [`Transcript`].
    pub async fn collect(self) -> anyhow::Result<Transcript> {
        let mut transcript = Transcript::default();
        let mut stream = self.emit();
        while let Some(chunk) = stream.next().await {
            transcript.push(chunk);
        }
        Ok(transcript)
    }

    /// Variant that accepts a user-supplied closure to post-process the
    /// result (success or failure) in one go.
    pub async fn collect_with<F, R>(self, handler: F) -> R
    where
        F: FnOnce(anyhow::Result<Transcript>) -> R,
    {
        let res = self.collect().await;
        handler(res)
    }

    /// Convenience: immediately obtain a `Stream<Item = String>` with
    /// only the plain text of each chunk.
    pub fn as_text(self) -> impl Stream<Item = String> {
        self.emit().map(|c| c.text)
    }
}

/* --------------------------------------------------------------------
Internal: placeholder bridge to actual Whisper decoding.
-------------------------------------------------------------------- */

mod internal {
    use super::*;
    use anyhow::Result;

    pub fn decode_file_to_stream(
        file: &str,
        template: Option<&str>,
        tx: mpsc::UnboundedSender<TtsChunk>,
    ) -> Result<()> {
        // Load PCM just so we can know the duration for the demo stub.
        let (pcm, sr) = pcm_decode::pcm_decode(file)?;
        let total_secs = pcm.len() as f64 / sr as f64;

        // ----- fake one-chunk transcript ---------------------------------
        let fake = TtsChunk {
            start: 0.0,
            end: total_secs,
            duration: total_secs,
            tokens: vec![],
            text: "[stub transcript]".into(),
            avg_logprob: 0.0,
            no_speech_prob: 0.0,
            temperature: 0.0,
            compression_ratio: 0.0,
        };
        tx.send(fake).ok();
        drop(tx); // close channel so stream ends
        // -----------------------------------------------------------------

        // Print progress once at 100 %.
        if let Some(t) = template {
            eprintln!("{}", t.replace("{file}", file).replace("{percent}", "100"));
        }

        Ok(())
    }
}
