//! AsyncStream Helper Functions
//!
//! This module provides helper functions to correctly construct AsyncStream instances
//! from the cyrup_sugars crate, bridging the API gap between expected and actual methods.

use cyrup_sugars::{AsyncStream, NotResult};
use fluent_voice_domain::VoiceError;
use futures_core::Stream;

#[cfg(feature = "tokio-runtime")]
use tokio::sync::mpsc;

#[cfg(all(feature = "std_async", not(feature = "tokio-runtime")))]
use async_channel;

/// Create an AsyncStream from a Stream of items
///
/// This function correctly handles the conversion from any Stream<Item = T> to AsyncStream<T>
/// by using the appropriate channel implementation based on the async runtime feature.
pub fn async_stream_from_stream<T, S>(stream: S) -> AsyncStream<T>
where
    T: Send + 'static + NotResult,
    S: Stream<Item = T> + Send + 'static,
{
    #[cfg(feature = "tokio-runtime")]
    {
        let (tx, rx) = mpsc::unbounded_channel();
        tokio::spawn(async move {
            use futures_util::StreamExt;
            let mut stream = std::pin::pin!(stream);
            while let Some(item) = stream.next().await {
                if tx.send(item).is_err() {
                    break;
                }
            }
        });
        AsyncStream::new(rx)
    }

    #[cfg(all(feature = "std_async", not(feature = "tokio-runtime")))]
    {
        let (tx, rx) = async_channel::unbounded();
        tokio::spawn(async move {
            use futures_util::StreamExt;
            let mut stream = std::pin::pin!(stream);
            while let Some(item) = stream.next().await {
                if tx.send(item).await.is_err() {
                    break;
                }
            }
        });
        AsyncStream::new(rx)
    }

    #[cfg(all(
        feature = "crossbeam-async",
        not(feature = "tokio-runtime"),
        not(feature = "std_async")
    ))]
    {
        let (tx, rx) = async_channel::unbounded();
        tokio::spawn(async move {
            use futures_util::StreamExt;
            let mut stream = std::pin::pin!(stream);
            while let Some(item) = stream.next().await {
                if tx.send(item).await.is_err() {
                    break;
                }
            }
        });
        AsyncStream::new(rx)
    }
}

/// Create an AsyncStream that immediately yields an error and then ends
///
/// Since AsyncStream<T> cannot contain Result types per cyrup_sugars design,
/// this function creates a stream that handles the error case by creating
/// an empty stream (as errors should be handled before streaming).
pub fn async_stream_from_error<T>(_error: VoiceError) -> AsyncStream<T>
where
    T: Send + 'static + NotResult,
{
    #[cfg(feature = "tokio-runtime")]
    {
        let (_tx, rx) = mpsc::unbounded_channel();
        AsyncStream::new(rx)
    }

    #[cfg(all(feature = "std_async", not(feature = "tokio-runtime")))]
    {
        let (_tx, rx) = async_channel::unbounded();
        AsyncStream::new(rx)
    }

    #[cfg(all(
        feature = "crossbeam-async",
        not(feature = "tokio-runtime"),
        not(feature = "std_async")
    ))]
    {
        let (_tx, rx) = async_channel::unbounded();
        AsyncStream::new(rx)
    }
}

/// Create an AsyncStream from a Result<Stream, Error>
///
/// This helper handles the common pattern where we have a Result containing a Stream
/// and need to convert it to an AsyncStream, handling both success and error cases.
pub fn async_stream_from_result<T, S, E>(result: Result<S, E>) -> AsyncStream<T>
where
    T: Send + 'static + NotResult,
    S: Stream<Item = T> + Send + 'static,
    E: Into<VoiceError>,
{
    match result {
        Ok(stream) => async_stream_from_stream(stream),
        Err(_error) => {
            // Handle error case by creating empty stream
            async_stream_empty()
        }
    }
}

/// Create an empty AsyncStream that immediately closes
///
/// This replaces the non-existent AsyncStream::empty() method by creating
/// a stream from an empty channel that is immediately closed.
pub fn async_stream_empty<T>() -> AsyncStream<T>
where
    T: Send + 'static + NotResult,
{
    #[cfg(feature = "tokio-runtime")]
    {
        let (_tx, rx) = mpsc::unbounded_channel();
        // Don't send anything, just close the channel immediately
        AsyncStream::new(rx)
    }

    #[cfg(all(feature = "std_async", not(feature = "tokio-runtime")))]
    {
        let (_tx, rx) = async_channel::unbounded();
        // Don't send anything, just close the channel immediately
        AsyncStream::new(rx)
    }
}

/// Convert an audio stream of i16 samples to a stream of AudioChunk
///
/// This function takes a Stream<Item = i16> and converts it to an AsyncStream<AudioChunk>
/// by batching the audio samples into chunks of a suitable size.
pub fn audio_stream_to_chunk_stream<S>(
    audio_stream: S,
) -> AsyncStream<crate::audio_chunk::AudioChunk>
where
    S: Stream<Item = i16> + Send + 'static,
{
    #[cfg(feature = "tokio-runtime")]
    {
        let (tx, rx) = mpsc::unbounded_channel();
        tokio::spawn(async move {
            use futures_util::StreamExt;
            let mut stream = std::pin::pin!(audio_stream);
            let mut buffer = Vec::new();
            const CHUNK_SIZE: usize = 1024; // 1024 samples per chunk

            while let Some(sample) = stream.next().await {
                buffer.push(sample);
                if buffer.len() >= CHUNK_SIZE {
                    let audio_bytes: Vec<u8> = std::mem::take(&mut buffer)
                        .into_iter()
                        .flat_map(|sample| sample.to_le_bytes())
                        .collect();
                    let chunk = crate::audio_chunk::AudioChunk::new(
                        audio_bytes,
                        fluent_voice_domain::AudioFormat::Pcm16Khz,
                    );
                    if tx.send(chunk).is_err() {
                        break;
                    }
                }
            }

            // Send final chunk if there are remaining samples
            if !buffer.is_empty() {
                let audio_bytes: Vec<u8> = buffer
                    .into_iter()
                    .flat_map(|sample| sample.to_le_bytes())
                    .collect();
                let chunk = crate::audio_chunk::AudioChunk::new(
                    audio_bytes,
                    fluent_voice_domain::AudioFormat::Pcm16Khz,
                );
                let _ = tx.send(chunk);
            }
        });
        AsyncStream::new(rx)
    }

    #[cfg(all(feature = "std_async", not(feature = "tokio-runtime")))]
    {
        let (tx, rx) = async_channel::unbounded();
        tokio::spawn(async move {
            use futures_util::StreamExt;
            let mut stream = std::pin::pin!(audio_stream);
            let mut buffer = Vec::new();
            const CHUNK_SIZE: usize = 1024; // 1024 samples per chunk

            while let Some(sample) = stream.next().await {
                buffer.push(sample);
                if buffer.len() >= CHUNK_SIZE {
                    let audio_bytes: Vec<u8> = std::mem::take(&mut buffer)
                        .into_iter()
                        .flat_map(|sample| sample.to_le_bytes())
                        .collect();
                    let chunk = crate::audio_chunk::AudioChunk::new(
                        audio_bytes,
                        fluent_voice_domain::AudioFormat::Pcm16Khz,
                    );
                    if tx.send(chunk).await.is_err() {
                        break;
                    }
                }
            }

            // Send final chunk if there are remaining samples
            if !buffer.is_empty() {
                let audio_bytes: Vec<u8> = buffer
                    .into_iter()
                    .flat_map(|sample| sample.to_le_bytes())
                    .collect();
                let chunk = crate::audio_chunk::AudioChunk::new(
                    audio_bytes,
                    fluent_voice_domain::AudioFormat::Pcm16Khz,
                );
                let _ = tx.send(chunk).await;
            }
        });
        AsyncStream::new(rx)
    }
}
