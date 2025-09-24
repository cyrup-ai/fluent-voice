#![allow(dead_code)]

use crate::client::Result;
use async_stream::stream;
use bytes::Bytes;
use futures_util::{Stream, StreamExt, pin_mut};
use std::sync::mpsc;
use std::{fs::File, io::prelude::*};
use tracing::{debug, error};

mod playback;
pub use playback::{play, stream_audio};

/// Save audio to a file
pub fn save(filename: &str, data: Bytes) -> Result<()> {
    let mut file = File::create(filename)?;
    file.write_all(&data)?;
    Ok(())
}

pub fn text_chunker<S>(text_stream: S) -> impl Stream<Item = Result<String>>
where
    S: Stream<Item = String> + Send + 'static,
{
    let splitters = [
        '.', ',', '?', '!', ';', ':', '—', '-', '(', ')', '[', ']', '{', '}', ' ',
    ];
    let mut buf = String::new();
    let (tx, rx) = mpsc::channel::<String>();

    tokio::spawn(async move {
        debug!("Starting text chunker for stream processing");
        pin_mut!(text_stream);
        while let Some(text) = text_stream.next().await {
            debug!("Processing text chunk: {} chars", text.len());
            debug!(
                "Buffer state: {} chars, ends_with_splitter: {}",
                buf.len(),
                buf.ends_with(splitters)
            );

            if buf.ends_with(splitters) {
                let chunk = format!("{} ", buf.as_str());
                if let Err(_) = tx.send(chunk.clone()) {
                    error!("Failed to send text chunk: receiver disconnected");
                    break; // Gracefully exit on receiver disconnection
                }
                debug!("Successfully sent text chunk: {} chars", chunk.len());
                buf = text;
            } else if text.starts_with(splitters) {
                // Safe character extraction with fallback
                let first_char = text.char_indices().next().map(|(_, c)| c).unwrap_or(' '); // Provide safe default for empty strings

                let chunk = format!("{}{} ", buf.as_str(), first_char);
                if let Err(_) = tx.send(chunk.clone()) {
                    error!("Failed to send text chunk with separator: receiver disconnected");
                    break; // Gracefully exit on receiver disconnection
                }
                debug!(
                    "Successfully sent text chunk with separator: {} chars",
                    chunk.len()
                );

                // Safe string slicing with bounds check
                buf = if text.len() > 1 {
                    text[1..].to_string()
                } else {
                    String::new()
                };
            } else {
                buf.push_str(&text);
            }
        }

        // Send final buffer if not empty
        if !buf.is_empty() {
            if let Err(_) = tx.send(buf.clone()) {
                error!("Failed to send final text buffer: receiver disconnected");
            } else {
                debug!("Successfully sent final text buffer: {} chars", buf.len());
            }
        }

        debug!("Text chunker task completed");
    });

    stream! {
        while let Ok(buf) = rx.recv() {
            yield Ok(buf);
        }

        // If we exit the loop, it means the sender was dropped
        // This is normal termination, not an error condition
        debug!("Text chunker stream completed: sender disconnected");
    }
}
