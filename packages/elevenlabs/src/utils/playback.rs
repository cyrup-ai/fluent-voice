use crate::client::Result;
use bytes::Bytes;
use bytes::{BufMut, BytesMut};
use futures_util::{Stream, StreamExt, pin_mut};
use rodio::Decoder;

/// Play audio
pub fn play(data: Bytes) -> Result<()> {
    let stream_handle = rodio::OutputStreamBuilder::open_default_stream()?;
    let sink = rodio::Sink::connect_new(&stream_handle.mixer());
    let source = Decoder::new(std::io::Cursor::new(data))?;
    sink.append(source);
    sink.sleep_until_end();
    Ok(())
}

// Define constants for audio processing
const AUDIO_CHUNK_SIZE: usize = 16384; // Size of audio chunks to decode and play
const PLAYER_LOOP_SLEEP_DURATION: std::time::Duration = std::time::Duration::from_millis(50); // Duration to sleep in the player loop

pub async fn stream_audio(data: impl Stream<Item = Result<Bytes>>) -> Result<()> {
    pin_mut!(data); // Pin the stream to allow calling `next()`

    // Initialize a buffer to accumulate incoming bytes.
    // The initial capacity is set to AUDIO_CHUNK_SIZE to potentially avoid reallocations
    // if incoming data segments are often around this size.
    let mut buf = BytesMut::with_capacity(AUDIO_CHUNK_SIZE);

    // Setup Rodio output stream and sink.
    // `audio_output_stream` must be kept alive for the duration of playback.
    let audio_output_stream = rodio::OutputStreamBuilder::open_default_stream()?;
    let audio_sink = rodio::Sink::connect_new(&audio_output_stream.mixer());

    while let Some(resulting_bytes) = data.next().await {
        let bytes = resulting_bytes?; // Handle potential errors from the stream
        buf.put(bytes); // Add received bytes to the buffer

        let mut chunk_appended_in_this_iteration = false;

        // Process full chunks from the buffer
        while buf.len() >= AUDIO_CHUNK_SIZE {
            let audio_data = buf.split_to(AUDIO_CHUNK_SIZE).freeze();
            let cursor = std::io::Cursor::new(audio_data); // Decoder needs Read + Seek
            let source = rodio::Decoder::new(cursor)?; // Decode the chunk
            audio_sink.append(source); // Append decoded audio to the sink
            chunk_appended_in_this_iteration = true;
        }

        if chunk_appended_in_this_iteration {
            // If we've added one or more chunks to the sink,
            // sleep for a short duration. This serves two purposes:
            // 1. Yields control to the Tokio runtime, allowing other async tasks to progress.
            // 2. Paces the production of audio data, preventing the sink's queue
            //    from growing excessively if the input stream is very fast,
            //    which could lead to high memory consumption.
            tokio::time::sleep(PLAYER_LOOP_SLEEP_DURATION).await;
        }
    }

    // Play any remaining bytes in the buffer after the stream has ended
    if !buf.is_empty() {
        let audio_data = buf.freeze();
        let cursor = std::io::Cursor::new(audio_data);
        let source = rodio::Decoder::new(cursor)?;
        audio_sink.append(source);
    }

    // Wait for the audio sink to finish playing all appended sources using async approach
    while !audio_sink.empty() {
        tokio::time::sleep(PLAYER_LOOP_SLEEP_DURATION).await;
    }

    // Keep the output stream alive until playback is finished.
    // This is implicitly handled as `audio_output_stream` is in scope until the function returns.
    // We can add an explicit drop or use `_audio_output_stream = audio_output_stream;` if preferred for clarity,
    // but it's not strictly necessary here due to RAII.
    drop(audio_output_stream);

    Ok(())
}
