use fluent_voice_domain::{AudioChunk, VoiceError};
use futures::Stream;
use std::future::Future;
use std::pin::Pin;

/// Extension trait to add `.play()` method to audio streams
pub trait AudioStreamExt {
    /// Play the audio stream using rodio, handling all playback internally
    fn play(self) -> Pin<Box<dyn Future<Output = Result<(), VoiceError>> + Send>>;
}

impl<S> AudioStreamExt for S
where
    S: Stream<Item = AudioChunk> + Send + Unpin + 'static,
{
    fn play(self) -> Pin<Box<dyn Future<Output = Result<(), VoiceError>> + Send>>
    where
        Self: Send + Unpin + 'static,
    {
        use crate::audio_stream::AudioStream;
        let audio_stream = AudioStream::new(Box::pin(self));
        Box::pin(audio_stream.play())
    }
}
