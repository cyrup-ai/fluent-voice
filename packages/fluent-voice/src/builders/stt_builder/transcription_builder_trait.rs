//! TranscriptionBuilder trait implementation - Part 2

use super::transcript_impl::TranscriptImpl;
use super::transcription_builder::TranscriptionBuilderImpl;
use crate::stream_ext::transcript_stream_to_string_stream;
use core::future::Future;
use fluent_voice_domain::{
    language::Language,
    noise_reduction::NoiseReduction,
    speech_source::SpeechSource,
    timestamps::{Diarization, Punctuation, TimestampsGranularity, WordTimestamps},
    transcription::{TranscriptionSegment, TranscriptionStream},
    vad_mode::VadMode,
    VoiceError,
};
use futures::stream;
use futures::StreamExt;
use futures_core::Stream;
use std::pin::Pin;

impl<S> crate::stt_conversation::TranscriptionBuilder for TranscriptionBuilderImpl<S>
where
    S: TranscriptionStream + 'static,
{
    type Transcript = TranscriptImpl<S>;

    fn vad_mode(mut self, mode: VadMode) -> Self {
        self.vad_mode = Some(mode);
        self
    }

    fn noise_reduction(mut self, level: NoiseReduction) -> Self {
        self.noise_reduction = Some(level);
        self
    }

    fn language_hint(mut self, lang: Language) -> Self {
        self.language_hint = Some(lang);
        self
    }

    fn diarization(mut self, d: Diarization) -> Self {
        self.diarization = Some(d);
        self
    }

    fn word_timestamps(mut self, w: WordTimestamps) -> Self {
        self.word_timestamps = Some(w);
        self
    }

    fn timestamps_granularity(mut self, g: TimestampsGranularity) -> Self {
        self.timestamps_granularity = Some(g);
        self
    }

    fn punctuation(mut self, p: Punctuation) -> Self {
        self.punctuation = Some(p);
        self
    }

    fn with_progress<S2: Into<String>>(mut self, template: S2) -> Self {
        self.progress_template = Some(template.into());
        self
    }

    fn emit(self) -> impl Stream<Item = String> + Send + Unpin {
        // Create a stream that delegates to the async result when polled
        let stream_fut = async move {
            match self.create_transcript().await {
                Ok(transcript) => {
                    // Convert transcript stream to string stream
                    let text_stream = transcript_stream_to_string_stream(transcript.stream);
                    Box::pin(text_stream) as Pin<Box<dyn Stream<Item = String> + Send>>
                }
                Err(e) => {
                    // Log error and return empty stream
                    log::error!("Transcription error: {}", e);
                    Box::pin(stream::empty::<String>())
                        as Pin<Box<dyn Stream<Item = String> + Send>>
                }
            }
        };

        Box::pin(stream::once(stream_fut).flatten()) as Pin<Box<dyn Stream<Item = String> + Send>>
    }

    fn collect(self) -> impl Future<Output = Result<Self::Transcript, VoiceError>> + Send {
        async move { self.create_transcript().await }
    }

    fn collect_with<F, R>(self, handler: F) -> impl Future<Output = R> + Send
    where
        F: FnOnce(Result<Self::Transcript, VoiceError>) -> R + Send + 'static,
    {
        async move {
            let result = self.collect().await;
            handler(result)
        }
    }

    fn into_text_stream(self) -> impl Stream<Item = String> + Send {
        // Use Box<dyn Stream> to erase the complex return type
        let stream_fut = async move {
            match self.create_transcript().await {
                Ok(transcript) => {
                    // Use map to transform successful segments to text
                    let text_stream = transcript.stream.map(|result| match result {
                        Ok(segment) => segment.text().to_string(),
                        Err(_) => "".to_string(), // Empty string for errors
                    });
                    Box::pin(text_stream) as Pin<Box<dyn Stream<Item = String> + Send>>
                }
                Err(_) => {
                    // Return empty stream on error
                    Box::pin(stream::empty::<String>())
                        as Pin<Box<dyn Stream<Item = String> + Send>>
                }
            }
        };

        // Create a stream that delegates to the async result when polled
        Box::pin(stream::once(stream_fut).flatten()) as Pin<Box<dyn Stream<Item = String> + Send>>
    }

    fn transcribe<M, ST>(self, matcher: M) -> ST
    where
        M: FnOnce(Result<Self::Transcript, VoiceError>) -> ST + Send + 'static,
        ST: futures_core::Stream<Item = fluent_voice_domain::TranscriptionSegmentImpl>
            + Send
            + Unpin
            + 'static,
    {
        // Build the transcript result synchronously, just like listen() does
        let stream = (self.stream_fn)(
            Some(SpeechSource::File {
                path: "".to_string(),
                format: fluent_voice_domain::AudioFormat::Pcm16Khz,
            }),
            self.vad_mode,
            self.noise_reduction,
            self.language_hint,
            self.diarization,
            self.word_timestamps,
            self.timestamps_granularity,
            self.punctuation,
        );

        let transcript_result = Ok(TranscriptImpl { stream });

        // Apply the matcher closure to the transcript result
        // The matcher contains the JSON syntax transformed by listen! macro
        matcher(transcript_result)
    }
}
