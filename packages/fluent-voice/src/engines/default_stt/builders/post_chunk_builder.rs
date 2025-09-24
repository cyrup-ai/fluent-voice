//! Post-chunk builder implementation

use crate::stt_conversation::{MicrophoneBuilder, SttPostChunkBuilder, TranscriptionBuilder};
use fluent_voice_domain::{TranscriptionSegmentImpl, VoiceError};

use super::{DefaultMicrophoneBuilder, DefaultTranscriptionBuilder};
use crate::engines::default_stt::{
    AudioProcessor, DefaultSTTConversation, DefaultSTTConversationBuilder, SendableClosure,
};

/// Post-chunk builder that provides access to action methods.
pub struct DefaultSTTPostChunkBuilder {
    pub(crate) inner: DefaultSTTConversationBuilder,
    pub(crate) chunk_processor: Box<
        dyn FnMut(Result<TranscriptionSegmentImpl, VoiceError>) -> TranscriptionSegmentImpl + Send,
    >,
}

impl SttPostChunkBuilder for DefaultSTTPostChunkBuilder {
    type Conversation = DefaultSTTConversation;

    fn with_microphone(self, device: impl Into<String>) -> impl MicrophoneBuilder {
        DefaultMicrophoneBuilder {
            device: device.into(),
            vad_config: self.inner.vad_config,
            wake_word_config: self.inner.wake_word_config,
            prediction_processor: self.inner.prediction_processor,
        }
    }

    fn transcribe(self, path: impl Into<String>) -> impl TranscriptionBuilder {
        DefaultTranscriptionBuilder {
            path: path.into(),
            vad_config: self.inner.vad_config,
            wake_word_config: self.inner.wake_word_config,
            prediction_processor: self.inner.prediction_processor,
            language_hint: None,
            diarization: None,
            word_timestamps: None,
            timestamps_granularity: None,
            punctuation: None,
            progress_template: None,
        }
    }
    fn listen<M>(self, matcher: M) -> cyrup_sugars::prelude::AsyncStream<TranscriptionSegmentImpl>
    where
        M: FnOnce(Result<DefaultSTTConversation, VoiceError>) -> cyrup_sugars::prelude::AsyncStream<TranscriptionSegmentImpl> + Send + 'static,
    {
        // Create AudioProcessor for wake word detection, VAD, and transcription
        let audio_processor = match AudioProcessor::new(self.inner.wake_word_config) {
            Ok(processor) => processor,
            Err(e) => {
                return matcher(Err(e));
            }
        };

        // Create AudioStreamManager configuration for microphone capture
        let stream_config = crate::audio_stream_manager::AudioStreamConfig {
            sample_rate: 16000,
            channels: 1,
            device_name: "default".to_string(), // Use default device for PostChunkBuilder
        };

        // Create AudioStreamManager and get the audio channel receiver
        let (stream_manager, audio_receiver) =
            match crate::audio_stream_manager::AudioStreamManager::new(stream_config) {
                Ok((manager, receiver)) => (manager, receiver),
                Err(e) => {
                    return matcher(Err(e));
                }
            };

        // Create the conversation result with the chunk processor and new components
        let mut chunk_processor = self.chunk_processor;
        let conversation_result = DefaultSTTConversation::new(
            self.inner.vad_config,
            self.inner.wake_word_config,
            self.inner.error_handler,
            self.inner.wake_handler,
            self.inner.turn_handler,
            self.inner.prediction_processor,
            Some(SendableClosure(Box::new(move |result| {
                chunk_processor(result)
            }))), // Use the stored chunk processor
            audio_receiver,
            audio_processor,
            stream_manager,
        );

        // Call the matcher with the result
        matcher(conversation_result)
    }
}
