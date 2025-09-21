//! STT Conversation Implementation
//!
//! Complete STT conversation that handles the full pipeline:
//! microphone input -> wake word detection -> VAD turn detection -> Whisper transcription

use fluent_voice_domain::{TranscriptionSegmentImpl, VadMode, VoiceError};

use super::audio_processor::AudioProcessor;
use super::config::{VadConfig, WakeWordConfig};
use super::types::{DefaultTranscriptStream, SendableClosure};
use crate::stt_conversation::SttConversation;

/// Complete STT conversation that handles the full pipeline:
/// microphone input -> wake word detection -> VAD turn detection -> Whisper transcription
///
/// Zero-allocation, no-locking architecture: creates new WhisperTranscriber instances
/// per transcription for optimal performance and thread safety.
pub struct DefaultSTTConversation {
    pub(crate) vad_config: VadConfig,
    pub wake_word_config: WakeWordConfig, // Made public for audio processor configuration
    pub(crate) speech_source: Option<fluent_voice_domain::SpeechSource>,
    // Event handlers that get called during processing
    pub(crate) error_handler:
        Option<SendableClosure<Box<dyn FnMut(VoiceError) -> String + Send + 'static>>>,
    pub(crate) wake_handler: Option<SendableClosure<Box<dyn FnMut(String) + Send + 'static>>>,
    pub(crate) turn_handler:
        Option<SendableClosure<Box<dyn FnMut(Option<String>, String) + Send + 'static>>>,
    pub(crate) prediction_processor:
        Option<SendableClosure<Box<dyn FnMut(String, String) + Send + 'static>>>,
    // CRITICAL: The chunk processor that transforms transcription results
    pub(crate) chunk_processor: Option<
        SendableClosure<
            Box<
                dyn FnMut(Result<TranscriptionSegmentImpl, VoiceError>) -> TranscriptionSegmentImpl
                    + Send
                    + 'static,
            >,
        >,
    >,
    // Crossbeam channel receiver for audio data from AudioStreamManager
    pub(crate) audio_receiver: crossbeam_channel::Receiver<Vec<f32>>,
    // AudioProcessor for wake word, VAD, and transcription processing
    pub(crate) audio_processor: AudioProcessor,
    // Stream manager handle for cleanup
    pub(crate) stream_manager: crate::audio_stream_manager::AudioStreamManager,
}
impl DefaultSTTConversation {
    pub fn new(
        vad_config: VadConfig,
        wake_word_config: WakeWordConfig,
        error_handler: Option<
            SendableClosure<Box<dyn FnMut(VoiceError) -> String + Send + 'static>>,
        >,
        wake_handler: Option<SendableClosure<Box<dyn FnMut(String) + Send + 'static>>>,
        turn_handler: Option<
            SendableClosure<Box<dyn FnMut(Option<String>, String) + Send + 'static>>,
        >,
        prediction_processor: Option<
            SendableClosure<Box<dyn FnMut(String, String) + Send + 'static>>,
        >,
        chunk_processor: Option<
            SendableClosure<
                Box<
                    dyn FnMut(
                            Result<TranscriptionSegmentImpl, VoiceError>,
                        ) -> TranscriptionSegmentImpl
                        + Send
                        + 'static,
                >,
            >,
        >,
        audio_receiver: crossbeam_channel::Receiver<Vec<f32>>,
        audio_processor: AudioProcessor,
        stream_manager: crate::audio_stream_manager::AudioStreamManager,
    ) -> Result<Self, VoiceError> {
        Ok(Self {
            vad_config,
            wake_word_config,
            speech_source: None,
            error_handler,
            wake_handler,
            turn_handler,
            prediction_processor,
            chunk_processor,
            audio_receiver,
            audio_processor,
            stream_manager,
        })
    }
}
impl SttConversation for DefaultSTTConversation {
    type Stream = DefaultTranscriptStream;

    fn into_stream(self) -> Self::Stream {
        // Ensure the struct is Send for async stream
        fn assert_send<T: Send>(_t: &T) {}
        assert_send(&self);

        tracing::info!("Starting STT conversation with channel-based audio stream");

        // Use futures::stream to handle the async function properly
        use futures::stream::{self, StreamExt};

        Box::pin(
            stream::once(async move { super::stream_impl::process_audio_loop(self).await })
                .flatten(),
        )
    }
}

impl DefaultSTTConversation {
    /// Create a new conversation from a builder configuration
    pub fn new_from_builder(
        builder: super::builders::DefaultSTTConversationBuilder,
    ) -> Result<Self, VoiceError> {
        // Use default configurations based on builder settings
        let vad_config = super::config::VadConfig {
            sensitivity: match builder.vad_mode.unwrap_or(VadMode::Accurate) {
                VadMode::Off => 0.0,
                VadMode::Fast => 0.3,
                VadMode::Accurate => 0.8,
            },
            min_speech_duration: 300,
            max_silence_duration: 1000,
            simd_level: 2,
        };

        let wake_word_config = super::config::WakeWordConfig::default();

        // Create audio processor
        let audio_processor = super::audio_processor::AudioProcessor::new(wake_word_config)?;

        // Create crossbeam channel for audio communication
        let (_audio_sender, audio_receiver) = crossbeam_channel::unbounded();

        // Create stream manager
        let stream_config = crate::audio_stream_manager::AudioStreamConfig::default();
        let (stream_manager, _) =
            crate::audio_stream_manager::AudioStreamManager::new(stream_config)?;

        // Create conversation using the builder's actual configuration
        Self::new(
            vad_config,
            wake_word_config,
            builder.error_handler,
            builder.wake_handler,
            builder.turn_handler,
            builder.prediction_processor,
            builder.chunk_handler,
            audio_receiver,
            audio_processor,
            stream_manager,
        )
    }

    /// Get VAD configuration for this conversation
    pub fn vad_config(&self) -> &VadConfig {
        &self.vad_config
    }

    /// Get a reference to the audio stream manager for monitoring stream health
    pub fn stream_manager(&self) -> &crate::audio_stream_manager::AudioStreamManager {
        // Stream manager is kept alive for the lifetime of this conversation
        // and handles cleanup automatically via Drop
        &self.stream_manager
    }
}

/// Implement Drop for DefaultSTTConversation to ensure proper cleanup
impl Drop for DefaultSTTConversation {
    fn drop(&mut self) {
        // AudioStreamManager implements Drop automatically for proper cleanup
        // This ensures the CPAL stream is stopped and resources are freed
        tracing::debug!("DefaultSTTConversation dropped, AudioStreamManager will handle cleanup");
    }
}
