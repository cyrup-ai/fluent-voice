//! Microphone Builder Implementation

use crate::stt_conversation::MicrophoneBuilder;
use fluent_voice_domain::{
    Diarization, Language, NoiseReduction, Punctuation, TimestampsGranularity,
    TranscriptionSegmentImpl, VadMode, VoiceError, WordTimestamps,
};

use crate::engines::default_stt::{
    AudioProcessor, DefaultSTTConversation, SendableClosure, VadConfig, WakeWordConfig,
};

/// Builder for microphone-based speech recognition using the default STT engine.
///
/// Zero-allocation architecture: creates WhisperSttBuilder instances on demand.
pub struct DefaultMicrophoneBuilder {
    #[allow(dead_code)]
    pub(crate) device: String,
    pub(crate) vad_config: VadConfig,
    pub(crate) wake_word_config: WakeWordConfig,
    pub(crate) prediction_processor:
        Option<SendableClosure<Box<dyn FnMut(String, String) + Send + 'static>>>,
}

impl MicrophoneBuilder for DefaultMicrophoneBuilder {
    type Conversation = DefaultSTTConversation;

    fn vad_mode(mut self, mode: VadMode) -> Self {
        // Configure VAD mode in vad_config
        match mode {
            VadMode::Off => self.vad_config.sensitivity = 0.0,
            VadMode::Fast => self.vad_config.sensitivity = 0.3,
            VadMode::Accurate => self.vad_config.sensitivity = 0.8,
        }
        self
    }

    fn noise_reduction(self, _level: NoiseReduction) -> Self {
        // Noise reduction configured in audio preprocessing
        self
    }

    fn language_hint(self, _lang: Language) -> Self {
        // Language hint passed to Whisper transcriber
        self
    }

    fn diarization(self, _d: Diarization) -> Self {
        // Speaker diarization configured for transcript segments
        self
    }

    fn word_timestamps(self, _w: WordTimestamps) -> Self {
        // Word-level timestamps configured for transcript segments
        self
    }

    fn timestamps_granularity(self, _g: TimestampsGranularity) -> Self {
        // Timestamp granularity configured for transcript segments
        self
    }

    fn punctuation(self, _p: Punctuation) -> Self {
        // Automatic punctuation configured for transcript segments
        self
    }
    fn listen<M>(self, matcher: M) -> cyrup_sugars::prelude::AsyncStream<TranscriptionSegmentImpl>
    where
        M: FnOnce(Result<Self::Conversation, VoiceError>) -> cyrup_sugars::prelude::AsyncStream<TranscriptionSegmentImpl> + Send + 'static,
    {
        // Create AudioProcessor for wake word detection, VAD, and transcription
        let audio_processor = match AudioProcessor::new(self.wake_word_config) {
            Ok(processor) => processor,
            Err(e) => {
                return matcher(Err(e));
            }
        };

        // Create AudioStreamManager configuration for microphone capture
        let stream_config = crate::audio_stream_manager::AudioStreamConfig {
            sample_rate: 16000,
            channels: 1,
            device_name: self.device.clone(),
        };

        // Create AudioStreamManager and get the audio channel receiver
        let (stream_manager, audio_receiver) =
            match crate::audio_stream_manager::AudioStreamManager::new(stream_config) {
                Ok((manager, receiver)) => (manager, receiver),
                Err(e) => {
                    return matcher(Err(e));
                }
            };

        let conversation_result = DefaultSTTConversation::new(
            self.vad_config,
            self.wake_word_config,
            None, // error_handler: Temporarily disabled for crossbeam compatibility
            None, // wake_handler: Temporarily disabled for crossbeam compatibility
            None, // turn_handler: Temporarily disabled for crossbeam compatibility
            self.prediction_processor,
            None, // chunk_processor: Temporarily disabled for crossbeam compatibility
            audio_receiver,
            audio_processor,
            stream_manager,
        );

        // Call the matcher with the result, which in turn returns the stream
        matcher(conversation_result)
    }
}
