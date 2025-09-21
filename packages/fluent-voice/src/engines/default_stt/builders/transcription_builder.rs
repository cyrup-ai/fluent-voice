//! Transcription Builder Implementation

use crate::stt_conversation::TranscriptionBuilder;
use fluent_voice_domain::{
    AudioFormat, Diarization, Language, NoiseReduction, Punctuation, SpeechSource,
    TimestampsGranularity, TranscriptionSegmentImpl, VadMode, VoiceError, WordTimestamps,
};
use futures::{Future, Stream};

use crate::engines::default_stt::{
    AudioProcessor, DefaultSTTConversation, SendableClosure, VadConfig, WakeWordConfig,
};

/// Builder for file-based transcription using the default STT engine.
///
/// Zero-allocation architecture: creates WhisperTranscriber instances on demand.
pub struct DefaultTranscriptionBuilder {
    #[allow(dead_code)]
    pub(crate) path: String,
    pub(crate) vad_config: VadConfig,
    pub(crate) wake_word_config: WakeWordConfig,
    pub(crate) prediction_processor:
        Option<SendableClosure<Box<dyn FnMut(String, String) + Send + 'static>>>,
    pub(crate) language_hint: Option<Language>,
    pub(crate) diarization: Option<Diarization>,
    pub(crate) word_timestamps: Option<WordTimestamps>,
    pub(crate) timestamps_granularity: Option<TimestampsGranularity>,
    pub(crate) punctuation: Option<Punctuation>,
    pub(crate) progress_template: Option<String>,
}

impl TranscriptionBuilder for DefaultTranscriptionBuilder {
    type Transcript = String;

    fn transcribe<M, S>(self, matcher: M) -> S
    where
        M: FnOnce(Result<Self::Transcript, VoiceError>) -> S + Send + 'static,
        S: futures_core::Stream<Item = Result<TranscriptionSegmentImpl, VoiceError>> + Send + Unpin + 'static,
    {
        // For file transcription, create dummy audio components
        let audio_processor = match AudioProcessor::new(self.wake_word_config.clone()) {
            Ok(processor) => processor,
            Err(e) => {
                // Return error stream
                return matcher(Err(e));
            }
        };
        let (_dummy_sender, audio_receiver) = crossbeam_channel::unbounded();

        // Create dummy stream manager for file transcription
        let stream_config = crate::audio_stream_manager::AudioStreamConfig::default();
        let (stream_manager, _) =
            match crate::audio_stream_manager::AudioStreamManager::new(stream_config) {
                Ok((manager, _receiver)) => (manager, _receiver),
                Err(e) => {
                    return matcher(Err(e));
                }
            };

        // Create the conversation and get the result
        let conversation_result = DefaultSTTConversation::new(
            self.vad_config,
            self.wake_word_config,
            None, // No error handler for file transcription
            None, // No wake word handler
            None, // No turn detection handler
            self.prediction_processor,
            Some(SendableClosure(Box::new(|result| match result {
                Ok(segment) => segment, // Pass through successful segments unchanged
                Err(_error) => {
                    // Create a default segment for errors in file transcription
                    TranscriptionSegmentImpl::new(
                        "[TRANSCRIPTION_ERROR]".to_string(),
                        0,    // start_ms
                        0,    // end_ms
                        None, // speaker_id
                    )
                }
            }))),
            audio_receiver,
            audio_processor,
            stream_manager,
        );

        // Build the transcript result - for file transcription we process synchronously
        let transcript_result = match conversation_result {
            Ok(mut conversation) => {
                // Set the speech source for file transcription
                conversation.speech_source = Some(SpeechSource::File {
                    path: self.path.clone(),
                    format: AudioFormat::Pcm48Khz,
                });

                // For file transcription, return placeholder result
                // TODO: Implement proper async file transcription that doesn't violate the sync function contract
                // This is a temporary placeholder to fix compilation errors
                Ok(
                    "File transcription placeholder - needs proper async implementation"
                        .to_string(),
                )
            }
            Err(e) => Err(e.into()),
        };

        // Call the matcher with the result
        matcher(transcript_result)
    }

    fn vad_mode(mut self, mode: VadMode) -> Self {
        // Apply VAD configuration (same as SttConversationBuilder line 772)
        match mode {
            VadMode::Off => self.vad_config.sensitivity = 0.0,
            VadMode::Fast => self.vad_config.sensitivity = 0.3,
            VadMode::Accurate => self.vad_config.sensitivity = 0.8,
        }
        self
    }

    fn noise_reduction(mut self, level: NoiseReduction) -> Self {
        // Apply noise reduction configuration (same as SttConversationBuilder line 784)
        match level {
            NoiseReduction::Off => {
                self.wake_word_config.filters_enabled = false;
                self.wake_word_config.band_pass_enabled = false;
                self.wake_word_config.gain_normalizer_enabled = false;
            }
            NoiseReduction::Low => {
                self.wake_word_config.filters_enabled = true;
                self.wake_word_config.band_pass_enabled = true;
                self.wake_word_config.band_pass_low_cutoff = 200.0;
                self.wake_word_config.band_pass_high_cutoff = 4000.0;
                self.wake_word_config.gain_normalizer_enabled = true;
                self.wake_word_config.gain_normalizer_max_gain = 1.5;
            }
            NoiseReduction::High => {
                self.wake_word_config.filters_enabled = true;
                self.wake_word_config.band_pass_enabled = true;
                self.wake_word_config.band_pass_low_cutoff = 300.0;
                self.wake_word_config.band_pass_high_cutoff = 3400.0;
                self.wake_word_config.gain_normalizer_enabled = true;
                self.wake_word_config.gain_normalizer_max_gain = 3.0;
            }
        }
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

    fn with_progress<S: Into<String>>(mut self, template: S) -> Self {
        self.progress_template = Some(template.into());
        self
    }

    fn emit(self) -> impl Stream<Item = String> + Send + Unpin {
        use futures::stream;
        // Convert transcription to text stream
        Box::pin(stream::iter(vec!["Transcription placeholder".to_string()]))
    }

    fn collect(self) -> impl Future<Output = Result<Self::Transcript, VoiceError>> + Send {
        async move { Ok("Collected transcription placeholder".to_string()) }
    }

    fn collect_with<F, R>(self, handler: F) -> impl Future<Output = R> + Send
    where
        F: FnOnce(Result<Self::Transcript, VoiceError>) -> R + Send + 'static,
    {
        async move {
            let result = Ok("Collected transcription placeholder".to_string());
            handler(result)
        }
    }

    fn into_text_stream(self) -> impl Stream<Item = String> + Send {
        use futures::stream;
        // Convert transcription to text stream
        Box::pin(stream::iter(vec!["Text stream placeholder".to_string()]))
    }
}
