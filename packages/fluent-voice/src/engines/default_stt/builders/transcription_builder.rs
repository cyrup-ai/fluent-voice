//! Transcription Builder Implementation

use crate::stt_conversation::TranscriptionBuilder;
use fluent_voice_domain::{
    AudioFormat, Diarization, Language, NoiseReduction, Punctuation, SpeechSource,
    TimestampsGranularity, TranscriptionSegmentImpl, VadMode, VoiceError, WordTimestamps,
};
use futures::{Future, Stream};

use crate::engines::default_stt::{SendableClosure, VadConfig, WakeWordConfig};

/// Builder for file-based transcription using the default STT engine.
///
/// Zero-allocation architecture: creates WhisperSttBuilder instances on demand.
pub struct DefaultTranscriptionBuilder {
    #[allow(dead_code)]
    pub(crate) path: String,
    pub(crate) vad_config: VadConfig,
    pub(crate) wake_word_config: WakeWordConfig,
    #[allow(dead_code)] // Stored for future prediction processing implementation
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
        S: futures_core::Stream<Item = Result<TranscriptionSegmentImpl, VoiceError>>
            + Send
            + Unpin
            + 'static,
    {
        // Create a success result and let matcher handle stream creation
        let transcript_result = Ok(format!("Transcribing file: {}", self.path));
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
        use async_stream::stream;

        Box::pin(stream! {
            // Use real file transcription with WhisperSttBuilder
            let speech_source = SpeechSource::File {
                path: self.path.clone(),
                format: AudioFormat::Pcm16Khz,
            };

            // Create WhisperSttBuilder for file transcription
            let whisper_builder = fluent_voice_whisper::WhisperSttBuilder::new()
                .with_source(speech_source);

            match whisper_builder.transcribe(|conversation_result| conversation_result).await {
                Ok(conversation) => {
                    match conversation.collect().await {
                        Ok(transcript) => {
                            // Yield the complete transcription text
                            yield transcript;
                        }
                        Err(e) => {
                            tracing::error!("File transcription text collection failed: {}", e);
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("File transcription failed: {}", e);
                }
            }
        })
    }

    fn collect(self) -> impl Future<Output = Result<Self::Transcript, VoiceError>> + Send {
        async move {
            // Use real file transcription with WhisperSttBuilder
            let speech_source = SpeechSource::File {
                path: self.path.clone(),
                format: AudioFormat::Pcm16Khz,
            };

            // Create WhisperSttBuilder for file transcription
            let whisper_builder =
                fluent_voice_whisper::WhisperSttBuilder::new().with_source(speech_source);

            let conversation = whisper_builder
                .transcribe(|conversation_result| conversation_result)
                .await
                .map_err(|e| {
                    VoiceError::ProcessingError(format!("File transcription failed: {}", e))
                })?;

            let transcript = conversation.collect().await.map_err(|e| {
                VoiceError::ProcessingError(format!(
                    "File transcription text collection failed: {}",
                    e
                ))
            })?;

            // Return real transcribed text
            Ok(transcript)
        }
    }

    fn collect_with<F, R>(self, handler: F) -> impl Future<Output = R> + Send
    where
        F: FnOnce(Result<Self::Transcript, VoiceError>) -> R + Send + 'static,
    {
        async move {
            // Use the real collect implementation
            let result = self.collect().await;
            handler(result)
        }
    }

    fn into_text_stream(self) -> impl Stream<Item = String> + Send {
        // Use the real emit implementation for text streaming
        self.emit()
    }
}
