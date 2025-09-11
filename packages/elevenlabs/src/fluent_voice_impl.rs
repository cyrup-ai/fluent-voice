//! FluentVoice trait implementation for ElevenLabs
//!
//! This module provides the public API for ElevenLabs through the fluent-voice trait system.

use crate::engine::{
    AudioFormat, AudioOutput, AudioStream, AudioWithTimestamps, DictionaryBuilder, DictionaryInfo,
    FluentVoiceError as VoiceError, MicrophoneBuilder as InternalMicrophoneBuilder, Result,
    SttBuilder as InternalSttBuilder, TranscriptOutput, TranscriptStream,
    TranscriptionBuilder as InternalTranscriptionBuilder, TtsBuilder as InternalTtsBuilder,
    TtsConversation as InternalTtsConversation, TtsEngine, TtsEngineBuilder, Voice, VoiceBuilder,
    VoiceDetails, VoiceEditBuilder, VoiceSettings,
};
use fluent_voice::prelude::*;
use fluent_voice::stt_conversation::SttConversation;
use fluent_voice::tts_conversation::TtsConversation;
use futures_util::{Stream, StreamExt, stream};
use std::pin::Pin;

/// ElevenLabs implementation of the FluentVoice trait
pub struct ElevenLabsVoice;

impl FluentVoice for ElevenLabsVoice {
    fn tts() -> impl TtsConversationBuilder {
        ElevenLabsTtsConversationBuilder::new()
    }

    fn stt() -> impl SttConversationBuilder {
        ElevenLabsSttConversationBuilder::new()
    }

    fn wake_word() -> impl WakeWordBuilder {
        // ElevenLabs doesn't support wake word detection
        // Return the default Koffee implementation
        fluent_voice::wake_word_koffee::KoffeeWakeWordBuilder::new()
    }

    fn voices() -> impl VoiceDiscoveryBuilder {
        panic!("Voice discovery is not yet implemented for ElevenLabs")
    }

    fn clone_voice() -> impl VoiceCloneBuilder {
        panic!("Voice cloning is not yet implemented for ElevenLabs")
    }

    fn speech_to_speech() -> impl SpeechToSpeechBuilder {
        panic!("Speech-to-speech is not yet implemented for ElevenLabs")
    }

    fn audio_isolation() -> impl AudioIsolationBuilder {
        panic!("Audio isolation is not yet implemented for ElevenLabs")
    }

    fn sound_effects() -> impl SoundEffectsBuilder {
        panic!("Sound effects generation is not yet implemented for ElevenLabs")
    }
}

/// TTS conversation builder for ElevenLabs
pub struct ElevenLabsTtsConversationBuilder {
    speakers: Vec<SpeakerLine>,
    language: Option<Language>,
    engine: Option<TtsEngine>,
}

impl ElevenLabsTtsConversationBuilder {
    fn new() -> Self {
        Self {
            speakers: Vec::new(),
            language: None,
            engine: None,
        }
    }

    async fn get_or_create_engine(&mut self) -> Result<&TtsEngine> {
        if self.engine.is_none() {
            // Create engine with default configuration
            let engine = TtsEngineBuilder::default()
                .api_key_from_env()?
                .http3_enabled(true)
                .build()?;
            self.engine = Some(engine);
        }
        Ok(self.engine.as_ref().ok_or(VoiceError::Configuration("Engine not initialized".to_string()))?)
    }
}

impl TtsConversationBuilder for ElevenLabsTtsConversationBuilder {
    fn with_speaker(mut self, speaker: impl Into<SpeakerLine>) -> Self {
        self.speakers.push(speaker.into());
        self
    }

    fn language_hint(mut self, lang: Language) -> Self {
        self.language = Some(lang);
        self
    }

    async fn synthesize<F, R>(mut self, matcher: F) -> R
    where
        F: FnOnce(Result<TtsConversation, VoiceError>) -> R,
    {
        // Create the engine if not already created
        let engine_result = self.get_or_create_engine().await;

        match engine_result {
            Ok(engine) => {
                // Convert to TtsConversation
                let conversation = ElevenLabsTtsConversation {
                    speakers: self.speakers,
                    language: self.language,
                    engine: engine.clone(),
                };

                matcher(Ok(Box::new(conversation)))
            }
            Err(e) => matcher(Err(VoiceError::EngineError(e.to_string()))),
        }
    }
}

/// TTS conversation implementation for ElevenLabs
struct ElevenLabsTtsConversation {
    speakers: Vec<SpeakerLine>,
    language: Option<Language>,
    engine: TtsEngine,
}

impl TtsConversation for ElevenLabsTtsConversation {
    type AudioStream = Pin<Box<dyn Stream<Item = i16> + Send>>;

    fn into_stream(self) -> Self::AudioStream {
        // Create an async stream that processes all speakers
        let stream = stream::unfold(
            (self.speakers.into_iter(), self.engine, Vec::new()),
            |(mut speakers, engine, mut buffer)| async move {
                // If we have buffered samples, return them first
                if !buffer.is_empty() {
                    return Some((buffer.remove(0), (speakers, engine, buffer)));
                }

                // Process next speaker
                if let Some(speaker) = speakers.next() {
                    // Synthesize audio for this speaker
                    let result = engine
                        .tts()
                        .text(speaker.text())
                        .voice(speaker.voice_id().map(|v| v.0.as_str()).unwrap_or("Sarah"))
                        .generate()
                        .await;

                    match result {
                        Ok(audio_output) => {
                            // Convert audio bytes to i16 samples
                            // This is a simplified conversion - real implementation would
                            // properly decode the audio format
                            let bytes = audio_output.bytes();
                            let samples: Vec<i16> = bytes
                                .chunks_exact(2)
                                .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
                                .collect();

                            if let Some(first) = samples.first() {
                                let mut remaining = samples;
                                let first_sample = remaining.remove(0);
                                buffer = remaining;
                                Some((first_sample, (speakers, engine, buffer)))
                            } else {
                                None
                            }
                        }
                        Err(_) => {
                            // Skip this speaker on error and continue
                            None
                        }
                    }
                } else {
                    // No more speakers
                    None
                }
            },
        );

        Box::pin(stream)
    }
}

/// STT conversation builder for ElevenLabs
pub struct ElevenLabsSttConversationBuilder {
    source: Option<SpeechSource>,
    vad_mode: Option<VadMode>,
    noise_reduction: Option<NoiseReduction>,
    language: Option<Language>,
    diarization: Option<Diarization>,
    word_timestamps: Option<WordTimestamps>,
    timestamps_granularity: Option<TimestampsGranularity>,
    punctuation: Option<Punctuation>,
    engine: Option<TtsEngine>,
}

impl ElevenLabsSttConversationBuilder {
    fn new() -> Self {
        Self {
            source: None,
            vad_mode: None,
            noise_reduction: None,
            language: None,
            diarization: None,
            word_timestamps: None,
            timestamps_granularity: None,
            punctuation: None,
            engine: None,
        }
    }

    async fn get_or_create_engine(&mut self) -> Result<&TtsEngine> {
        if self.engine.is_none() {
            // Create engine with default configuration
            let engine = TtsEngineBuilder::default()
                .api_key_from_env()?
                .http3_enabled(true)
                .build()?;
            self.engine = Some(engine);
        }
        Ok(self.engine.as_ref().ok_or(VoiceError::Configuration("Engine not initialized".to_string()))?)
    }
}

impl SttConversationBuilder for ElevenLabsSttConversationBuilder {
    fn with_source(mut self, source: SpeechSource) -> Self {
        self.source = Some(source);
        self
    }

    fn vad_mode(mut self, mode: VadMode) -> Self {
        self.vad_mode = Some(mode);
        self
    }

    fn noise_reduction(mut self, reduction: NoiseReduction) -> Self {
        self.noise_reduction = Some(reduction);
        self
    }

    fn language_hint(mut self, lang: Language) -> Self {
        self.language = Some(lang);
        self
    }

    fn diarization(mut self, diarization: Diarization) -> Self {
        self.diarization = Some(diarization);
        self
    }

    fn word_timestamps(mut self, timestamps: WordTimestamps) -> Self {
        self.word_timestamps = Some(timestamps);
        self
    }

    fn timestamps_granularity(mut self, granularity: TimestampsGranularity) -> Self {
        self.timestamps_granularity = Some(granularity);
        self
    }

    fn punctuation(mut self, punctuation: Punctuation) -> Self {
        self.punctuation = Some(punctuation);
        self
    }

    fn with_microphone(mut self, backend: impl Into<String>) -> impl MicrophoneBuilder {
        self.source = Some(SpeechSource::Microphone {
            backend: MicBackend::Custom(backend.into()),
            format: AudioFormat::Pcm16Khz,
            sample_rate: 16_000,
        });
        ElevenLabsMicrophoneBuilder { builder: self }
    }

    fn transcribe(mut self, path: impl Into<String>) -> impl TranscriptionBuilder {
        self.source = Some(SpeechSource::File {
            path: path.into(),
            format: AudioFormat::Pcm16Khz,
        });
        ElevenLabsTranscriptionBuilder {
            builder: self,
            progress_template: None,
        }
    }

    async fn listen<F, R>(mut self, matcher: F) -> R
    where
        F: FnOnce(Result<SttConversation, VoiceError>) -> R,
    {
        // ElevenLabs doesn't support live microphone transcription
        matcher(Err(VoiceError::UnsupportedOperation(
            "Live microphone transcription is not supported by ElevenLabs".to_string(),
        )))
    }
}

/// Microphone builder for ElevenLabs (not supported)
pub struct ElevenLabsMicrophoneBuilder {
    builder: ElevenLabsSttConversationBuilder,
}

impl MicrophoneBuilder for ElevenLabsMicrophoneBuilder {
    async fn listen<F, R>(self, matcher: F) -> R
    where
        F: FnOnce(Result<SttConversation, VoiceError>) -> R,
    {
        // ElevenLabs doesn't support live microphone transcription
        matcher(Err(VoiceError::UnsupportedOperation(
            "Live microphone transcription is not supported by ElevenLabs".to_string(),
        )))
    }
}

/// Transcription builder for ElevenLabs
pub struct ElevenLabsTranscriptionBuilder {
    builder: ElevenLabsSttConversationBuilder,
    progress_template: Option<String>,
}

impl TranscriptionBuilder for ElevenLabsTranscriptionBuilder {
    type Transcript = ElevenLabsSttConversation;

    fn with_progress(mut self, template: impl Into<String>) -> Self {
        self.progress_template = Some(template.into());
        self
    }

    async fn emit<F, R>(mut self, matcher: F) -> R
    where
        F: FnOnce(Result<Transcript, VoiceError>) -> R,
    {
        // Get the engine
        let engine_result = self.builder.get_or_create_engine().await;

        match engine_result {
            Ok(engine) => {
                // Get the file path from source
                if let Some(SpeechSource::File { path, .. }) = &self.builder.source {
                    // Create STT builder
                    let mut stt_builder = engine.stt();

                    // Configure language if provided
                    if let Some(lang) = &self.builder.language {
                        stt_builder = stt_builder.language(&lang.0);
                    }

                    // Configure diarization
                    if let Some(Diarization::On) = self.builder.diarization {
                        stt_builder = stt_builder.diarization(true);
                    }

                    // Configure word timestamps
                    if let Some(WordTimestamps::On) = self.builder.word_timestamps {
                        stt_builder = stt_builder.with_word_timestamps();
                    }

                    // Transcribe the file
                    let result = stt_builder.transcribe(path.clone()).collect().await;

                    match result {
                        Ok(transcript_output) => {
                            // Convert to fluent-voice Transcript
                            let transcript = ElevenLabsTranscript {
                                output: transcript_output,
                            };

                            let stream = Box::pin(stream::iter(
                                transcript.output.words.into_iter().map(|word| {
                                    Ok(ElevenLabsTranscriptSegment {
                                        text: word.text.clone(),
                                        start_ms: word
                                            .start
                                            .map(|s| (s * 1000.0) as u32)
                                            .unwrap_or(0),
                                        end_ms: word.end.map(|e| (e * 1000.0) as u32).unwrap_or(0),
                                        speaker_id: word.speaker,
                                    })
                                }),
                            ));

                            matcher(Ok(ElevenLabsSttConversation { stream }))
                        }
                        Err(e) => matcher(Err(VoiceError::EngineError(e.to_string()))),
                    }
                } else {
                    matcher(Err(VoiceError::InvalidInput(
                        "No file path specified for transcription".to_string(),
                    )))
                }
            }
            Err(e) => matcher(Err(VoiceError::EngineError(e.to_string()))),
        }
    }
}

/// STT conversation implementation for ElevenLabs
struct ElevenLabsSttConversation {
    stream: Pin<Box<dyn Stream<Item = Result<ElevenLabsTranscriptSegment, VoiceError>> + Send>>,
}

impl SttConversation for ElevenLabsSttConversation {
    type Stream =
        Pin<Box<dyn Stream<Item = Result<ElevenLabsTranscriptSegment, VoiceError>> + Send>>;

    fn into_stream(self) -> Self::Stream {
        self.stream
    }
}

/// Transcript segment implementation for ElevenLabs
#[derive(Debug, Clone)]
struct ElevenLabsTranscriptSegment {
    text: String,
    start_ms: u32,
    end_ms: u32,
    speaker_id: Option<String>,
}

impl TranscriptSegment for ElevenLabsTranscriptSegment {
    fn start_ms(&self) -> u32 {
        self.start_ms
    }

    fn end_ms(&self) -> u32 {
        self.end_ms
    }

    fn text(&self) -> &str {
        &self.text
    }

    fn speaker_id(&self) -> Option<&str> {
        self.speaker_id.as_deref()
    }
}
