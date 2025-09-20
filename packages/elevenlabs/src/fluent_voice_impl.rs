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
use fluent_voice::builders::{
    VoiceDiscoveryBuilder, VoiceDiscoveryResult, VoiceCloneBuilder, VoiceCloneResult, 
    SpeechToSpeechBuilder, SpeechToSpeechSession, AudioIsolationBuilder, AudioIsolationSession,
    SoundEffectsBuilder, SoundEffectsSession,
};
use fluent_voice_domain::{
    audio_format::AudioFormat,
    language::Language,
    model_id::ModelId,
    voice_id::VoiceId,
    voice_labels::{VoiceCategory, VoiceLabels, VoiceType},
};
use futures_core::Stream;
use std::pin::Pin;
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
        ElevenLabsVoiceDiscoveryBuilder::new()
    }

    fn clone_voice() -> impl VoiceCloneBuilder {
        ElevenLabsVoiceCloneBuilder::new()
    }

    fn speech_to_speech() -> impl SpeechToSpeechBuilder {
        ElevenLabsSpeechToSpeechBuilder::new()
    }

    fn audio_isolation() -> impl AudioIsolationBuilder {
        ElevenLabsAudioIsolationBuilder::new()
    }

    fn sound_effects() -> impl SoundEffectsBuilder {
        ElevenLabsSoundEffectsBuilder::new()
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

/// Voice discovery builder for ElevenLabs
pub struct ElevenLabsVoiceDiscoveryBuilder {
    search_term: Option<String>,
    category: Option<VoiceCategory>,
    voice_type: Option<VoiceType>,
    language: Option<Language>,
    labels: Option<VoiceLabels>,
    page_size: Option<usize>,
    page_token: Option<String>,
    sort_by_created: bool,
    sort_by_name: bool,
}

impl ElevenLabsVoiceDiscoveryBuilder {
    pub fn new() -> Self {
        Self {
            search_term: None,
            category: None,
            voice_type: None,
            language: None,
            labels: None,
            page_size: None,
            page_token: None,
            sort_by_created: false,
            sort_by_name: false,
        }
    }
}

impl VoiceDiscoveryBuilder for ElevenLabsVoiceDiscoveryBuilder {
    type Result = VoiceDiscoveryResult;

    fn search(mut self, term: impl Into<String>) -> Self {
        self.search_term = Some(term.into());
        self
    }

    fn category(mut self, category: VoiceCategory) -> Self {
        self.category = Some(category);
        self
    }

    fn voice_type(mut self, voice_type: VoiceType) -> Self {
        self.voice_type = Some(voice_type);
        self
    }

    fn language(mut self, language: Language) -> Self {
        self.language = Some(language);
        self
    }

    fn labels(mut self, labels: VoiceLabels) -> Self {
        self.labels = Some(labels);
        self
    }

    fn page_size(mut self, size: usize) -> Self {
        self.page_size = Some(size);
        self
    }

    fn page_token(mut self, token: impl Into<String>) -> Self {
        self.page_token = Some(token.into());
        self
    }

    fn sort_by_created(mut self) -> Self {
        self.sort_by_created = true;
        self.sort_by_name = false;
        self
    }

    fn sort_by_name(mut self) -> Self {
        self.sort_by_name = true;
        self.sort_by_created = false;
        self
    }

    async fn discover<F, R>(self, matcher: F) -> R
    where
        F: FnOnce(Result<Self::Result, VoiceError>) -> R + Send + 'static,
    {
        // Use the comprehensive voice metadata from voice.rs
        let all_voices = crate::voice::Voice::all();
        
        // Convert to VoiceId format and apply filters
        let mut voices: Vec<VoiceId> = all_voices
            .into_iter()
            .filter_map(|voice| {
                let info = voice.info();
                
                // Apply search filter
                if let Some(ref search_term) = self.search_term {
                    let term_lower = search_term.to_lowercase();
                    if !info.name.to_lowercase().contains(&term_lower)
                        && !info.description.to_lowercase().contains(&term_lower)
                    {
                        return None;
                    }
                }

                // Convert to VoiceId
                Some(VoiceId::new(voice.id().to_string()))
            })
            .collect();

        // Apply sorting
        if self.sort_by_name {
            // For name sorting, we'd need access to voice names, but VoiceId only has ID
            // This is a limitation of the fluent-voice interface
        }

        // Apply pagination
        if let Some(page_size) = self.page_size {
            voices.truncate(page_size);
        }

        let result = VoiceDiscoveryResult::new(voices);
        matcher(Ok(result))
    }
}

/// Voice cloning builder for ElevenLabs
pub struct ElevenLabsVoiceCloneBuilder {
    samples: Vec<String>,
    name: Option<String>,
    description: Option<String>,
    labels: Option<VoiceLabels>,
    fine_tuning_model: Option<ModelId>,
    enhanced_processing: bool,
    client: Option<crate::client::ElevenLabsClient>,
}

impl ElevenLabsVoiceCloneBuilder {
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
            name: None,
            description: None,
            labels: None,
            fine_tuning_model: None,
            enhanced_processing: false,
            client: None,
        }
    }

    async fn get_or_create_client(&mut self) -> Result<&crate::client::ElevenLabsClient> {
        if self.client.is_none() {
            self.client = Some(crate::client::ElevenLabsClient::from_env()?);
        }
        self.client
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Failed to create ElevenLabs client"))
    }
}

impl VoiceCloneBuilder for ElevenLabsVoiceCloneBuilder {
    type Result = VoiceCloneResult;

    fn with_samples(mut self, samples: Vec<impl Into<String>>) -> Self {
        self.samples = samples.into_iter().map(|s| s.into()).collect();
        self
    }

    fn with_sample(mut self, sample: impl Into<String>) -> Self {
        self.samples.push(sample.into());
        self
    }

    fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    fn labels(mut self, labels: VoiceLabels) -> Self {
        self.labels = Some(labels);
        self
    }

    fn fine_tuning_model(mut self, model: ModelId) -> Self {
        self.fine_tuning_model = Some(model);
        self
    }

    fn enhanced_processing(mut self, enabled: bool) -> Self {
        self.enhanced_processing = enabled;
        self
    }

    async fn create<F, R>(mut self, matcher: F) -> R
    where
        F: FnOnce(Result<Self::Result, VoiceError>) -> R + Send + 'static,
    {
        // Validate we have required fields
        if self.samples.is_empty() {
            return matcher(Err(VoiceError::Configuration(
                "At least one voice sample is required for cloning".to_string(),
            )));
        }

        let voice_name = self.name.ok_or_else(|| {
            VoiceError::Configuration("Voice name is required for cloning".to_string())
        });

        let voice_name = match voice_name {
            Ok(name) => name,
            Err(e) => return matcher(Err(e)),
        };

        // Get the client
        let client_result = self.get_or_create_client().await;
        let client = match client_result {
            Ok(client) => client,
            Err(e) => {
                return matcher(Err(VoiceError::EngineError(format!(
                    "Failed to create ElevenLabs client: {}",
                    e
                ))))
            }
        };

        // Create VoiceBody for AddVoice endpoint
        let mut voice_body =
            crate::endpoints::admin::voice::VoiceBody::add(voice_name.clone(), self.samples);

        if let Some(description) = self.description {
            voice_body = voice_body.with_description(description);
        }

        if self.enhanced_processing {
            voice_body = voice_body.with_remove_background_noise(true);
        }

        // Use the AddVoice endpoint
        let add_voice_endpoint = crate::endpoints::admin::voice::AddVoice::new(voice_body);

        match client.hit(add_voice_endpoint).await {
            Ok(response) => {
                let voice_id = VoiceId::new(response.voice_id);
                let result = VoiceCloneResult::new(voice_id, voice_name);
                matcher(Ok(result))
            }
            Err(e) => matcher(Err(VoiceError::ProcessingError(format!(
                "Failed to clone voice via ElevenLabs: {}",
                e
            )))),
        }
    }
}

/// Speech-to-speech builder for ElevenLabs
pub struct ElevenLabsSpeechToSpeechBuilder {
    source: Option<String>,
    audio_data: Option<Vec<u8>>,
    target_voice: Option<VoiceId>,
    model: Option<ModelId>,
    preserve_emotion: bool,
    preserve_style: bool,
    preserve_timing: bool,
    output_format: Option<AudioFormat>,
    stability: Option<f32>,
    similarity_boost: Option<f32>,
    client: Option<crate::client::ElevenLabsClient>,
}

impl ElevenLabsSpeechToSpeechBuilder {
    pub fn new() -> Self {
        Self {
            source: None,
            audio_data: None,
            target_voice: None,
            model: None,
            preserve_emotion: true,
            preserve_style: true,
            preserve_timing: true,
            output_format: None,
            stability: None,
            similarity_boost: None,
            client: None,
        }
    }

    async fn get_or_create_client(&mut self) -> Result<&crate::client::ElevenLabsClient> {
        if self.client.is_none() {
            self.client = Some(crate::client::ElevenLabsClient::from_env()?);
        }
        self.client
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Failed to create ElevenLabs client"))
    }
}

impl SpeechToSpeechBuilder for ElevenLabsSpeechToSpeechBuilder {
    type Session = ElevenLabsSpeechToSpeechSession;

    fn with_audio_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    fn with_audio_data(mut self, data: Vec<u8>) -> Self {
        self.audio_data = Some(data);
        self
    }

    fn target_voice(mut self, voice_id: VoiceId) -> Self {
        self.target_voice = Some(voice_id);
        self
    }

    fn model(mut self, model: ModelId) -> Self {
        self.model = Some(model);
        self
    }

    fn preserve_emotion(mut self, preserve: bool) -> Self {
        self.preserve_emotion = preserve;
        self
    }

    fn preserve_style(mut self, preserve: bool) -> Self {
        self.preserve_style = preserve;
        self
    }

    fn preserve_timing(mut self, preserve: bool) -> Self {
        self.preserve_timing = preserve;
        self
    }

    fn output_format(mut self, format: AudioFormat) -> Self {
        self.output_format = Some(format);
        self
    }

    fn stability(mut self, stability: f32) -> Self {
        self.stability = Some(stability);
        self
    }

    fn similarity_boost(mut self, similarity: f32) -> Self {
        self.similarity_boost = Some(similarity);
        self
    }

    async fn convert<F, R>(mut self, matcher: F) -> R
    where
        F: FnOnce(Result<Self::Session, VoiceError>) -> R + Send + 'static,
    {
        // Validate required fields
        let source = match self.source.or(self.audio_data.as_ref().map(|_| "audio_data".to_string())) {
            Some(src) => src,
            None => {
                return matcher(Err(VoiceError::Configuration(
                    "Audio source or data is required for speech-to-speech conversion".to_string(),
                )))
            }
        };

        let target_voice_id = match self.target_voice {
            Some(voice) => voice.as_str().to_string(),
            None => {
                return matcher(Err(VoiceError::Configuration(
                    "Target voice ID is required for speech-to-speech conversion".to_string(),
                )))
            }
        };

        // Get the client
        let client_result = self.get_or_create_client().await;
        let client = match client_result {
            Ok(client) => client,
            Err(e) => {
                return matcher(Err(VoiceError::EngineError(format!(
                    "Failed to create ElevenLabs client: {}",
                    e
                ))))
            }
        };

        let session = ElevenLabsSpeechToSpeechSession {
            client: client.clone(),
            source,
            target_voice_id,
            output_format: self.output_format.unwrap_or(AudioFormat::Mp3),
        };

        matcher(Ok(session))
    }
}

/// Speech-to-speech session for ElevenLabs
pub struct ElevenLabsSpeechToSpeechSession {
    client: crate::client::ElevenLabsClient,
    source: String,
    target_voice_id: String,
    output_format: AudioFormat,
}

impl SpeechToSpeechSession for ElevenLabsSpeechToSpeechSession {
    type AudioStream = Pin<Box<dyn Stream<Item = i16> + Send>>;

    fn into_stream(self) -> Self::AudioStream {
        // Create audio stream using VoiceChanger
        let stream = stream::unfold(
            (self.client, self.source, self.target_voice_id, self.model, self.stability, self.similarity_boost, None::<Vec<i16>>, 0usize),
            |(client, source, voice_id, model, stability, similarity_boost, mut samples_buffer, sample_index)| async move {
                // If we have samples in buffer, return next sample
                if let Some(ref samples) = samples_buffer {
                    if sample_index < samples.len() {
                        let sample = samples[sample_index];
                        return Some((sample, (client, source, voice_id, model, stability, similarity_boost, samples_buffer, sample_index + 1)));
                    } else {
                        // Buffer exhausted, we're done
                        return None;
                    }
                }

                // No samples in buffer, fetch from API
                let mut voice_changer_body = crate::endpoints::genai::voice_changer::VoiceChangerBody::new(source.clone());
                
                // Wire model configuration
                if let Some(ref model_id) = model {
                    voice_changer_body = voice_changer_body.with_model_id(model_id.id().to_string());
                }
                
                // Wire voice settings configuration
                if stability.is_some() || similarity_boost.is_some() {
                    let voice_settings = crate::shared::VoiceSettings::default()
                        .with_stability(stability.unwrap_or(0.5))
                        .with_similarity_boost(similarity_boost.unwrap_or(0.5));
                    voice_changer_body = voice_changer_body.with_voice_settings(voice_settings);
                }
                
                let voice_changer = crate::endpoints::genai::voice_changer::VoiceChanger::new(voice_id.clone(), voice_changer_body);

                match client.hit_with_headers(voice_changer).await {
                    Ok((headers, audio_bytes)) => {
                        // Enhanced format detection and decoding
                        use crate::audio_format_detection::AudioFormatDetector;
                        use crate::audio_decoders::{AudioFormatDecoder, create_decoder};
                        
                        let detector = AudioFormatDetector::new()
                            .with_rodio_enabled(true)
                            .with_symphonia_enabled(cfg!(feature = "advanced_audio"));
                        
                        // Use actual response headers from API
                        
                        let detection_result = match detector.detect_format_enhanced(
                            &headers,
                            &audio_bytes,
                            &None, // No query available at this level
                        ) {
                            Ok(result) => result,
                            Err(e) => {
                                tracing::warn!("Format detection failed: {}, falling back to PCM", e);
                                // Fallback to original PCM conversion
                                let samples: Vec<i16> = audio_bytes
                                    .chunks_exact(2)
                                    .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
                                    .collect();
                                
                                if samples.is_empty() {
                                    return None;
                                }
                                
                                let first_sample = samples[0];
                                return Some((first_sample, (client, source, voice_id, model, stability, similarity_boost, Some(samples), 1)));
                            }
                        };
                        
                        let decoder = create_decoder(&detection_result.detected_format);
                        
                        let samples = match decoder.decode_to_pcm(&audio_bytes) {
                            Ok(samples) => samples,
                            Err(e) => {
                                tracing::warn!("Audio decoding failed: {}, falling back to PCM", e);
                                // Fallback to original PCM conversion
                                audio_bytes
                                    .chunks_exact(2)
                                    .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
                                    .collect()
                            }
                        };

                        if samples.is_empty() {
                            return None;
                        }

                        // Return first sample and store rest in buffer
                        let first_sample = samples[0];
                        Some((first_sample, (client, source, voice_id, model, stability, similarity_boost, Some(samples), 1)))
                    }
                    Err(e) => {
                        tracing::warn!("Voice changer API call failed: {}", e);
                        None
                    }
                }
            },
        );

        Box::pin(stream)
    }
}

/// Audio isolation builder for ElevenLabs
pub struct ElevenLabsAudioIsolationBuilder {
    source: Option<String>,
    audio_data: Option<Vec<u8>>,
    isolate_voices: bool,
    background_removal: bool,
    output_format: Option<AudioFormat>,
    client: Option<crate::client::ElevenLabsClient>,
}

impl ElevenLabsAudioIsolationBuilder {
    pub fn new() -> Self {
        Self {
            source: None,
            audio_data: None,
            isolate_voices: true,
            background_removal: true,
            output_format: None,
            client: None,
        }
    }

    async fn get_or_create_client(&mut self) -> Result<&crate::client::ElevenLabsClient> {
        if self.client.is_none() {
            self.client = Some(crate::client::ElevenLabsClient::from_env()?);
        }
        self.client
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Failed to create ElevenLabs client"))
    }
    
    fn map_audio_format_to_elevenlabs(&self) -> Option<String> {
        match self.output_format {
            Some(AudioFormat::Pcm16) => Some("pcm_s16le_16".to_string()),
            Some(AudioFormat::Mp3) | None => None, // Default format
            _ => None, // Fallback to default
        }
    }
}

impl AudioIsolationBuilder for ElevenLabsAudioIsolationBuilder {
    type Session = ElevenLabsAudioIsolationSession;

    fn with_file(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    fn with_audio_data(mut self, data: Vec<u8>) -> Self {
        self.audio_data = Some(data);
        self
    }

    fn isolate_voices(mut self, isolate: bool) -> Self {
        self.isolate_voices = isolate;
        self
    }

    fn remove_background(mut self, remove: bool) -> Self {
        self.background_removal = remove;
        self
    }

    fn output_format(mut self, format: AudioFormat) -> Self {
        self.output_format = Some(format);
        self
    }

    async fn isolate<F, R>(mut self, matcher: F) -> R
    where
        F: FnOnce(Result<Self::Session, VoiceError>) -> R + Send + 'static,
    {
        let source = match self.source {
            Some(src) => src,
            None => {
                return matcher(Err(VoiceError::Configuration(
                    "Audio source is required for isolation".to_string(),
                )))
            }
        };

        let client_result = self.get_or_create_client().await;
        let client = match client_result {
            Ok(client) => client,
            Err(e) => {
                return matcher(Err(VoiceError::EngineError(format!(
                    "Failed to create ElevenLabs client: {}",
                    e
                ))))
            }
        };

        let session = ElevenLabsAudioIsolationSession {
            client: client.clone(),
            source,
            output_format: self.output_format.unwrap_or(AudioFormat::Mp3),
            file_format: self.map_audio_format_to_elevenlabs(),
            isolate_voices: self.isolate_voices,
            background_removal: self.background_removal,
        };

        matcher(Ok(session))
    }
}

/// Audio isolation session for ElevenLabs
pub struct ElevenLabsAudioIsolationSession {
    client: crate::client::ElevenLabsClient,
    source: String,
    output_format: AudioFormat,
    file_format: Option<String>,
    isolate_voices: bool,
    background_removal: bool,
}

impl AudioIsolationSession for ElevenLabsAudioIsolationSession {
    type AudioStream = Pin<Box<dyn Stream<Item = i16> + Send>>;

    fn into_stream(self) -> Self::AudioStream {
        let stream = stream::unfold(
            (self.client, self.source, self.file_format, false),
            |(client, source, file_format, done)| async move {
                if done {
                    return None;
                }

                let mut audio_isolation_body = crate::endpoints::genai::audio_isolation::AudioIsolationBody::new(source.clone());
                if let Some(format) = &file_format {
                    audio_isolation_body = audio_isolation_body.with_file_format(format.clone());
                }
                let audio_isolation = crate::endpoints::genai::audio_isolation::AudioIsolation::new(audio_isolation_body);

                match client.hit(audio_isolation).await {
                    Ok(audio_bytes) => {
                        let samples: Vec<i16> = audio_bytes
                            .chunks_exact(2)
                            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
                            .collect();

                        if let Some(first_sample) = samples.first() {
                            let first = *first_sample;
                            Some((first, (client, source, file_format, true)))
                        } else {
                            None
                        }
                    }
                    Err(_) => None,
                }
            },
        );

        Box::pin(stream)
    }
}

/// Sound effects builder for ElevenLabs
pub struct ElevenLabsSoundEffectsBuilder {
    description: Option<String>,
    duration_seconds: Option<f32>,
    intensity: f32,
    output_format: Option<AudioFormat>,
    client: Option<crate::client::ElevenLabsClient>,
}

impl ElevenLabsSoundEffectsBuilder {
    pub fn new() -> Self {
        Self {
            description: None,
            duration_seconds: None,
            intensity: 0.3,
            output_format: None,
            client: None,
        }
    }

    async fn get_or_create_client(&mut self) -> Result<&crate::client::ElevenLabsClient> {
        if self.client.is_none() {
            self.client = Some(crate::client::ElevenLabsClient::from_env()?);
        }
        self.client
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Failed to create ElevenLabs client"))
    }
}

impl SoundEffectsBuilder for ElevenLabsSoundEffectsBuilder {
    type Session = ElevenLabsSoundEffectsSession;

    fn describe(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    fn duration_seconds(mut self, seconds: f32) -> Self {
        self.duration_seconds = Some(seconds);
        self
    }

    fn intensity(mut self, intensity: f32) -> Self {
        self.intensity = intensity;
        self
    }

    fn output_format(mut self, format: AudioFormat) -> Self {
        self.output_format = Some(format);
        self
    }

    async fn generate<F, R>(mut self, matcher: F) -> R
    where
        F: FnOnce(Result<Self::Session, VoiceError>) -> R + Send + 'static,
    {
        let description = match self.description {
            Some(desc) => desc,
            None => {
                return matcher(Err(VoiceError::Configuration(
                    "Sound description is required for generation".to_string(),
                )))
            }
        };

        let client_result = self.get_or_create_client().await;
        let client = match client_result {
            Ok(client) => client,
            Err(e) => {
                return matcher(Err(VoiceError::EngineError(format!(
                    "Failed to create ElevenLabs client: {}",
                    e
                ))))
            }
        };

        let session = ElevenLabsSoundEffectsSession {
            client: client.clone(),
            description,
            duration_seconds: self.duration_seconds,
            intensity: self.intensity,
            output_format: self.output_format.unwrap_or(AudioFormat::Mp3),
        };

        matcher(Ok(session))
    }
}

/// Sound effects session for ElevenLabs
pub struct ElevenLabsSoundEffectsSession {
    client: crate::client::ElevenLabsClient,
    description: String,
    duration_seconds: Option<f32>,
    intensity: f32,
    output_format: AudioFormat,
}

impl SoundEffectsSession for ElevenLabsSoundEffectsSession {
    type AudioStream = Pin<Box<dyn Stream<Item = i16> + Send>>;

    fn into_stream(self) -> Self::AudioStream {
        let stream = stream::unfold(
            (self.client, self.description, self.duration_seconds, self.intensity, false),
            |(client, description, duration, intensity, done)| async move {
                if done {
                    return None;
                }

                let mut body = crate::endpoints::genai::sound_effects::CreateSoundEffectBody::new(description.clone())
                    .with_prompt_influence(intensity);

                if let Some(duration) = duration {
                    body = body.with_duration_seconds(duration);
                }

                let sound_effect = crate::endpoints::genai::sound_effects::CreateSoundEffect::new(body);

                match client.hit(sound_effect).await {
                    Ok(audio_bytes) => {
                        let samples: Vec<i16> = audio_bytes
                            .chunks_exact(2)
                            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
                            .collect();

                        if let Some(first_sample) = samples.first() {
                            let first = *first_sample;
                            Some((first, (client, description, duration, intensity, true)))
                        } else {
                            None
                        }
                    }
                    Err(_) => None,
                }
            },
        );

        Box::pin(stream)
    }
}
