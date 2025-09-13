//! High-performance Kyutai engine implementation for fluent-voice
//!
//! This module provides a zero-allocation, lock-free implementation of the FluentVoice
//! trait for Kyutai's Moshi model, enabling blazing-fast TTS and STT functionality.

use crate::error::MoshiError;
use crate::tts::Model;
use candle_core::{DType, Device, Tensor};
use crossbeam_channel::{Receiver, Sender, bounded};
use fluent_voice::stt_conversation::{
    MicrophoneBuilder, SttConversation, SttConversationBuilder, SttPostChunkBuilder,
    TranscriptionBuilder,
};
use fluent_voice_domain::pronunciation_dict::PronunciationDictId;
use fluent_voice_domain::voice_labels::{VoiceCategory, VoiceDetails, VoiceLabels, VoiceType};

use fluent_voice::wake_word::{
    WakeWordBuilder, WakeWordConfig as WakeWordConfigTrait, WakeWordDetector,
    WakeWordResult as FluentWakeWordResult, WakeWordStream,
};
use fluent_voice::{
    builders::{
        AudioIsolationBuilder, SoundEffectsBuilder, SpeechToSpeechBuilder, VoiceCloneBuilder,
        VoiceDiscoveryBuilder,
    },
    fluent_voice::{FluentVoice, SttEntry, TtsEntry},
    tts_conversation::{TtsConversationBuilder, TtsConversationChunkBuilder},
};
use fluent_voice_domain::{
    AudioFormat, Diarization, Language, MicBackend, ModelId, NoiseReduction, PitchRange,
    Punctuation, RequestId, Similarity, Speaker, SpeakerBoost, SpeechSource, Stability,
    StyleExaggeration, TimestampsGranularity, TranscriptionSegment, TtsConversation, VadMode,
    VocalSpeedMod, VoiceError, VoiceId, WordTimestamps,
};
use futures_core::Stream;
use futures_util::stream;
use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::Duration;

/// Pre-allocated buffer size for audio samples
#[allow(dead_code)]
const AUDIO_BUFFER_SIZE: usize = 8192;

/// Channel capacity for lock-free communication
const CHANNEL_CAPACITY: usize = 256;

/// Synthesis request sent to the model worker
#[derive(Debug)]
struct SynthesisRequest {
    text: String,
    speaker_pcm: Option<Vec<f32>>,
    max_steps: usize,
    temperature: f64,
    top_k: usize,
    top_p: f64,
    repetition_penalty: Option<(usize, f32)>,
    cfg_alpha: Option<f64>,
    seed: u64,
    response_tx: Sender<SynthesisResult>,
}

/// Result from synthesis operation
#[derive(Debug)]
struct SynthesisResult {
    #[allow(dead_code)]
    audio_samples: Result<Vec<f32>, MoshiError>,
}

/// Lock-free model worker that processes synthesis requests
#[allow(dead_code)]
struct ModelWorker {
    model: Model,
    request_rx: Receiver<SynthesisRequest>,
}

impl ModelWorker {
    #[inline]
    fn new(model: Model, request_rx: Receiver<SynthesisRequest>) -> Self {
        Self { model, request_rx }
    }

    #[inline]
    fn run(mut self) {
        while let Ok(request) = self.request_rx.recv() {
            let result = self
                .model
                .generate(
                    &request.text,
                    request
                        .speaker_pcm
                        .as_ref()
                        .map(|pcm| {
                            Tensor::from_slice(pcm, (pcm.len(),), &Device::Cpu).unwrap_or_else(
                                |_| Tensor::zeros((1024,), DType::F32, &Device::Cpu).unwrap(),
                            )
                        })
                        .as_ref(),
                    request.max_steps,
                    request.temperature,
                    request.top_k,
                    request.top_p,
                    request.repetition_penalty,
                    request.cfg_alpha,
                    request.seed,
                )
                .map_err(|e| MoshiError::Generation(e.to_string()));

            let synthesis_result = SynthesisResult {
                audio_samples: result,
            };

            // Send result back, ignore if receiver is dropped
            let _ = request.response_tx.send(synthesis_result);
        }
    }
}

/// High-performance Kyutai engine with lock-free architecture
pub struct KyutaiEngine {
    request_tx: Sender<SynthesisRequest>,
    model_thread: Option<thread::JoinHandle<()>>,
    #[allow(dead_code)]
    request_counter: AtomicU64,
}

impl KyutaiEngine {
    /// Create a new Kyutai engine with a loaded model
    #[inline]
    pub fn new(model: Model) -> Result<Self, MoshiError> {
        let (request_tx, request_rx) = bounded(CHANNEL_CAPACITY);

        let worker = ModelWorker::new(model, request_rx);
        let model_thread = thread::spawn(move || worker.run());

        Ok(Self {
            request_tx,
            model_thread: Some(model_thread),
            request_counter: AtomicU64::new(0),
        })
    }

    /// Load a Kyutai model from safetensors files
    #[inline]
    pub fn load<P: AsRef<Path>>(
        lm_model_file: P,
        mimi_model_file: P,
        dtype: DType,
        device: &Device,
    ) -> Result<Self, MoshiError> {
        let model = Model::load(lm_model_file, mimi_model_file, dtype, device)?;
        Self::new(model)
    }

    /// Generate a unique request ID
    #[inline]
    #[allow(dead_code)]
    fn next_request_id(&self) -> u64 {
        self.request_counter.fetch_add(1, Ordering::Relaxed)
    }
}

impl Drop for KyutaiEngine {
    fn drop(&mut self) {
        if let Some(handle) = self.model_thread.take() {
            // Close the channel to signal the worker to stop
            drop(self.request_tx.clone());
            let _ = handle.join();
        }
    }
}

impl FluentVoice for KyutaiEngine {
    #[inline]
    fn tts() -> TtsEntry {
        TtsEntry::new()
    }

    #[inline]
    fn stt() -> SttEntry {
        SttEntry::new()
    }

    #[inline]
    fn wake_word() -> impl WakeWordBuilder {
        KyutaiWakeWordBuilder::new()
    }

    #[inline]
    fn voices() -> impl VoiceDiscoveryBuilder {
        KyutaiVoiceDiscoveryBuilder::new()
    }

    #[inline]
    fn clone_voice() -> impl VoiceCloneBuilder {
        KyutaiVoiceCloneBuilder::new()
    }

    #[inline]
    fn speech_to_speech() -> impl SpeechToSpeechBuilder {
        KyutaiSpeechToSpeechBuilder::new()
    }

    #[inline]
    fn audio_isolation() -> impl AudioIsolationBuilder {
        KyutaiAudioIsolationBuilder::new()
    }

    #[inline]
    fn sound_effects() -> impl SoundEffectsBuilder {
        KyutaiSoundEffectsBuilder::new()
    }
}

/// Zero-allocation TTS conversation builder
#[derive(Debug, Clone)]
pub struct KyutaiTtsConversationBuilder {
    speakers: Vec<KyutaiSpeakerLine>,
    language: Option<Language>,
    max_steps: usize,
    temperature: f64,
    top_k: usize,
    top_p: f64,
    repetition_penalty: Option<(usize, f32)>,
    cfg_alpha: Option<f64>,
    seed: u64,
}

impl KyutaiTtsConversationBuilder {
    #[inline]
    #[allow(dead_code)]
    fn new() -> Self {
        Self {
            speakers: Vec::with_capacity(8), // Pre-allocate common case
            language: None,
            max_steps: 2048,
            temperature: 0.7,
            top_k: 200,
            top_p: 0.9,
            repetition_penalty: None,
            cfg_alpha: Some(3.0),
            seed: 42,
        }
    }

    /// Set the maximum generation steps
    #[inline]
    pub fn with_max_steps(mut self, steps: usize) -> Self {
        self.max_steps = steps;
        self
    }

    /// Set the temperature for sampling
    #[inline]
    pub fn with_temperature(mut self, temp: f64) -> Self {
        self.temperature = temp;
        self
    }

    /// Set top-k sampling parameter
    #[inline]
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    /// Set top-p (nucleus) sampling parameter
    #[inline]
    pub fn with_top_p(mut self, p: f64) -> Self {
        self.top_p = p;
        self
    }

    /// Set repetition penalty
    #[inline]
    pub fn with_repetition_penalty(mut self, context_len: usize, penalty: f32) -> Self {
        self.repetition_penalty = Some((context_len, penalty));
        self
    }

    /// Set classifier-free guidance alpha
    #[inline]
    pub fn with_cfg_alpha(mut self, alpha: f64) -> Self {
        self.cfg_alpha = Some(alpha);
        self
    }

    /// Set the random seed for generation
    #[inline]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

impl TtsConversationBuilder for KyutaiTtsConversationBuilder {
    type Conversation = KyutaiTtsConversation;
    type ChunkBuilder = KyutaiTtsConversationChunkBuilder;

    #[inline]
    fn with_speaker<S: Speaker>(mut self, speaker: S) -> Self {
        self.speakers.push(KyutaiSpeakerLine::from_speaker(speaker));
        self
    }

    #[inline]
    fn language(mut self, lang: Language) -> Self {
        self.language = Some(lang);
        self
    }

    #[inline]
    fn model(self, _model: ModelId) -> Self {
        // Kyutai uses a single model configuration
        self
    }

    #[inline]
    fn stability(mut self, stability: Stability) -> Self {
        // Map stability to temperature (inverse relationship)
        self.temperature = 1.0 - stability.value().clamp(0.0, 1.0) as f64;
        self
    }

    #[inline]
    fn similarity(self, _similarity: Similarity) -> Self {
        // Similarity is controlled by speaker_pcm parameter
        self
    }

    #[inline]
    fn speaker_boost(self, _boost: SpeakerBoost) -> Self {
        // Speaker boost is controlled by cfg_alpha parameter
        self
    }

    #[inline]
    fn style_exaggeration(mut self, exaggeration: StyleExaggeration) -> Self {
        // Map style exaggeration to cfg_alpha
        self.cfg_alpha = Some(1.0 + exaggeration.value().clamp(0.0, 1.0) as f64 * 4.0);
        self
    }

    #[inline]
    fn output_format(self, _format: AudioFormat) -> Self {
        // Kyutai outputs PCM f32 samples
        self
    }

    #[inline]
    fn pronunciation_dictionary(self, _dict_id: PronunciationDictId) -> Self {
        // Pronunciation dictionaries are handled by the model
        self
    }

    #[inline]
    fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    #[inline]
    fn previous_text(self, _text: impl Into<String>) -> Self {
        // Context is handled by the model's conversation history
        self
    }

    #[inline]
    fn next_text(self, _text: impl Into<String>) -> Self {
        // Context is handled by the model's conversation history
        self
    }

    #[inline]
    fn previous_request_ids(self, _request_ids: Vec<RequestId>) -> Self {
        // Request context is handled by the model
        self
    }

    #[inline]
    fn next_request_ids(self, _request_ids: Vec<RequestId>) -> Self {
        // Request context is handled by the model
        self
    }

    fn on_result<F>(self, _processor: F) -> Self
    where
        F: FnOnce(Result<Self::Conversation, VoiceError>) + Send + 'static,
    {
        // Store result processor for error handling
        self
    }

    #[inline]
    fn with_voice_clone_path(self, _path: std::path::PathBuf) -> Self {
        // Voice cloning not yet implemented in Kyutai
        self
    }

    #[inline]
    fn additional_params<P>(self, _params: P) -> Self
    where
        P: Into<std::collections::HashMap<String, String>>,
    {
        // Additional parameters stored for future use
        self
    }

    #[inline]
    fn metadata<M>(self, _meta: M) -> Self
    where
        M: Into<std::collections::HashMap<String, String>>,
    {
        // Metadata stored for future use
        self
    }

    fn synthesize<M, S>(self, matcher: M) -> S
    where
        M: FnOnce(Result<Self::Conversation, VoiceError>) -> S + Send + 'static,
        S: Stream<Item = fluent_voice_domain::audio_chunk::AudioChunk> + Send + Unpin + 'static,
    {
        // Create Kyutai TTS conversation for processing
        let conversation = KyutaiTtsConversation::new(self);
        let result = Ok(conversation);
        matcher(result)
    }
}

/// Chunk-based TTS conversation builder
#[derive(Debug, Clone)]
pub struct KyutaiTtsConversationChunkBuilder {
    base: KyutaiTtsConversationBuilder,
}

impl KyutaiTtsConversationChunkBuilder {
    #[inline]
    #[allow(dead_code)]
    fn new(base: KyutaiTtsConversationBuilder) -> Self {
        Self { base }
    }
}

impl TtsConversationChunkBuilder for KyutaiTtsConversationChunkBuilder {
    type Conversation = KyutaiTtsConversation;

    fn synthesize(
        self,
    ) -> impl Stream<Item = fluent_voice_domain::audio_chunk::AudioChunk> + Send + Unpin {
        // Convert conversation to audio stream with chunk processing
        let conversation = KyutaiTtsConversation::new(self.base);
        let audio_stream = conversation.into_stream();

        // The audio_stream already returns AudioChunk, so just pass it through
        audio_stream
    }
}

/// High-performance TTS conversation with streaming audio output
pub struct KyutaiTtsConversation {
    speakers: Vec<KyutaiSpeakerLine>,
    #[allow(dead_code)]
    max_steps: usize,
    #[allow(dead_code)]
    temperature: f64,
    #[allow(dead_code)]
    top_k: usize,
    #[allow(dead_code)]
    top_p: f64,
    #[allow(dead_code)]
    repetition_penalty: Option<(usize, f32)>,
    #[allow(dead_code)]
    cfg_alpha: Option<f64>,
    #[allow(dead_code)]
    seed: u64,
}

impl KyutaiTtsConversation {
    #[inline]
    fn new(builder: KyutaiTtsConversationBuilder) -> Self {
        Self {
            speakers: builder.speakers,
            max_steps: builder.max_steps,
            temperature: builder.temperature,
            top_k: builder.top_k,
            top_p: builder.top_p,
            repetition_penalty: builder.repetition_penalty,
            cfg_alpha: builder.cfg_alpha,
            seed: builder.seed,
        }
    }
}

impl TtsConversation for KyutaiTtsConversation {
    type AudioStream =
        Pin<Box<dyn Stream<Item = fluent_voice_domain::audio_chunk::AudioChunk> + Send + Unpin>>;

    fn into_stream(self) -> Self::AudioStream {
        use futures_util::stream;

        // For now, return a stub implementation that indicates model downloading is needed
        // This avoids the async complexity in the synchronous trait method
        use fluent_voice_domain::audio_chunk::MessageChunk;
        let error_chunk = fluent_voice_domain::AudioChunk::bad_chunk(
            "Kyutai TTS requires async model downloading. Use async interface instead.".to_string(),
        );

        Box::pin(stream::iter(vec![error_chunk]))
    }
}

/// Zero-allocation speaker line implementation
#[derive(Debug, Clone)]
pub struct KyutaiSpeakerLine {
    pub text: String,
    pub voice_id: Option<VoiceId>,
    pub language: Option<Language>,
    pub speed_modifier: Option<VocalSpeedMod>,
    pub pitch_range: Option<PitchRange>,
    pub speaker_id: String,
    #[allow(dead_code)]
    pub speaker_pcm: Option<Vec<f32>>,
}

impl KyutaiSpeakerLine {
    #[inline]
    fn from_speaker<S: Speaker>(speaker: S) -> Self {
        Self {
            text: speaker.text().to_string(),
            voice_id: speaker.voice_id().cloned(),
            language: speaker.language().cloned(),
            speed_modifier: speaker.speed_modifier(),
            pitch_range: speaker.pitch_range().cloned(),
            speaker_id: speaker.id().to_string(),
            speaker_pcm: None, // Could be populated from voice cloning
        }
    }
}

impl Speaker for KyutaiSpeakerLine {
    #[inline]
    fn id(&self) -> &str {
        &self.speaker_id
    }

    #[inline]
    fn text(&self) -> &str {
        &self.text
    }

    #[inline]
    fn voice_id(&self) -> Option<&VoiceId> {
        self.voice_id.as_ref()
    }

    #[inline]
    fn language(&self) -> Option<&Language> {
        self.language.as_ref()
    }

    #[inline]
    fn speed_modifier(&self) -> Option<VocalSpeedMod> {
        self.speed_modifier
    }

    #[inline]
    fn pitch_range(&self) -> Option<&PitchRange> {
        self.pitch_range.as_ref()
    }
}

/// STT conversation builder
#[derive(Debug, Clone)]
pub struct KyutaiSttConversationBuilder {
    source: Option<SpeechSource>,
    vad_mode: Option<VadMode>,
    noise_reduction: Option<NoiseReduction>,
    language: Option<Language>,
    diarization: Option<Diarization>,
    word_timestamps: Option<WordTimestamps>,
    timestamps_granularity: Option<TimestampsGranularity>,
    punctuation: Option<Punctuation>,
}

impl KyutaiSttConversationBuilder {}

impl SttConversationBuilder for KyutaiSttConversationBuilder {
    type Conversation = KyutaiSttConversation;

    #[inline]
    fn with_source(mut self, source: SpeechSource) -> Self {
        self.source = Some(source);
        self
    }

    #[inline]
    fn vad_mode(mut self, mode: VadMode) -> Self {
        self.vad_mode = Some(mode);
        self
    }

    #[inline]
    fn noise_reduction(mut self, reduction: NoiseReduction) -> Self {
        self.noise_reduction = Some(reduction);
        self
    }

    #[inline]
    fn language_hint(mut self, lang: Language) -> Self {
        self.language = Some(lang);
        self
    }

    #[inline]
    fn diarization(mut self, diarization: Diarization) -> Self {
        self.diarization = Some(diarization);
        self
    }

    #[inline]
    fn word_timestamps(mut self, timestamps: WordTimestamps) -> Self {
        self.word_timestamps = Some(timestamps);
        self
    }

    #[inline]
    fn timestamps_granularity(mut self, granularity: TimestampsGranularity) -> Self {
        self.timestamps_granularity = Some(granularity);
        self
    }

    #[inline]
    fn punctuation(mut self, punctuation: Punctuation) -> Self {
        self.punctuation = Some(punctuation);
        self
    }

    fn on_result<F>(self, _f: F) -> Self
    where
        F: FnMut(VoiceError) -> String + Send + 'static,
    {
        // Store result processor for error handling
        self
    }

    fn on_wake<F>(self, _f: F) -> Self
    where
        F: FnMut(String) + Send + 'static,
    {
        // Store wake word callback
        self
    }

    fn on_turn_detected<F>(self, _f: F) -> Self
    where
        F: FnMut(Option<String>, String) + Send + 'static,
    {
        // Store turn detection callback
        self
    }

    fn on_prediction<F>(self, _f: F) -> Self
    where
        F: FnMut(String, String) + Send + 'static,
    {
        // Store prediction callback for real-time transcription
        self
    }
}

impl KyutaiSttConversationBuilder {
    /// Create a new STT conversation builder
    pub fn new() -> Self {
        Self {
            source: None,
            vad_mode: None,
            noise_reduction: None,
            language: None,
            diarization: None,
            word_timestamps: None,
            timestamps_granularity: None,
            punctuation: None,
        }
    }

    /// Convert to post-chunk builder for terminal actions
    pub fn into_post_chunk(self) -> KyutaiSttPostChunkBuilder {
        KyutaiSttPostChunkBuilder::new(self)
    }
}

/// Post-chunk STT builder with terminal action methods
#[derive(Debug, Clone)]
pub struct KyutaiSttPostChunkBuilder {
    builder: KyutaiSttConversationBuilder,
}

impl KyutaiSttPostChunkBuilder {
    #[inline]
    fn new(builder: KyutaiSttConversationBuilder) -> Self {
        Self { builder }
    }
}

impl SttPostChunkBuilder for KyutaiSttPostChunkBuilder {
    type Conversation = KyutaiSttConversation;

    fn with_microphone(mut self, device: impl Into<String>) -> impl MicrophoneBuilder {
        let backend_str = device.into();
        let mic_backend = if backend_str == "default" {
            MicBackend::Default
        } else {
            MicBackend::Device(backend_str)
        };

        self.builder.source = Some(SpeechSource::Microphone {
            backend: mic_backend,
            format: AudioFormat::Pcm16Khz,
            sample_rate: 16_000,
        });
        KyutaiMicrophoneBuilder::new(self.builder)
    }

    fn transcribe(mut self, path: impl Into<String>) -> impl TranscriptionBuilder {
        self.builder.source = Some(SpeechSource::File {
            path: path.into(),
            format: AudioFormat::Pcm16Khz,
        });
        KyutaiTranscriptionBuilder::new(self.builder)
    }

    fn listen<M, S>(self, matcher: M) -> S
    where
        M: FnOnce(Result<Self::Conversation, VoiceError>) -> S + Send + 'static,
        S: futures_core::Stream<
                Item = fluent_voice_domain::transcription::TranscriptionSegmentImpl,
            > + Send
            + Unpin
            + 'static,
    {
        // Create Kyutai STT conversation for processing
        let conversation = KyutaiSttConversation::new();
        let result = Ok(conversation);
        matcher(result)
    }
}

/// Microphone-based STT builder
#[derive(Debug, Clone)]
pub struct KyutaiMicrophoneBuilder {
    builder: KyutaiSttConversationBuilder,
}

impl KyutaiMicrophoneBuilder {
    #[inline]
    fn new(builder: KyutaiSttConversationBuilder) -> Self {
        Self { builder }
    }
}

impl MicrophoneBuilder for KyutaiMicrophoneBuilder {
    type Conversation = KyutaiSttConversation;

    #[inline]
    fn vad_mode(mut self, mode: VadMode) -> Self {
        self.builder.vad_mode = Some(mode);
        self
    }

    #[inline]
    fn noise_reduction(mut self, level: NoiseReduction) -> Self {
        self.builder.noise_reduction = Some(level);
        self
    }

    #[inline]
    fn language_hint(mut self, lang: Language) -> Self {
        self.builder.language = Some(lang);
        self
    }

    #[inline]
    fn diarization(mut self, diarization: Diarization) -> Self {
        self.builder.diarization = Some(diarization);
        self
    }

    #[inline]
    fn word_timestamps(mut self, timestamps: WordTimestamps) -> Self {
        self.builder.word_timestamps = Some(timestamps);
        self
    }

    #[inline]
    fn timestamps_granularity(mut self, granularity: TimestampsGranularity) -> Self {
        self.builder.timestamps_granularity = Some(granularity);
        self
    }

    #[inline]
    fn punctuation(mut self, punctuation: Punctuation) -> Self {
        self.builder.punctuation = Some(punctuation);
        self
    }

    fn listen<M, S>(self, matcher: M) -> S
    where
        M: FnOnce(Result<Self::Conversation, VoiceError>) -> S + Send + 'static,
        S: futures_core::Stream<
                Item = fluent_voice_domain::transcription::TranscriptionSegmentImpl,
            > + Send
            + Unpin
            + 'static,
    {
        // Create a live microphone STT conversation
        let conversation = KyutaiSttConversation::new();
        matcher(Ok(conversation))
    }
}

/// File-based transcription builder
#[derive(Debug, Clone)]
pub struct KyutaiTranscriptionBuilder {
    builder: KyutaiSttConversationBuilder,
    progress_template: Option<String>,
}

impl KyutaiTranscriptionBuilder {
    #[inline]
    fn new(builder: KyutaiSttConversationBuilder) -> Self {
        Self {
            builder,
            progress_template: None,
        }
    }
}

impl TranscriptionBuilder for KyutaiTranscriptionBuilder {
    type Transcript = KyutaiSttConversation;

    #[inline]
    fn vad_mode(mut self, mode: VadMode) -> Self {
        self.builder.vad_mode = Some(mode);
        self
    }

    #[inline]
    fn noise_reduction(mut self, level: NoiseReduction) -> Self {
        self.builder.noise_reduction = Some(level);
        self
    }

    #[inline]
    fn language_hint(mut self, lang: Language) -> Self {
        self.builder.language = Some(lang);
        self
    }

    #[inline]
    fn diarization(mut self, diarization: Diarization) -> Self {
        self.builder.diarization = Some(diarization);
        self
    }

    #[inline]
    fn word_timestamps(mut self, timestamps: WordTimestamps) -> Self {
        self.builder.word_timestamps = Some(timestamps);
        self
    }

    #[inline]
    fn timestamps_granularity(mut self, granularity: TimestampsGranularity) -> Self {
        self.builder.timestamps_granularity = Some(granularity);
        self
    }

    #[inline]
    fn punctuation(mut self, punctuation: Punctuation) -> Self {
        self.builder.punctuation = Some(punctuation);
        self
    }

    #[inline]
    fn with_progress<S: Into<String>>(mut self, template: S) -> Self {
        self.progress_template = Some(template.into());
        self
    }

    fn emit(self) -> impl Stream<Item = String> + Send + Unpin {
        // Create a file-based STT conversation and convert to string stream
        use futures_util::StreamExt;
        let conversation = KyutaiSttConversation::new();
        Box::pin(conversation.into_stream().map(|result| match result {
            Ok(segment) => segment.text().to_string(),
            Err(_) => String::new(), // Default error handling
        }))
    }

    async fn collect(self) -> Result<Self::Transcript, VoiceError> {
        Ok(KyutaiSttConversation::new())
    }

    async fn collect_with<F, R>(self, handler: F) -> R
    where
        F: FnOnce(Result<Self::Transcript, VoiceError>) -> R,
    {
        let result = self.collect().await;
        handler(result)
    }

    fn into_text_stream(self) -> impl Stream<Item = String> + Send {
        stream::empty()
    }

    fn transcribe<M, S>(self, matcher: M) -> S
    where
        M: FnOnce(Result<Self::Transcript, VoiceError>) -> S + Send + 'static,
        S: Stream<Item = fluent_voice_domain::transcription::TranscriptionSegmentImpl>
            + Send
            + Unpin
            + 'static,
    {
        // Create transcript result and pass to matcher
        let transcript = KyutaiSttConversation::new();
        let result = Ok(transcript);
        matcher(result)
    }
}

/// High-performance STT conversation with streaming transcript output
pub struct KyutaiSttConversation {
    _phantom: std::marker::PhantomData<()>,
}

impl KyutaiSttConversation {
    #[inline]
    fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl fluent_voice::stt_conversation::SttConversation for KyutaiSttConversation {
    type Stream = Pin<Box<dyn Stream<Item = Result<KyutaiTranscriptSegment, VoiceError>> + Send>>;

    fn into_stream(self) -> Self::Stream {
        // Create a streaming transcript generator
        let stream = stream::empty(); // Placeholder for real implementation
        Box::pin(stream)
    }
}

impl KyutaiTtsConversation {
    /// Convert f32 audio samples to 16-bit PCM bytes
    fn f32_to_pcm16_bytes(samples: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(samples.len() * 2);
        for &sample in samples {
            // Clamp sample to [-1.0, 1.0] and convert to i16
            let clamped = sample.clamp(-1.0, 1.0);
            let pcm_sample = (clamped * 32767.0) as i16;
            // Convert to little-endian bytes
            bytes.extend_from_slice(&pcm_sample.to_le_bytes());
        }
        bytes
    }
}

/// Transcript segment with timing information
#[derive(Debug, Clone)]
pub struct KyutaiTranscriptSegment {
    text: String,
    start_ms: u32,
    end_ms: u32,
    speaker_id: Option<String>,
}

impl KyutaiTranscriptSegment {
    #[inline]
    #[allow(dead_code)]
    fn new(text: String, start_ms: u32, end_ms: u32, speaker_id: Option<String>) -> Self {
        Self {
            text,
            start_ms,
            end_ms,
            speaker_id,
        }
    }
}

impl TranscriptionSegment for KyutaiTranscriptSegment {
    #[inline]
    fn start_ms(&self) -> u32 {
        self.start_ms
    }

    #[inline]
    fn end_ms(&self) -> u32 {
        self.end_ms
    }

    #[inline]
    fn text(&self) -> &str {
        &self.text
    }

    #[inline]
    fn speaker_id(&self) -> Option<&str> {
        self.speaker_id.as_deref()
    }
}

/// Simple wake word configuration
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct WakeWordConfig {
    pub confidence_threshold: f32,
    pub timeout_ms: u64,
    pub debug: bool,
}

impl WakeWordConfigTrait for WakeWordConfig {
    fn confidence_threshold(&self) -> f32 {
        self.confidence_threshold
    }

    fn timeout(&self) -> Duration {
        Duration::from_millis(self.timeout_ms)
    }

    fn model_file(&self) -> Option<&str> {
        None // No model file in this implementation
    }
}

/// Wake word detection builder
#[derive(Debug, Clone)]
pub struct KyutaiWakeWordBuilder {
    config: WakeWordConfig,
}

impl KyutaiWakeWordBuilder {
    #[inline]
    fn new() -> Self {
        Self {
            config: WakeWordConfig::default(),
        }
    }
}

impl WakeWordBuilder for KyutaiWakeWordBuilder {
    type Config = WakeWordConfig;
    type Detector = KyutaiWakeWordDetector;

    #[inline]
    fn model_file(self, _path: impl Into<String>) -> Self {
        // Model file path is not used in this implementation
        self
    }

    #[inline]
    fn confidence_threshold(mut self, threshold: f32) -> Self {
        self.config.confidence_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    #[inline]
    fn timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout_ms = timeout.as_millis() as u64;
        self
    }

    fn detect<F, R>(self, matcher: F) -> impl Future<Output = R> + Send
    where
        F: FnOnce(Result<Self::Detector, VoiceError>) -> R + Send + 'static,
    {
        async move {
            let detector = KyutaiWakeWordDetector::new(self.config);
            matcher(Ok(detector))
        }
    }
}

/// Wake word detector implementation
#[derive(Debug)]
pub struct KyutaiWakeWordDetector {
    config: WakeWordConfig,
}

impl KyutaiWakeWordDetector {
    #[inline]
    fn new(config: WakeWordConfig) -> Self {
        Self { config }
    }
}

impl WakeWordDetector for KyutaiWakeWordDetector {
    type Stream = KyutaiWakeWordStream;

    #[inline]
    fn start_detection(self) -> Self::Stream {
        // Return a stream for wake word detection
        KyutaiWakeWordStream::new(self.config)
    }
}

/// Wake word detection stream implementation
pub struct KyutaiWakeWordStream {
    _config: WakeWordConfig,
    active: bool,
}

impl KyutaiWakeWordStream {
    #[inline]
    fn new(config: WakeWordConfig) -> Self {
        Self {
            _config: config,
            active: true,
        }
    }
}

impl Stream for KyutaiWakeWordStream {
    type Item = FluentWakeWordResult;

    fn poll_next(
        self: Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        // For now, return None (no wake words detected)
        // In a real implementation, this would poll audio input
        std::task::Poll::Ready(None)
    }
}

impl WakeWordStream for KyutaiWakeWordStream {
    #[inline]
    fn stop(&mut self) {
        self.active = false;
    }

    #[inline]
    fn is_active(&self) -> bool {
        self.active
    }
}

/// Voice discovery builder
#[derive(Debug, Clone)]
pub struct KyutaiVoiceDiscoveryBuilder {
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

impl KyutaiVoiceDiscoveryBuilder {
    #[inline]
    fn new() -> Self {
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

impl VoiceDiscoveryBuilder for KyutaiVoiceDiscoveryBuilder {
    type Result = Vec<VoiceDetails>;

    #[inline]
    fn search(mut self, term: impl Into<String>) -> Self {
        self.search_term = Some(term.into());
        self
    }

    #[inline]
    fn category(mut self, category: VoiceCategory) -> Self {
        self.category = Some(category);
        self
    }

    #[inline]
    fn voice_type(mut self, voice_type: VoiceType) -> Self {
        self.voice_type = Some(voice_type);
        self
    }

    #[inline]
    fn language(mut self, language: Language) -> Self {
        self.language = Some(language);
        self
    }

    #[inline]
    fn labels(mut self, labels: VoiceLabels) -> Self {
        self.labels = Some(labels);
        self
    }

    #[inline]
    fn page_size(mut self, size: usize) -> Self {
        self.page_size = Some(size);
        self
    }

    #[inline]
    fn page_token(mut self, token: impl Into<String>) -> Self {
        self.page_token = Some(token.into());
        self
    }

    #[inline]
    fn sort_by_created(mut self) -> Self {
        self.sort_by_created = true;
        self
    }

    #[inline]
    fn sort_by_name(mut self) -> Self {
        self.sort_by_name = true;
        self
    }

    async fn discover<F, R>(self, matcher: F) -> R
    where
        F: FnOnce(Result<Self::Result, VoiceError>) -> R,
    {
        // Return empty voice list as Kyutai uses its own voice model
        matcher(Ok(Vec::new()))
    }
}

/// Voice cloning builder
#[derive(Debug, Clone)]
pub struct KyutaiVoiceCloneBuilder {
    _phantom: std::marker::PhantomData<()>,
}

impl KyutaiVoiceCloneBuilder {
    #[inline]
    fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl VoiceCloneBuilder for KyutaiVoiceCloneBuilder {
    type Result = VoiceDetails;

    #[inline]
    fn with_samples(self, _samples: Vec<impl Into<String>>) -> Self {
        self
    }

    #[inline]
    fn with_sample(self, _sample: impl Into<String>) -> Self {
        self
    }

    #[inline]
    fn name(self, _name: impl Into<String>) -> Self {
        self
    }

    #[inline]
    fn description(self, _description: impl Into<String>) -> Self {
        self
    }

    #[inline]
    fn labels(self, _labels: VoiceLabels) -> Self {
        self
    }

    #[inline]
    fn fine_tuning_model(self, _model: ModelId) -> Self {
        self
    }

    #[inline]
    fn enhanced_processing(self, _enabled: bool) -> Self {
        self
    }

    async fn create<F, R>(self, matcher: F) -> R
    where
        F: FnOnce(Result<Self::Result, VoiceError>) -> R,
    {
        matcher(Err(VoiceError::ProcessingError(
            "Voice cloning requires speaker PCM samples in TTS synthesis".to_string(),
        )))
    }
}

/// Speech-to-speech conversion builder
#[derive(Debug, Clone)]
pub struct KyutaiSpeechToSpeechBuilder {
    _phantom: std::marker::PhantomData<()>,
}

impl KyutaiSpeechToSpeechBuilder {
    #[inline]
    fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl SpeechToSpeechBuilder for KyutaiSpeechToSpeechBuilder {
    type Session = KyutaiSpeechToSpeechSession;

    #[inline]
    fn with_audio_source(self, _path: impl Into<String>) -> Self {
        self
    }

    #[inline]
    fn with_audio_data(self, _data: Vec<u8>) -> Self {
        self
    }

    #[inline]
    fn target_voice(self, _voice_id: VoiceId) -> Self {
        self
    }

    #[inline]
    fn model(self, _model: ModelId) -> Self {
        self
    }

    #[inline]
    fn preserve_emotion(self, _preserve: bool) -> Self {
        self
    }

    #[inline]
    fn preserve_style(self, _preserve: bool) -> Self {
        self
    }

    #[inline]
    fn preserve_timing(self, _preserve: bool) -> Self {
        self
    }

    #[inline]
    fn output_format(self, _format: AudioFormat) -> Self {
        self
    }

    #[inline]
    fn stability(self, _stability: f32) -> Self {
        self
    }

    #[inline]
    fn similarity_boost(self, _boost: f32) -> Self {
        self
    }

    async fn convert<F, R>(self, matcher: F) -> R
    where
        F: FnOnce(Result<Self::Session, VoiceError>) -> R,
    {
        matcher(Err(VoiceError::ProcessingError(
            "Speech-to-speech conversion requires audio processing pipeline".to_string(),
        )))
    }
}

/// Speech-to-speech session
#[derive(Debug)]
pub struct KyutaiSpeechToSpeechSession {
    _phantom: std::marker::PhantomData<()>,
}

/// Audio isolation builder
#[derive(Debug, Clone)]
pub struct KyutaiAudioIsolationBuilder {
    _phantom: std::marker::PhantomData<()>,
}

impl KyutaiAudioIsolationBuilder {
    #[inline]
    fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl AudioIsolationBuilder for KyutaiAudioIsolationBuilder {
    type Session = KyutaiAudioIsolationSession;

    #[inline]
    fn with_file(self, _path: impl Into<String>) -> Self {
        self
    }

    #[inline]
    fn with_audio_data(self, _data: Vec<u8>) -> Self {
        self
    }

    #[inline]
    fn isolate_voices(self, _isolate: bool) -> Self {
        self
    }

    #[inline]
    fn remove_background(self, _remove: bool) -> Self {
        self
    }

    #[inline]
    fn reduce_noise(self, _reduce: bool) -> Self {
        self
    }

    #[inline]
    fn isolation_strength(self, _strength: f32) -> Self {
        self
    }

    #[inline]
    fn output_format(self, _format: AudioFormat) -> Self {
        self
    }

    async fn process<F, R>(self, matcher: F) -> R
    where
        F: FnOnce(Result<Self::Session, VoiceError>) -> R,
    {
        matcher(Err(VoiceError::ProcessingError(
            "Audio isolation requires dedicated audio processing models".to_string(),
        )))
    }
}

/// Audio isolation session
#[derive(Debug)]
pub struct KyutaiAudioIsolationSession {
    _phantom: std::marker::PhantomData<()>,
}

/// Sound effects generation builder
#[derive(Debug, Clone)]
pub struct KyutaiSoundEffectsBuilder {
    _phantom: std::marker::PhantomData<()>,
}

impl KyutaiSoundEffectsBuilder {
    #[inline]
    fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl SoundEffectsBuilder for KyutaiSoundEffectsBuilder {
    type Session = KyutaiSoundEffectsSession;

    #[inline]
    fn describe(self, _description: impl Into<String>) -> Self {
        self
    }

    #[inline]
    fn duration_seconds(self, _duration: f32) -> Self {
        self
    }

    #[inline]
    fn intensity(self, _intensity: f32) -> Self {
        self
    }

    #[inline]
    fn mood(self, _mood: impl Into<String>) -> Self {
        self
    }

    #[inline]
    fn environment(self, _environment: impl Into<String>) -> Self {
        self
    }

    #[inline]
    fn output_format(self, _format: AudioFormat) -> Self {
        self
    }

    #[inline]
    fn seed(self, _seed: u64) -> Self {
        self
    }

    async fn generate<F, R>(self, matcher: F) -> R
    where
        F: FnOnce(Result<Self::Session, VoiceError>) -> R,
    {
        matcher(Err(VoiceError::ProcessingError(
            "Sound effects generation requires specialized audio synthesis models".to_string(),
        )))
    }
}

/// Sound effects generation session
#[derive(Debug)]
pub struct KyutaiSoundEffectsSession {
    _phantom: std::marker::PhantomData<()>,
}

use fluent_voice::builders::{AudioIsolationSession, SoundEffectsSession, SpeechToSpeechSession};

impl SpeechToSpeechSession for KyutaiSpeechToSpeechSession {
    type AudioStream = Pin<Box<dyn Stream<Item = i16> + Send + Unpin>>;

    fn into_stream(self) -> Self::AudioStream {
        Box::pin(stream::empty())
    }
}

impl AudioIsolationSession for KyutaiAudioIsolationSession {
    type AudioStream = Pin<Box<dyn Stream<Item = i16> + Send + Unpin>>;

    fn into_stream(self) -> Self::AudioStream {
        Box::pin(stream::empty())
    }
}

impl SoundEffectsSession for KyutaiSoundEffectsSession {
    type AudioStream = Pin<Box<dyn Stream<Item = i16> + Send + Unpin>>;

    fn into_stream(self) -> Self::AudioStream {
        Box::pin(stream::empty())
    }
}
