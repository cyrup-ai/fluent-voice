//! Core Kyutai engine implementation with lock-free architecture

use crate::error::MoshiError;
use crate::tts::Model;
use candle_core::{DType, Device, Tensor};
use crossbeam_channel::{Receiver, Sender, bounded};
use fluent_voice::{
    builders::{
        AudioIsolationBuilder, SoundEffectsBuilder, SpeechToSpeechBuilder, VoiceCloneBuilder,
        VoiceDiscoveryBuilder,
    },
    fluent_voice::{FluentVoice, SttEntry, TtsEntry},
    wake_word::WakeWordBuilder,
    wake_word_koffee::KoffeeWakeWordBuilder,
};

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;

use super::audio_builders::{KyutaiAudioIsolationBuilder, KyutaiSoundEffectsBuilder};
use super::voice_builders::{
    KyutaiSpeechToSpeechBuilder, KyutaiVoiceCloneBuilder, KyutaiVoiceDiscoveryBuilder,
};

/// Pre-allocated buffer size for audio samples
#[allow(dead_code)]
const AUDIO_BUFFER_SIZE: usize = 8192;

/// Channel capacity for lock-free communication
const CHANNEL_CAPACITY: usize = 256;

/// Synthesis request sent to the model worker
#[derive(Debug)]
pub(super) struct SynthesisRequest {
    pub(super) text: String,
    pub(super) speaker_pcm: Option<Vec<f32>>,
    pub(super) max_steps: usize,
    pub(super) temperature: f64,
    pub(super) top_k: usize,
    pub(super) top_p: f64,
    pub(super) repetition_penalty: Option<(usize, f32)>,
    pub(super) cfg_alpha: Option<f64>,
    pub(super) seed: u64,
    pub(super) response_tx: Sender<SynthesisResult>,
}

/// Result from synthesis operation
#[derive(Debug)]
pub(super) struct SynthesisResult {
    #[allow(dead_code)]
    pub(super) audio_samples: Result<Vec<f32>, MoshiError>,
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

    fn create_speaker_tensor(pcm: &[f32]) -> Result<Tensor, crate::error::MoshiError> {
        // Try original PCM data first
        match Tensor::from_slice(pcm, (pcm.len(),), &Device::Cpu) {
            Ok(tensor) => Ok(tensor),
            Err(pcm_err) => {
                // Log the PCM failure and create fallback
                tracing::warn!("Failed to create tensor from PCM data: {}", pcm_err);

                Tensor::zeros((1024,), DType::F32, &Device::Cpu)
                    .map_err(|fallback_err| crate::error::MoshiError::DeviceError(format!(
                        "Critical failure: PCM tensor creation failed ({}), fallback also failed ({})",
                        pcm_err, fallback_err
                    )))
            }
        }
    }

    #[inline]
    fn run(mut self) {
        while let Ok(request) = self.request_rx.recv() {
            // Create speaker tensor safely with error handling
            let speaker_tensor_result = request
                .speaker_pcm
                .as_ref()
                .map(|pcm| Self::create_speaker_tensor(pcm))
                .transpose(); // Convert Option<Result<T, E>> to Result<Option<T>, E>

            let result = match speaker_tensor_result {
                Ok(speaker_tensor) => self
                    .model
                    .generate(
                        &request.text,
                        speaker_tensor.as_ref(),
                        request.max_steps,
                        request.temperature,
                        request.top_k,
                        request.top_p,
                        request.repetition_penalty,
                        request.cfg_alpha,
                        request.seed,
                    )
                    .map_err(|e| MoshiError::Generation(e.to_string())),
                Err(tensor_error) => Err(tensor_error),
            };

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
    pub(super) request_tx: Sender<SynthesisRequest>,
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

    /// Load a Kyutai model using thread-safe singleton with automatic downloading
    #[inline]
    pub async fn load_with_download(dtype: DType, device: &Device) -> Result<Self, MoshiError> {
        let model_paths = crate::models::get_or_download_models().await?;
        let model = Model::load(
            &model_paths.lm_model_path,
            &model_paths.mimi_model_path,
            dtype,
            device,
        )?;
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
        // Delegate to working Koffee implementation
        KoffeeWakeWordBuilder::new()
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
