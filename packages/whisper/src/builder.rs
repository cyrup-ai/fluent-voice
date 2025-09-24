//! Whisper STT functionality using domain objects for interoperability.

use crate::transcript::Transcript;
use crate::types::TtsChunk;
use fluent_voice_domain::prelude::*;

#[cfg(feature = "microphone")]
use crate::microphone::Model;
#[cfg(not(feature = "microphone"))]
use crate::whisper::Model;
use crate::whisper::{Decoder, Task, WhichModel, token_id};

use anyhow::Result;
use byteorder::{ByteOrder, LittleEndian};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as m, Config};
use futures::stream::StreamExt;
use futures_core::Stream;
use hf_hub::api::sync::Api;
use std::path::PathBuf;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokenizers::Tokenizer;

/// Type alias for the thread-safe chunk callback
type ChunkCallback =
    std::sync::Arc<std::sync::Mutex<Box<dyn FnMut(Result<TtsChunk, VoiceError>) + Send + 'static>>>;

/// Configuration for Whisper model loading and transcription
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_id: String,
    pub which_model: WhichModel,
    pub model_path: Option<PathBuf>,
    pub tokenizer_path: Option<PathBuf>,
    pub seed: u64,
    pub timestamps: bool,
    pub verbose: bool,
    pub temperature: f64,
    pub task: Option<Task>,
    pub language: Option<String>,
    pub quantized: bool,
    pub cpu: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_id: "base".to_string(),
            which_model: WhichModel::Base,
            model_path: None,
            tokenizer_path: None,
            seed: 299792458,
            timestamps: true,
            verbose: false,
            temperature: 0.0,
            task: None,
            language: None,
            quantized: false,
            cpu: false,
        }
    }
}

/// Immutable fluent builder for Whisper STT operations
pub struct WhisperSttBuilder {
    config: ModelConfig,
    source: Option<SpeechSource>,
    chunk_callback: Option<ChunkCallback>,
}

/// Conversation result containing transcript and stream capability
#[derive(Debug)]
pub struct WhisperConversation {
    transcript: Transcript,
}

impl Default for WhisperSttBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl WhisperSttBuilder {
    /// Create a new Whisper STT builder with default configuration
    #[inline]
    pub fn new() -> Self {
        Self {
            config: ModelConfig::default(),
            source: None,
            chunk_callback: None,
        }
    }

    // with_config method removed - not used in current implementation

    /// Configure the speech input source
    #[inline]
    pub fn with_source(mut self, source: SpeechSource) -> Self {
        self.source = Some(source);
        self
    }

    /// Set real-time chunk callback following README.md pattern
    /// Callback receives Result<TtsChunk, VoiceError> and should return transcription segment
    pub fn on_chunk<F>(mut self, callback: F) -> Self
    where
        F: FnMut(Result<TtsChunk, VoiceError>) + Send + 'static,
    {
        self.chunk_callback = Some(std::sync::Arc::new(std::sync::Mutex::new(Box::new(
            callback,
        ))));
        self
    }

    /// Execute microphone transcription with matcher pattern following README.md
    /// Matcher receives Result<WhisperConversation, VoiceError> and returns desired output
    pub async fn listen<F, R>(self, matcher: F) -> R
    where
        F: FnOnce(Result<WhisperConversation, VoiceError>) -> R + Send + 'static,
        R: Send + 'static,
    {
        // Ensure we have microphone source
        match &self.source {
            Some(SpeechSource::Microphone { .. }) => {
                let result = self.execute_transcription().await;
                matcher(result)
            }
            Some(_) => {
                let error = VoiceError::Configuration(
                    "listen() is for microphone transcription. Use transcribe() for files."
                        .to_string(),
                );
                matcher(Err(error))
            }
            None => {
                let error = VoiceError::Configuration(
                    "No speech source configured for transcription".to_string(),
                );
                matcher(Err(error))
            }
        }
    }

    /// Execute file transcription with matcher pattern following README.md
    /// Matcher receives Result<WhisperConversation, VoiceError> and returns desired output
    pub async fn transcribe<F, R>(self, matcher: F) -> R
    where
        F: FnOnce(Result<WhisperConversation, VoiceError>) -> R + Send + 'static,
        R: Send + 'static,
    {
        // Ensure we have file or memory source
        match &self.source {
            Some(SpeechSource::File { .. }) | Some(SpeechSource::Memory { .. }) => {
                let result = self.execute_transcription().await;
                matcher(result)
            }
            Some(SpeechSource::Microphone { .. }) => {
                let error = VoiceError::Configuration(
                    "transcribe() is for file transcription. Use listen() for microphone."
                        .to_string(),
                );
                matcher(Err(error))
            }
            None => {
                let error = VoiceError::Configuration(
                    "No speech source configured for transcription".to_string(),
                );
                matcher(Err(error))
            }
        }
    }
}

impl WhisperSttBuilder {
    /// Internal transcription execution - delegates to existing working modules
    async fn execute_transcription(mut self) -> Result<WhisperConversation, VoiceError> {
        let source = self
            .source
            .clone()
            .ok_or_else(|| VoiceError::Configuration("Speech source not configured".to_string()))?;

        let mut transcript = Transcript::empty();

        match source {
            SpeechSource::File { path, format: _ } => {
                let (model, tokenizer, device, mel_filters) = self.load_model_data().await?;
                let audio_data = self.load_audio_file(&path).await?;
                let mel = self
                    .samples_to_mel(&audio_data, &mel_filters, &device)
                    .await?;
                self.process_with_decoder(model, tokenizer, device, mel, &mut transcript)
                    .await?;
            }
            SpeechSource::Microphone {
                backend: _,
                format: _,
                sample_rate: _,
            } => {
                #[cfg(feature = "microphone")]
                {
                    // Load model data for microphone transcription
                    let (model, tokenizer, device, mel_filters) = self.load_model_data().await?;

                    // Create whisper config from model config
                    let whisper_config = candle_transformers::models::whisper::Config {
                        vocab_size: 51864,
                        num_mel_bins: 80,
                        encoder_layers: 6,
                        encoder_attention_heads: 8,
                        decoder_layers: 6,
                        decoder_attention_heads: 8,
                        d_model: 512,
                        suppress_tokens: vec![],
                        max_target_positions: 448,
                        max_source_positions: 1500,
                    };

                    // Extract callback for microphone
                    let callback = self.chunk_callback.take().map(|cb| {
                        move |chunk: crate::types::TtsChunk| {
                            if let Ok(mut callback) = cb.lock() {
                                callback(Ok(chunk));
                            }
                        }
                    });

                    // Delegate to microphone module with builder support
                    let mic_transcript = crate::microphone::record_with_builder(
                        model,
                        tokenizer,
                        device,
                        mel_filters,
                        whisper_config,
                        self.config.task,
                        self.config.language.clone(),
                        callback,
                    )
                    .await
                    .map_err(|e| VoiceError::ProcessingError(e.to_string()))?;

                    // Merge microphone transcript with main transcript
                    for chunk in mic_transcript.iter() {
                        transcript.push(chunk.clone());
                    }
                }
                #[cfg(not(feature = "microphone"))]
                {
                    return Err(VoiceError::Configuration(
                        "Microphone transcription requires 'microphone' feature".to_string(),
                    ));
                }
            }
            SpeechSource::Memory {
                data,
                format: _,
                sample_rate,
            } => {
                let (model, tokenizer, device, mel_filters) = self.load_model_data().await?;
                let samples = self.bytes_to_samples(data, sample_rate)?;
                let mel = self.samples_to_mel(&samples, &mel_filters, &device).await?;
                self.process_with_decoder(model, tokenizer, device, mel, &mut transcript)
                    .await?;
            }
        }

        Ok(WhisperConversation { transcript })
    }

    /// Process transcription with decoder and callback support
    async fn process_with_decoder(
        &mut self,
        model: Model,
        tokenizer: Tokenizer,
        device: Device,
        mel: Tensor,
        transcript: &mut Transcript,
    ) -> Result<(), VoiceError> {
        let language_token = if self.config.which_model.is_multilingual() {
            match &self.config.language {
                Some(lang) => Some(token_id(&tokenizer, &format!("<|{lang}|>")).map_err(|_| {
                    VoiceError::Configuration(format!("Language {lang} not supported"))
                })?),
                None => Some(token_id(&tokenizer, "<|en|>").map_err(|_| {
                    VoiceError::Configuration("English language token not found".to_string())
                })?),
            }
        } else {
            None
        };

        let mut decoder = Decoder::new(
            model,
            tokenizer,
            self.config.seed,
            &device,
            language_token,
            self.config.task,
            self.config.timestamps,
            self.config.verbose,
        )
        .map_err(|e| VoiceError::ProcessingError(e.to_string()))?;

        let segments = if let Some(callback) = self.chunk_callback.take() {
            decoder
                .run_with_callback(&mel, &mut move |chunk| {
                    if let Ok(mut cb) = callback.lock() {
                        cb(Ok(chunk));
                    }
                })
                .map_err(|e| VoiceError::ProcessingError(e.to_string()))?
        } else {
            decoder
                .run(&mel)
                .map_err(|e| VoiceError::ProcessingError(e.to_string()))?
        };

        for segment in segments {
            let chunk = TtsChunk::new(
                segment.start,
                segment.start + segment.duration,
                segment.dr.tokens,
                segment.dr.text,
                segment.dr.avg_logprob,
                segment.dr.no_speech_prob,
                segment.dr.temperature,
                segment.dr.compression_ratio,
            );
            transcript.push(chunk);
        }

        Ok(())
    }

    /// PRESERVED: Load model data using existing working Candle ML integration
    async fn load_model_data(&self) -> Result<(Model, Tokenizer, Device, Vec<f32>), VoiceError> {
        let device = if self.config.cpu {
            Device::Cpu
        } else {
            Device::new_metal(0)
                .or_else(|_| Device::new_cuda(0))
                .unwrap_or(Device::Cpu)
        };

        let (default_model, _default_revision) = if self.config.quantized {
            ("lmz/candle-whisper", "main")
        } else {
            self.config.which_model.model_and_revision()
        };

        let model_id = self
            .config
            .model_path
            .as_ref()
            .and_then(|p| p.to_str())
            .unwrap_or(default_model);
        let _revision = "main";

        // Download model using hf-hub
        let api = Api::new().map_err(|e| {
            VoiceError::ProcessingError(format!("Failed to create hf-hub client: {e}"))
        })?;
        let repo = api.model(model_id.to_string());

        let (config_filename, tokenizer_filename, weights_filename) = if self.config.quantized {
            let ext = match self.config.which_model {
                WhichModel::TinyEn => "tiny-en",
                WhichModel::Tiny => "tiny",
                _ => {
                    return Err(VoiceError::Configuration(format!(
                        "No quantized support for {:?}",
                        self.config.which_model
                    )));
                }
            };
            (
                repo.get(&format!("config-{ext}.json")).map_err(|e| {
                    VoiceError::ProcessingError(format!(
                        "Failed to download config-{ext}.json: {e}"
                    ))
                })?,
                repo.get(&format!("tokenizer-{ext}.json")).map_err(|e| {
                    VoiceError::ProcessingError(format!(
                        "Failed to download tokenizer-{ext}.json: {e}"
                    ))
                })?,
                repo.get(&format!("model-{ext}-q80.gguf")).map_err(|e| {
                    VoiceError::ProcessingError(format!(
                        "Failed to download model-{ext}-q80.gguf: {e}"
                    ))
                })?,
            )
        } else {
            (
                repo.get("config.json").map_err(|e| {
                    VoiceError::ProcessingError(format!("Failed to download config.json: {e}"))
                })?,
                repo.get("tokenizer.json").map_err(|e| {
                    VoiceError::ProcessingError(format!("Failed to download tokenizer.json: {e}"))
                })?,
                repo.get("model.safetensors").map_err(|e| {
                    VoiceError::ProcessingError(format!(
                        "Failed to download model.safetensors: {e}"
                    ))
                })?,
            )
        };

        let whisper_config: Config = serde_json::from_str(
            &std::fs::read_to_string(config_filename)
                .map_err(|e| VoiceError::ProcessingError(e.to_string()))?,
        )
        .map_err(|e| VoiceError::ProcessingError(e.to_string()))?;

        let tokenizer = Tokenizer::from_file(tokenizer_filename)
            .map_err(|e| VoiceError::ProcessingError(e.to_string()))?;

        let model = if self.config.quantized {
            let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
                &weights_filename,
                &device,
            )
            .map_err(|e| VoiceError::ProcessingError(e.to_string()))?;
            Model::Quantized(
                m::quantized_model::Whisper::load(&vb, whisper_config.clone())
                    .map_err(|e| VoiceError::ProcessingError(e.to_string()))?,
            )
        } else {
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_filename], m::DTYPE, &device)
                    .map_err(|e| VoiceError::ProcessingError(e.to_string()))?
            };
            Model::Normal(
                m::model::Whisper::load(&vb, whisper_config.clone())
                    .map_err(|e| VoiceError::ProcessingError(e.to_string()))?,
            )
        };

        let mel_bytes = match whisper_config.num_mel_bins {
            80 => include_bytes!("melfilters.bytes").as_slice(),
            128 => include_bytes!("melfilters128.bytes").as_slice(),
            nmel => {
                return Err(VoiceError::Configuration(format!(
                    "Unexpected num_mel_bins {nmel}"
                )));
            }
        };
        let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
        LittleEndian::read_f32_into(mel_bytes, &mut mel_filters);

        Ok((model, tokenizer, device, mel_filters))
    }

    /// Load audio file using comprehensive multi-format decoder
    async fn load_audio_file(&self, path: &str) -> Result<Vec<f32>, VoiceError> {
        // Use the comprehensive PCM decoder that supports many formats
        let (mut audio_data, sample_rate) = crate::pcm_decode::pcm_decode(path).map_err(|e| {
            VoiceError::ProcessingError(format!("Failed to decode audio file {}: {}", path, e))
        })?;

        // Resample to 16kHz if needed
        if sample_rate != 16000 {
            use rubato::{FastFixedIn, PolynomialDegree, Resampler};
            let resample_ratio = 16000.0 / sample_rate as f64;
            let mut resampler =
                FastFixedIn::<f32>::new(resample_ratio, 10.0, PolynomialDegree::Septic, 1024, 1)
                    .map_err(|e| {
                        VoiceError::ProcessingError(format!("Resampler init failed: {}", e))
                    })?;

            let output = resampler
                .process(&[&audio_data], None)
                .map_err(|e| VoiceError::ProcessingError(format!("Resampling failed: {}", e)))?;
            audio_data = output[0].clone();
        }

        Ok(audio_data)
    }

    /// PRESERVED: Convert samples to mel spectrogram using existing working implementation
    async fn samples_to_mel(
        &self,
        samples: &[f32],
        mel_filters: &[f32],
        device: &Device,
    ) -> Result<Tensor, VoiceError> {
        let config = Config {
            vocab_size: 51864,
            num_mel_bins: 80,
            encoder_layers: 6,
            encoder_attention_heads: 8,
            decoder_layers: 6,
            decoder_attention_heads: 8,
            d_model: 512,
            suppress_tokens: vec![],
            max_target_positions: 448,
            max_source_positions: 1500,
        };

        let mel =
            candle_transformers::models::whisper::audio::pcm_to_mel(&config, samples, mel_filters);

        let mel_len = mel.len();
        let mel_tensor = Tensor::from_vec(
            mel,
            (1, config.num_mel_bins, mel_len / config.num_mel_bins),
            device,
        )
        .map_err(|e| VoiceError::ProcessingError(e.to_string()))?;

        Ok(mel_tensor)
    }

    /// PRESERVED: Convert bytes to samples using existing working implementation
    fn bytes_to_samples(&self, data: Vec<u8>, _sample_rate: u32) -> Result<Vec<f32>, VoiceError> {
        let samples = data
            .chunks(2)
            .map(|chunk| {
                let sample = i16::from_le_bytes([chunk[0], chunk.get(1).copied().unwrap_or(0)]);
                sample as f32 / 32768.0
            })
            .collect();
        Ok(samples)
    }
}

impl WhisperConversation {
    /// Access the internal transcript for advanced processing
    pub fn transcript(self) -> Transcript {
        self.transcript
    }

    /// Convert conversation to transcript stream following README.md pattern
    pub fn into_stream(
        self,
    ) -> impl Stream<Item = Result<TranscriptionSegmentImpl, VoiceError>> + Send + Unpin {
        struct TranscriptStream {
            transcript: Transcript,
            position: usize,
        }

        impl Stream for TranscriptStream {
            type Item = Result<TranscriptionSegmentImpl, VoiceError>;

            fn poll_next(
                mut self: Pin<&mut Self>,
                _cx: &mut Context<'_>,
            ) -> Poll<Option<Self::Item>> {
                if self.position >= self.transcript.len() {
                    Poll::Ready(None)
                } else {
                    let chunk_text = self.transcript[self.position].text.clone();
                    let chunk_start = self.transcript[self.position].start_ms();
                    let chunk_end = self.transcript[self.position].end_ms();
                    let chunk_speaker = self.transcript[self.position]
                        .speaker_id()
                        .map(|s| s.to_string());
                    self.position += 1;

                    let segment = TranscriptionSegmentImpl::new(
                        chunk_text,
                        chunk_start,
                        chunk_end,
                        chunk_speaker,
                    );

                    Poll::Ready(Some(Ok(segment)))
                }
            }
        }

        TranscriptStream {
            transcript: self.transcript,
            position: 0,
        }
    }

    /// Collect all transcript segments into a complete text transcript
    pub async fn collect(self) -> Result<String, VoiceError> {
        let mut stream = self.into_stream();
        let mut full_text = String::new();

        while let Some(result) = StreamExt::next(&mut stream).await {
            match result {
                Ok(segment) => {
                    if !full_text.is_empty() {
                        full_text.push(' ');
                    }
                    full_text.push_str(segment.text());
                }
                Err(e) => return Err(e),
            }
        }

        Ok(full_text)
    }
}
