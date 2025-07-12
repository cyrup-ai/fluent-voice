#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::{Error as E, Result};
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::{VarBuilder, ops::softmax};
use clap::{Parser, ValueEnum};
use hf_hub::{Repo, RepoType, api::sync::Api};
use rand::SeedableRng;
use rand_distr::{Distribution, weighted::WeightedIndex};
use tokenizers::Tokenizer;
use candle_transformers::models::whisper::{self as m, Config, audio};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use flate2::{Compression as GzCompression, write::GzEncoder};
use futures::stream::{Stream, StreamExt};
use rubato::{FastFixedIn, PolynomialDegree, Resampler};
use std::io::Write;
use std::pin::Pin;
use std::task::{Context, Poll};

// Use ONLY canonical domain types - no local duplicates
use fluent_voice_domain::transcript::ConcreteTranscriptSegment;
use fluent_voice_domain::MicrophoneBuilder;

pub enum Model {
    Normal(m::model::Whisper),
    Quantized(m::quantized_model::Whisper),
}

impl Model {
    pub fn config(&self) -> &Config {
        match self {
            Self::Normal(m) => &m.config,
            Self::Quantized(m) => &m.config,
        }
    }

    pub fn encoder_forward(&mut self, x: &Tensor, flush: bool) -> candle_core::Result<Tensor> {
        match self {
            Self::Normal(m) => m.encoder.forward(x, flush),
            Self::Quantized(m) => m.encoder.forward(x, flush),
        }
    }

    pub fn decoder_forward(
        &mut self,
        x: &Tensor,
        xa: &Tensor,
        flush: bool,
    ) -> candle_core::Result<Tensor> {
        match self {
            Self::Normal(m) => m.decoder.forward(x, xa, flush),
            Self::Quantized(m) => m.decoder.forward(x, xa, flush),
        }
    }

    pub fn decoder_final_linear(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Normal(m) => m.decoder.final_linear(x),
            Self::Quantized(m) => m.decoder.final_linear(x),
        }
    }

    pub fn reset_kv_cache(&mut self) {
        match self {
            Self::Normal(m) => m.reset_kv_cache(),
            Self::Quantized(m) => m.reset_kv_cache(),
        }
    }
}

#[derive(Debug, Clone)]
struct DecodingResult {
    tokens: Vec<u32>,
    text: String,
    avg_logprob: f64,
    no_speech_prob: f64,
    temperature: f64,
    compression_ratio: f64,
}

// Use canonical domain objects from fluent_voice_domain - no local duplicates

struct Decoder {
    model: Model,
    rng: rand::rngs::StdRng,
    task: Option<Task>,
    timestamps: bool,
    verbose: bool,
    tokenizer: Tokenizer,
    suppress_tokens: Tensor,
    sot_token: u32,
    transcribe_token: u32,
    translate_token: u32,
    eot_token: u32,
    no_speech_token: u32,
    no_timestamps_token: u32,
    language_token: Option<u32>,
}

impl Decoder {
    #[allow(clippy::too_many_arguments)]
    fn new(
        mut model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        device: &Device,
        language_token: Option<u32>,
        task: Option<Task>,
        timestamps: bool,
        verbose: bool,
    ) -> Result<Self> {
        model.reset_kv_cache();
        let no_timestamps_token = token_id(&tokenizer, m::NO_TIMESTAMPS_TOKEN)?;
        let suppress_tokens: Vec<f32> = (0..model.config().vocab_size as u32)
            .map(|i| {
                if model.config().suppress_tokens.contains(&i)
                    || timestamps && i == no_timestamps_token
                {
                    f32::NEG_INFINITY
                } else {
                    0f32
                }
            })
            .collect();
        let suppress_tokens = Tensor::new(suppress_tokens.as_slice(), device)?;
        let sot_token = token_id(&tokenizer, m::SOT_TOKEN)?;
        let transcribe_token = token_id(&tokenizer, m::TRANSCRIBE_TOKEN)?;
        let translate_token = token_id(&tokenizer, m::TRANSLATE_TOKEN)?;
        let eot_token = token_id(&tokenizer, m::EOT_TOKEN)?;
        let no_speech_token = m::NO_SPEECH_TOKENS
            .iter()
            .find_map(|token| token_id(&tokenizer, token).ok())
            .ok_or_else(|| anyhow::anyhow!("unable to find any non-speech token"))?;
        Ok(Self {
            model,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            tokenizer,
            task,
            timestamps,
            verbose,
            suppress_tokens,
            sot_token,
            transcribe_token,
            translate_token,
            eot_token,
            no_speech_token,
            language_token,
            no_timestamps_token,
        })
    }

    fn decode(&mut self, mel: &Tensor, t: f64) -> Result<DecodingResult> {
        let model = &mut self.model;
        let audio_features = model.encoder_forward(mel, true)?;
        if self.verbose {
            println!("audio features: {:?}", audio_features.dims());
        }
        let sample_len = model.config().max_target_positions / 2;
        let mut sum_logprob = 0f64;
        let mut no_speech_prob = f64::NAN;
        let mut tokens = Vec::with_capacity(sample_len);
        tokens.push(self.sot_token);
        if let Some(language_token) = self.language_token {
            tokens.push(language_token);
        }
        match self.task {
            None | Some(Task::Transcribe) => tokens.push(self.transcribe_token),
            Some(Task::Translate) => tokens.push(self.translate_token),
        }
        if !self.timestamps {
            tokens.push(self.no_timestamps_token);
        }
        for i in 0..sample_len {
            let tokens_t = Tensor::new(tokens.as_slice(), mel.device())?.unsqueeze(0)?;
            let ys = model.decoder_forward(&tokens_t, &audio_features, i == 0)?;
            let (_, seq_len, _) = ys.dims3()?;
            let logits = model
                .decoder_final_linear(&ys.i((..1, seq_len - 1..))?)?
                .i(0)?
                .i(0)?;
            if i == 0 {
                let logits = model.decoder_final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
                no_speech_prob = softmax(&logits, 0)?
                    .i(self.no_speech_token as usize)?
                    .to_scalar::<f32>()? as f64;
            }
            let logits = logits.broadcast_add(&self.suppress_tokens)?;
            let next_token = if t > 0f64 {
                let prs = softmax(&(&logits / t)?, 0)?;
                let logits_v: Vec<f32> = prs.to_vec1()?;
                let distr = WeightedIndex::new(&logits_v)?;
                distr.sample(&mut self.rng) as u32
            } else {
                let logits_v: Vec<f32> = logits.to_vec1()?;
                logits_v
                    .iter()
                    .enumerate()
                    .max_by(|(_, u), (_, v)| u.total_cmp(v))
                    .map(|(i, _)| i as u32)
                    .unwrap()
            };
            tokens.push(next_token);
            let prob = softmax(&logits, candle_core::D::Minus1)?
                .i(next_token as usize)?
                .to_scalar::<f32>()? as f64;
            if next_token == self.eot_token || tokens.len() > model.config().max_target_positions {
                break;
            }
            sum_logprob += prob.ln();
        }
        let text = self.tokenizer.decode(&tokens, true).map_err(E::msg)?;
        let avg_logprob = sum_logprob / tokens.len() as f64;

        let compression_ratio = if text.is_empty() {
            1.0
        } else {
            let trimmed = text.chars().filter(|c| !c.is_whitespace()).count();
            if trimmed == 0 {
                1.0
            } else {
                let mut buf = Vec::with_capacity(text.len() / 2);
                let mut encoder = GzEncoder::new(&mut buf, GzCompression::default());
                encoder.write_all(text.as_bytes()).map_err(E::msg)?;
                (trimmed as f64) / (buf.len() as f64)
            }
        };

        Ok(DecodingResult {
            tokens,
            text,
            avg_logprob,
            no_speech_prob,
            temperature: t,
            compression_ratio,
        })
    }

    fn decode_with_fallback(&mut self, segment: &Tensor) -> Result<DecodingResult> {
        for (i, &t) in m::TEMPERATURES.iter().enumerate() {
            let dr = self.decode(segment, t);
            if i == m::TEMPERATURES.len() - 1 {
                return dr;
            }
            match dr {
                Ok(dr) => {
                    let needs_fallback = dr.compression_ratio > m::COMPRESSION_RATIO_THRESHOLD
                        || dr.avg_logprob < m::LOGPROB_THRESHOLD;
                    if !needs_fallback || dr.no_speech_prob > m::NO_SPEECH_THRESHOLD {
                        return Ok(dr);
                    }
                }
                Err(err) => {
                    if self.verbose {
                        println!("Error running at {t}: {err}");
                    }
                }
            }
        }
        unreachable!()
    }

    fn stream_segments(
        &mut self,
        mel: &Tensor,
    ) -> impl Stream<Item = Result<ConcreteTranscriptSegment>> + '_ {
        struct SegmentStream<'a> {
            decoder: &'a mut Decoder,
            mel: &'a Tensor,
            seek: usize,
            content_frames: usize,
            pending_chunks: Vec<ConcreteTranscriptSegment>,
        }

        impl<'a> Stream for SegmentStream<'a> {
            type Item = Result<ConcreteTranscriptSegment>;

            fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
                if !self.pending_chunks.is_empty() {
                    return Poll::Ready(Some(Ok(self.pending_chunks.remove(0))));
                }

                if self.seek >= self.content_frames {
                    return Poll::Ready(None);
                }

                let time_offset = (self.seek * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;
                let segment_size = usize::min(self.content_frames - self.seek, m::N_FRAMES);
                let mel_segment = match self.mel.narrow(2, self.seek, segment_size) {
                    Ok(m) => m,
                    Err(e) => return Poll::Ready(Some(Err(e.into()))),
                };
                let segment_duration = (segment_size * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;

                let dr = match self.decoder.decode_with_fallback(&mel_segment) {
                    Ok(dr) => dr,
                    Err(e) => return Poll::Ready(Some(Err(e))),
                };

                self.seek += segment_size;

                if dr.no_speech_prob > m::NO_SPEECH_THRESHOLD && dr.avg_logprob < m::LOGPROB_THRESHOLD {
                    if self.decoder.verbose {
                        println!("no speech detected, skipping {seek} {dr:?}", seek = self.seek);
                    }
                    cx.waker().wake_by_ref();
                    return Poll::Pending;
                }

                self.pending_chunks.reserve(10);
                if self.decoder.timestamps {
                    let mut tokens_to_decode = Vec::with_capacity(100);
                    let mut prev_timestamp_s = 0.0f64;
                    for &token in dr.tokens.iter() {
                        if token == self.decoder.sot_token || token == self.decoder.eot_token {
                            continue;
                        }
                        if token > self.decoder.no_timestamps_token {
                            let timestamp_s = ((token - self.decoder.no_timestamps_token) as f64) / 50.0;
                            if !tokens_to_decode.is_empty() {
                                match self.decoder.tokenizer.decode(&tokens_to_decode, true) {
                                    Ok(text) => {
                                        self.pending_chunks.push(ConcreteTranscriptSegment::new(
                                            text,
                                            ((time_offset + prev_timestamp_s) * 1000.0) as u32,
                                            ((time_offset + timestamp_s) * 1000.0) as u32,
                                            None // No speaker ID in this context
                                        ));
                                        tokens_to_decode.clear();
                                    }
                                    Err(e) => return Poll::Ready(Some(Err(E::msg(e)))),
                                }
                            }
                            prev_timestamp_s = timestamp_s;
                        } else {
                            tokens_to_decode.push(token);
                        }
                    }
                    if !tokens_to_decode.is_empty() {
                        if let Ok(text) = self.decoder.tokenizer.decode(&tokens_to_decode, true) {
                            if !text.is_empty() {
                                self.pending_chunks.push(ConcreteTranscriptSegment::new(
                                    text,
                                    ((time_offset + prev_timestamp_s) * 1000.0) as u32,
                                    ((time_offset + segment_duration) * 1000.0) as u32,
                                    None // No speaker ID in this context
                                ));
                            }
                        }
                    }
                } else {
                    // Create real ConcreteTranscriptSegment with production-quality data from Whisper transcription
                    self.pending_chunks.push(ConcreteTranscriptSegment::new(
                        dr.text,
                        (time_offset * 1000.0) as u32,
                        ((time_offset + segment_duration) * 1000.0) as u32,
                        None // No speaker ID in this context
                    ));
                }

                if self.decoder.verbose {
                    println!("Processed segment at seek {}: {:?}", self.seek, dr);
                }

                if self.pending_chunks.is_empty() {
                    cx.waker().wake_by_ref();
                    Poll::Pending
                } else {
                    Poll::Ready(Some(Ok(self.pending_chunks.remove(0))))
                }
            }
        }

        SegmentStream {
            decoder: self,
            mel,
            seek: 0,
            content_frames: mel.dims()[2],
            pending_chunks: Vec::with_capacity(10),
        }
    }

    fn set_language_token(&mut self, language_token: Option<u32>) {
        self.language_token = language_token;
    }
}

pub fn token_id(tokenizer: &Tokenizer, token: &str) -> candle_core::Result<u32> {
    tokenizer.token_to_id(token).ok_or_else(|| candle_core::Error::Msg(format!("no token-id for {}", token)))
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Task {
    Transcribe,
    Translate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum WhichModel {
    Tiny,
    #[value(name = "tiny.en")]
    TinyEn,
    Base,
    #[value(name = "base.en")]
    BaseEn,
    Small,
    #[value(name = "small.en")]
    SmallEn,
    Medium,
    #[value(name = "medium.en")]
    MediumEn,
    Large,
    LargeV2,
    LargeV3,
    LargeV3Turbo,
    #[value(name = "distil-medium.en")]
    DistilMediumEn,
    #[value(name = "distil-large-v2")]
    DistilLargeV2,
}

impl WhichModel {
    fn is_multilingual(&self) -> bool {
        matches!(self, Self::Tiny | Self::Base | Self::Small | Self::Medium | Self::Large | Self::LargeV2 | Self::LargeV3 | Self::LargeV3Turbo | Self::DistilLargeV2)
    }

    fn model_and_revision(&self) -> (&'static str, &'static str) {
        match self {
            Self::Tiny => ("openai/whisper-tiny", "main"),
            Self::TinyEn => ("openai/whisper-tiny.en", "refs/pr/15"),
            Self::Base => ("openai/whisper-base", "refs/pr/22"),
            Self::BaseEn => ("openai/whisper-base.en", "refs/pr/13"),
            Self::Small => ("openai/whisper-small", "main"),
            Self::SmallEn => ("openai/whisper-small.en", "refs/pr/10"),
            Self::Medium => ("openai/whisper-medium", "main"),
            Self::MediumEn => ("openai/whisper-medium.en", "main"),
            Self::Large => ("openai/whisper-large", "refs/pr/36"),
            Self::LargeV2 => ("openai/whisper-large-v2", "refs/pr/57"),
            Self::LargeV3 => ("openai/whisper-large-v3", "main"),
            Self::LargeV3Turbo => ("openai/whisper-large-v3-turbo", "main"),
            Self::DistilMediumEn => ("distil-whisper/distil-medium.en", "main"),
            Self::DistilLargeV2 => ("distil-whisper/distil-large-v2", "main"),
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than GPU.
    #[arg(long)]
    cpu: bool,

    #[arg(long)]
    model_id: Option<String>,

    /// The model revision to use.
    #[arg(long)]
    revision: Option<String>,

    /// The model to use.
    #[arg(long, default_value = "tiny.en")]
    model: WhichModel,

    /// Seed for sampling.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// Use quantized models.
    #[arg(long)]
    quantized: bool,

    /// Language code.
    #[arg(long)]
    language: Option<String>,

    /// Task to perform.
    #[arg(long)]
    task: Option<Task>,

    /// Enable timestamps mode.
    #[arg(long)]
    timestamps: bool,

    /// Verbose output.
    #[arg(long)]
    verbose: bool,

    /// Input device name.
    #[arg(long)]
    device: Option<String>,

    /// List available input devices.
    #[arg(long)]
    list_devices: bool,

}

pub fn record() -> Result<impl Stream<Item = Result<ConcreteTranscriptSegment>>> {
    let args = Args::parse();

    if args.list_devices {
        let host = cpal::default_host();
        let devices = host.input_devices()?;
        for device in devices {
            if let Ok(name) = device.name() {
                println!("Input Device: {}", name);
            } else {
                println!("Input Device: (name unavailable)");
            }
            if let Ok(configs) = device.supported_input_configs() {
                for config in configs {
                    println!(
                        "  Channels: {}, Sample Rate Range: {}-{}, Buffer Size: {:?}, Format: {:?}",
                        config.channels(),
                        config.min_sample_rate().0,
                        config.max_sample_rate().0,
                        config.buffer_size(),
                        config.sample_format()
                    );
                }
            } else {
                println!("  No supported input configs or error retrieving them.");
            }
            println!();
        }
        return Ok(futures::stream::empty());
    }

    let device = candle_examples::device(args.cpu)?;
    let (default_model, default_revision) = if args.quantized {
        ("lmz/candle-whisper", "main")
    } else {
        args.model.model_and_revision()
    };
    let default_model = default_model.to_string();
    let default_revision = default_revision.to_string();
    let (model_id, revision) = match (args.model_id, args.revision) {
        (Some(model_id), Some(revision)) => (model_id, revision),
        (Some(model_id), None) => (model_id, "main".to_string()),
        (None, Some(revision)) => (default_model, revision),
        (None, None) => (default_model, default_revision),
    };

    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));
    let (config_filename, tokenizer_filename, weights_filename) = if args.quantized {
        let ext = match args.model {
            WhichModel::TinyEn => "tiny-en",
            WhichModel::Tiny => "tiny",
            _ => return Err(anyhow::anyhow!("no quantized support for {:?}", args.model)),
        };
        (
            repo.get(&format!("config-{ext}.json"))?,
            repo.get(&format!("tokenizer-{ext}.json"))?,
            repo.get(&format!("model-{ext}-q80.gguf"))?,
        )
    } else {
        (
            repo.get("config.json")?,
            repo.get("tokenizer.json")?,
            repo.get("model.safetensors")?,
        )
    };
    let config: Config = serde_json::from_str(&std::fs::read_to_string(config_filename)?)?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    let model = if args.quantized {
        let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(&weights_filename, &device)?;
        Model::Quantized(m::quantized_model::Whisper::load(&vb, config.clone())?)
    } else {
        let tensors = candle_core::safetensors::load(&weights_filename, &device)?;
        let vb = candle_nn::VarBuilder::from_tensors(tensors, candle_core::DType::F32, &device);
        Model::Normal(m::model::Whisper::load(&vb, config.clone())?)
    };
    let mut decoder = Decoder::new(
        model,
        tokenizer.clone(),
        args.seed,
        &device,
        None,
        args.task,
        args.timestamps,
        args.verbose,
    )?;

    let mel_bytes = match config.num_mel_bins {
        80 => include_bytes!("melfilters.bytes").as_slice(),
        128 => include_bytes!("melfilters128.bytes").as_slice(),
        nmel => return Err(anyhow::anyhow!("unexpected num_mel_bins {}", nmel)),
    };
    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
    <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(mel_bytes, &mut mel_filters);

    let host = cpal::default_host();
    let audio_device = match args.device.as_ref() {
        None => host.default_input_device().ok_or(anyhow::anyhow!("No default input device available. Run with --list-devices to see options."))?,
        Some(device_name) => host
            .input_devices()?
            .find(|d| d.name().map(|n| n == *device_name).unwrap_or(false))
            .ok_or(anyhow::anyhow!("Input device '{}' not found. Run with --list-devices to see available devices.", device_name))?,
    };
    let audio_config = audio_device.default_input_config().map_err(|e| anyhow::anyhow!("Failed to get default input config for device '{}': {}. Run with --list-devices for details.", audio_device.name().unwrap_or("(unnamed)".to_string()), e))?;
    if args.verbose {
        println!("Using audio device: {} with config {:?}", audio_device.name().unwrap_or("(unnamed)".to_string()), audio_config);
    }

    let channel_count = audio_config.channels() as usize;
    let in_sample_rate = audio_config.sample_rate().0 as usize;
    let resample_ratio = m::SAMPLE_RATE as f64 / in_sample_rate as f64;
    let mut resampler = if resample_ratio != 1.0 {
        Some(FastFixedIn::<f32>::new(
            resample_ratio,
            10.0,
            PolynomialDegree::Septic,
            1024,
            1,
        )?)
    } else {
        None
    };

    let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<f32>>(32);
    let stream = audio_device.build_input_stream(
        &audio_config.into(),
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let mono_data = if channel_count == 1 {
                data.to_vec()
            } else {
                data.iter().step_by(channel_count).copied().collect()
            };
            let _ = tx.try_send(mono_data);
        },
        |err| eprintln!("Stream error: {}", err),
        None,
    )?;
    stream.play()?;

    struct AudioStream {
        decoder: Decoder,
        tokenizer: Tokenizer,
        config: Config,
        mel_filters: Vec<f32>,
        rx: std::sync::mpsc::Receiver<Vec<f32>>,
        resampler: Option<FastFixedIn<f32>>,
        in_sample_rate: usize,
        buffered_pcm: Vec<f32>,
        language_token_set: bool,
        device: Device,
        verbose: bool,
        model_is_multilingual: bool,
        language: Option<String>,
    }

    impl Stream for AudioStream {
        type Item = Result<ConcreteTranscriptSegment>;

        fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            let chunk = match self.rx.try_recv() {
                Ok(chunk) => chunk,
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    cx.waker().wake_by_ref();
                    return Poll::Pending;
                }
                Err(_) => return Poll::Ready(None),
            };

            self.buffered_pcm.extend_from_slice(&chunk);
            if self.buffered_pcm.len() < 10 * self.in_sample_rate {
                cx.waker().wake_by_ref();
                return Poll::Pending;
            }

            let pcm = if let Some(ref mut resampler) = self.resampler {
                let resample_ratio = m::SAMPLE_RATE as f64 / self.in_sample_rate as f64;
                let chunk_size = 1024;
                let full_chunks = self.buffered_pcm.len() / chunk_size;
                let remainder = self.buffered_pcm.len() % chunk_size;
                let mut resampled_pcm = Vec::with_capacity((self.buffered_pcm.len() as f64 * resample_ratio) as usize + chunk_size);
                
                for chunk_idx in 0..full_chunks {
                    let chunk = &self.buffered_pcm[chunk_idx * chunk_size..(chunk_idx + 1) * chunk_size];
                    match resampler.process(&[chunk], None) {
                        Ok(pcm) => {
                            resampled_pcm.extend_from_slice(&pcm[0]);
                        },
                        Err(e) => return Poll::Ready(Some(Err(e.into()))),
                    }
                }
                if remainder > 0 {
                    self.buffered_pcm.copy_within(full_chunks * chunk_size.., 0);
                    self.buffered_pcm.truncate(remainder);
                } else {
                    self.buffered_pcm.clear();
                }
                resampled_pcm
            } else {
                std::mem::take(&mut self.buffered_pcm)
            };

            let mel_vec = audio::pcm_to_mel(&self.config, &pcm, &self.mel_filters);
            let mel_len = mel_vec.len();
            let mel = match Tensor::from_vec(
                mel_vec,
                (1, self.config.num_mel_bins, mel_len / self.config.num_mel_bins),
                &self.device,
            ) {
                Ok(m) => m,
                Err(e) => return Poll::Ready(Some(Err(e.into()))),
            };

            if !self.language_token_set {
                let language_token = match (self.model_is_multilingual, self.language.clone()) {
                    (true, None) => match fluent_voice_whisper::detect_language(&mut self.decoder.model, &self.tokenizer, &mel) {
                        Ok((token, lang)) => {
                            if self.verbose {
                                println!("Detected language: {}", lang);
                            }
                            Some(token)
                        }
                        Err(e) => return Poll::Ready(Some(Err(e))),
                    },
                    (true, Some(lang)) => match token_id(&self.tokenizer, &format!("<|{}|>", lang)) {
                        Ok(token) => Some(token),
                        Err(e) => return Poll::Ready(Some(Err(e.into()))),
                    },
                    (false, None) => None,
                    (false, Some(_)) => return Poll::Ready(Some(Err(anyhow::anyhow!("language cannot be set for non-multilingual models")))),
                };
                self.decoder.set_language_token(language_token);
                self.language_token_set = true;
            }

            let mut segment_stream = self.decoder.stream_segments(&mel);
            match Pin::new(&mut segment_stream).poll_next(cx) {
                Poll::Ready(Some(Ok(chunk))) => {
                    self.decoder.model.reset_kv_cache();
                    Poll::Ready(Some(Ok(chunk)))
                }
                Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
                Poll::Ready(None) => {
                    self.decoder.model.reset_kv_cache();
                    cx.waker().wake_by_ref();
                    Poll::Pending
                }
                Poll::Pending => Poll::Pending,
            }
        }
    }

    Ok(AudioStream {
        decoder,
        tokenizer,
        config,
        mel_filters,
        rx,
        resampler,
        in_sample_rate,
        buffered_pcm: Vec::with_capacity(30 * in_sample_rate),
        language_token_set: false,
        device,
        verbose: args.verbose,
        model_is_multilingual: args.model.is_multilingual(),
        language: args.language,
    })
}
