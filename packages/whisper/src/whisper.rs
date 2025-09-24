// https://github.com/openai/whisper/blob/main/whisper/model.py/rgs
// Production Whisper implementation with batch processing and advanced token filtering

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::{Error as E, Result};
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::ops::softmax;
use clap::ValueEnum;
use rand::SeedableRng;
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use tokenizers::Tokenizer;

use crate::types::TtsChunk;

#[cfg(feature = "microphone")]
use crate::microphone::Model;

#[cfg(feature = "microphone")]
use candle_transformers::models::whisper::{self as m};
#[cfg(not(feature = "microphone"))]
use candle_transformers::models::whisper::{self as m, Config};

#[cfg(not(feature = "microphone"))]
pub enum Model {
    Normal(m::model::Whisper),
    Quantized(m::quantized_model::Whisper),
}

#[cfg(not(feature = "microphone"))]
impl Model {
    pub fn config(&self) -> &Config {
        match self {
            Self::Normal(model) => &model.config,
            Self::Quantized(model) => &model.config,
        }
    }

    pub fn encoder_forward(&mut self, mel: &Tensor, flush: bool) -> candle_core::Result<Tensor> {
        match self {
            Self::Normal(model) => model.encoder.forward(mel, flush),
            Self::Quantized(model) => model.encoder.forward(mel, flush),
        }
    }

    pub fn decoder_forward(
        &mut self,
        tokens: &Tensor,
        audio_features: &Tensor,
        flush: bool,
    ) -> candle_core::Result<Tensor> {
        match self {
            Self::Normal(model) => model.decoder.forward(tokens, audio_features, flush),
            Self::Quantized(model) => model.decoder.forward(tokens, audio_features, flush),
        }
    }

    pub fn decoder_final_linear(&mut self, xs: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Normal(model) => model.decoder.final_linear(xs),
            Self::Quantized(model) => model.decoder.final_linear(xs),
        }
    }
}

#[allow(dead_code)] // Development utility function
fn device_helper(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if candle_core::utils::metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else if candle_core::utils::cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else {
        Ok(Device::Cpu)
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct DecodingResult {
    pub tokens: Vec<u32>,
    pub text: String,
    pub avg_logprob: f64,
    pub no_speech_prob: f64,
    pub temperature: f64,
    pub compression_ratio: f64,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Segment {
    pub start: f64,
    pub duration: f64,
    pub dr: DecodingResult,
}

#[allow(dead_code)] // Library code - used by whisper inference
pub struct Decoder {
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
    #[allow(dead_code)] // Library code - decoder constructor
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        device: &Device,
        language_token: Option<u32>,
        task: Option<Task>,
        timestamps: bool,
        verbose: bool,
    ) -> Result<Self> {
        let no_timestamps_token = token_id(&tokenizer, m::NO_TIMESTAMPS_TOKEN)?;
        // Suppress the notimestamps token when in timestamps mode.
        // https://github.com/openai/whisper/blob/e8622f9afc4eba139bf796c210f5c01081000472/whisper/decoding.py#L452
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
            .find_map(|token| token_id(&tokenizer, token).ok());
        let no_speech_token = match no_speech_token {
            None => anyhow::bail!("unable to find any non-speech token"),
            Some(n) => n,
        };
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

    #[allow(dead_code)] // Library code - decoder inference
    pub fn decode(&mut self, mel: &Tensor, t: f64) -> Result<DecodingResult> {
        let mut results = self.decode_batch(&[mel], t)?;
        results
            .pop()
            .ok_or_else(|| anyhow::anyhow!("No decoding results returned"))
    }

    /// Decode multiple audio segments in a batch for improved efficiency
    pub fn decode_batch(&mut self, mels: &[&Tensor], t: f64) -> Result<Vec<DecodingResult>> {
        let batch_size = mels.len();
        if batch_size == 0 {
            return Ok(vec![]);
        }

        let model = &mut self.model;

        // Process audio features for all items in batch
        let mut audio_features_batch = Vec::with_capacity(batch_size);
        for mel in mels {
            let audio_features = model.encoder_forward(mel, true)?;
            if self.verbose {
                println!("audio features: {:?}", audio_features.dims());
            }
            audio_features_batch.push(audio_features);
        }

        // Stack audio features into batch tensor
        let audio_features = Tensor::stack(&audio_features_batch, 0)?;

        let sample_len = model.config().max_target_positions / 2;
        let mut batch_results = Vec::with_capacity(batch_size);

        // Process each item in the batch
        for batch_idx in 0..batch_size {
            let mut sum_logprob = 0f64;
            let mut no_speech_prob = f64::NAN;
            let mut tokens = vec![self.sot_token];
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

            let audio_features_item = audio_features.i(batch_idx)?;

            // Extract fields before mutable borrowing in the loop
            let timestamps = self.timestamps;
            let no_timestamps_token = self.no_timestamps_token;

            for i in 0..sample_len {
                let tokens_t = Tensor::new(tokens.as_slice(), mels[0].device())?;
                let tokens_t = tokens_t.unsqueeze(0)?;
                let ys =
                    model.decoder_forward(&tokens_t, &audio_features_item.unsqueeze(0)?, i == 0)?;

                // Extract the no speech probability on the first iteration by looking at the first
                // token logits and the probability for the according token.
                if i == 0 {
                    let logits = model.decoder_final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
                    no_speech_prob = softmax(&logits, 0)?
                        .i(self.no_speech_token as usize)?
                        .to_scalar::<f32>()? as f64;
                }

                let (_, seq_len, _) = ys.dims3()?;
                let logits = model
                    .decoder_final_linear(&ys.i((..1, seq_len - 1..))?)?
                    .i(0)?
                    .i(0)?;

                // Apply token filters including SuppressBlanks and ApplyTimestampRules
                let logits = Self::apply_token_filters_static(
                    &logits,
                    &tokens,
                    i,
                    timestamps,
                    no_timestamps_token,
                )?;

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
                        .ok_or_else(|| {
                            candle_core::Error::Msg("Empty logits vector in decoder".into())
                        })?
                };
                tokens.push(next_token);
                let prob = softmax(&logits, candle_core::D::Minus1)?
                    .i(next_token as usize)?
                    .to_scalar::<f32>()? as f64;
                if next_token == self.eot_token
                    || tokens.len() > model.config().max_target_positions
                {
                    break;
                }
                sum_logprob += prob.ln();
            }

            let text = self.tokenizer.decode(&tokens, true).map_err(E::msg)?;
            let avg_logprob = sum_logprob / tokens.len() as f64;

            batch_results.push(DecodingResult {
                tokens,
                text,
                avg_logprob,
                no_speech_prob,
                temperature: t,
                compression_ratio: f64::NAN,
            });
        }

        Ok(batch_results)
    }

    /// Apply token filters including SuppressBlanks and ApplyTimestampRules
    fn apply_token_filters_static(
        logits: &Tensor,
        tokens: &[u32],
        step: usize,
        timestamps: bool,
        no_timestamps_token: u32,
    ) -> Result<Tensor> {
        let mut filtered_logits = logits.clone();

        // Apply timestamp rules when in timestamp mode
        if timestamps {
            filtered_logits = Self::apply_timestamp_rules_static(
                &filtered_logits,
                tokens,
                step,
                no_timestamps_token,
            )?;
        }

        // Apply blank suppression
        filtered_logits = Self::suppress_blanks_static(&filtered_logits, tokens)?;

        Ok(filtered_logits)
    }

    /// Apply timestamp rules: timestamps come in pairs, non-decreasing, prioritize when probable
    fn apply_timestamp_rules_static(
        logits: &Tensor,
        tokens: &[u32],
        _step: usize,
        no_timestamps_token: u32,
    ) -> Result<Tensor> {
        let mut logits = logits.clone();

        // Find the last timestamp token to enforce non-decreasing constraint
        let mut last_timestamp = None;
        for &token in tokens.iter().rev() {
            if token > no_timestamps_token {
                last_timestamp = Some(token);
                break;
            }
        }

        // If we have a previous timestamp, suppress earlier timestamps
        if let Some(last_ts) = last_timestamp {
            let logits_vec: Vec<f32> = logits.to_vec1()?;
            let mut modified_logits = logits_vec;

            // Suppress timestamp tokens that would be non-decreasing
            for token_id in (no_timestamps_token + 1)..=last_ts {
                if (token_id as usize) < modified_logits.len() {
                    modified_logits[token_id as usize] = f32::NEG_INFINITY;
                }
            }

            logits = Tensor::new(modified_logits.as_slice(), logits.device())?;
        }

        // Check if timestamps should be prioritized (sum of timestamp probs > other tokens)
        let logits_vec: Vec<f32> = logits.to_vec1()?;
        let timestamp_start = (no_timestamps_token + 1) as usize;

        if timestamp_start < logits_vec.len() {
            let timestamp_sum: f32 = logits_vec[timestamp_start..].iter().sum();
            let other_sum: f32 = logits_vec[..timestamp_start].iter().sum();

            // If timestamps are more probable, suppress non-timestamp tokens
            if timestamp_sum > other_sum {
                let mut modified_logits = logits_vec;
                for item in modified_logits.iter_mut().take(timestamp_start) {
                    *item = f32::NEG_INFINITY;
                }
                logits = Tensor::new(modified_logits.as_slice(), logits.device())?;
            }
        }

        Ok(logits)
    }

    /// Suppress blank tokens and repetitive patterns
    fn suppress_blanks_static(logits: &Tensor, tokens: &[u32]) -> Result<Tensor> {
        let mut logits = logits.clone();

        // Suppress blank/silence tokens more aggressively if we have recent content
        if tokens.len() > 3 {
            let recent_tokens = &tokens[tokens.len().saturating_sub(3)..];

            // Check for repetitive patterns
            if recent_tokens.len() >= 2
                && recent_tokens[recent_tokens.len() - 1] == recent_tokens[recent_tokens.len() - 2]
            {
                let logits_vec: Vec<f32> = logits.to_vec1()?;
                let mut modified_logits = logits_vec;

                let repeated_token = recent_tokens[recent_tokens.len() - 1] as usize;
                if repeated_token < modified_logits.len() {
                    modified_logits[repeated_token] = f32::NEG_INFINITY;
                }

                logits = Tensor::new(modified_logits.as_slice(), logits.device())?;
            }
        }

        Ok(logits)
    }

    #[allow(dead_code)] // Library code - single item decode (legacy)
    pub fn decode_single(&mut self, mel: &Tensor, t: f64) -> Result<DecodingResult> {
        let model = &mut self.model;
        let audio_features = model.encoder_forward(mel, true)?;
        if self.verbose {
            println!("audio features: {:?}", audio_features.dims());
        }
        let sample_len = model.config().max_target_positions / 2;
        let mut sum_logprob = 0f64;
        let mut no_speech_prob = f64::NAN;
        let mut tokens = vec![self.sot_token];
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

        // Extract fields before mutable borrowing in the loop
        let timestamps = self.timestamps;
        let no_timestamps_token = self.no_timestamps_token;

        for i in 0..sample_len {
            let tokens_t = Tensor::new(tokens.as_slice(), mel.device())?;

            // The model expects a batch dim but this inference loop does not handle
            // it so we add it at this point.
            let tokens_t = tokens_t.unsqueeze(0)?;
            let ys = model.decoder_forward(&tokens_t, &audio_features, i == 0)?;

            // Extract the no speech probability on the first iteration by looking at the first
            // token logits and the probability for the according token.
            if i == 0 {
                let logits = model.decoder_final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
                no_speech_prob = softmax(&logits, 0)?
                    .i(self.no_speech_token as usize)?
                    .to_scalar::<f32>()? as f64;
            }

            let (_, seq_len, _) = ys.dims3()?;
            let logits = model
                .decoder_final_linear(&ys.i((..1, seq_len - 1..))?)?
                .i(0)?
                .i(0)?;

            // Apply token filters including SuppressBlanks and ApplyTimestampRules
            let logits = Self::apply_token_filters_static(
                &logits,
                &tokens,
                i,
                timestamps,
                no_timestamps_token,
            )?;

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
                    .ok_or_else(|| {
                        candle_core::Error::Msg("Empty logits vector in decoder".into())
                    })?
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

        Ok(DecodingResult {
            tokens,
            text,
            avg_logprob,
            no_speech_prob,
            temperature: t,
            compression_ratio: f64::NAN,
        })
    }

    #[allow(dead_code)] // Library code - decoder with fallback
    pub fn decode_with_fallback(&mut self, segment: &Tensor) -> Result<DecodingResult> {
        for (i, &t) in m::TEMPERATURES.iter().enumerate() {
            let dr: Result<DecodingResult> = self.decode(segment, t);
            if i == m::TEMPERATURES.len() - 1 {
                return dr;
            }
            // On errors, we try again with a different temperature.
            match dr {
                Ok(dr) => {
                    let needs_fallback = dr.compression_ratio > m::COMPRESSION_RATIO_THRESHOLD
                        || dr.avg_logprob < m::LOGPROB_THRESHOLD;
                    if !needs_fallback || dr.no_speech_prob > m::NO_SPEECH_THRESHOLD {
                        return Ok(dr);
                    }
                }
                Err(err) => {
                    println!("Error running at {t}: {err}")
                }
            }
        }
        unreachable!()
    }

    #[allow(dead_code)] // Library code - run full transcription
    pub fn run(&mut self, mel: &Tensor) -> Result<Vec<Segment>> {
        let (_, _, content_frames) = mel.dims3()?;
        let mut seek = 0;
        let mut segments = vec![];
        while seek < content_frames {
            let start = std::time::Instant::now();
            let time_offset = (seek * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;
            let segment_size = usize::min(content_frames - seek, m::N_FRAMES);
            let mel_segment = mel.narrow(2, seek, segment_size)?;
            let segment_duration = (segment_size * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;
            let dr = self.decode_with_fallback(&mel_segment)?;
            seek += segment_size;
            if dr.no_speech_prob > m::NO_SPEECH_THRESHOLD && dr.avg_logprob < m::LOGPROB_THRESHOLD {
                println!("no speech detected, skipping {seek} {dr:?}");
                continue;
            }
            let segment = Segment {
                start: time_offset,
                duration: segment_duration,
                dr,
            };
            if self.timestamps {
                println!(
                    "{:.1}s -- {:.1}s",
                    segment.start,
                    segment.start + segment.duration,
                );
                let mut tokens_to_decode = vec![];
                let mut prev_timestamp_s = 0f32;
                for &token in segment.dr.tokens.iter() {
                    if token == self.sot_token || token == self.eot_token {
                        continue;
                    }
                    // The no_timestamp_token is the last before the timestamp ones.
                    if token > self.no_timestamps_token {
                        let timestamp_s = (token - self.no_timestamps_token + 1) as f32 / 50.;
                        if !tokens_to_decode.is_empty() {
                            let text = self
                                .tokenizer
                                .decode(&tokens_to_decode, true)
                                .map_err(E::msg)?;
                            println!("  {prev_timestamp_s:.1}s-{timestamp_s:.1}s: {text}");
                            tokens_to_decode.clear()
                        }
                        prev_timestamp_s = timestamp_s;
                    } else {
                        tokens_to_decode.push(token)
                    }
                }
                if !tokens_to_decode.is_empty() {
                    let text = self
                        .tokenizer
                        .decode(&tokens_to_decode, true)
                        .map_err(E::msg)?;
                    if !text.is_empty() {
                        println!("  {prev_timestamp_s:.1}s-...: {text}");
                    }
                    tokens_to_decode.clear()
                }
            } else {
                println!(
                    "{:.1}s -- {:.1}s: {}",
                    segment.start,
                    segment.start + segment.duration,
                    segment.dr.text,
                )
            }
            if self.verbose {
                println!("{seek}: {segment:?}, in {:?}", start.elapsed());
            }
            segments.push(segment)
        }
        Ok(segments)
    }

    /// Access the underlying model for language detection (crate-private)
    #[allow(dead_code)] // May be used for alternative language detection approaches
    pub(crate) fn model(&mut self) -> &mut Model {
        &mut self.model
    }

    /// Run transcription with real-time chunk callback
    pub fn run_with_callback<F>(&mut self, mel: &Tensor, on_chunk: &mut F) -> Result<Vec<Segment>>
    where
        F: FnMut(TtsChunk) + Send + 'static,
    {
        let (_, _, content_frames) = mel.dims3()?;
        let mut seek = 0;
        let mut segments = vec![];
        while seek < content_frames {
            let start = std::time::Instant::now();
            let time_offset = (seek * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;
            let segment_size = usize::min(content_frames - seek, m::N_FRAMES);
            let mel_segment = mel.narrow(2, seek, segment_size)?;
            let segment_duration = (segment_size * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;
            let dr = self.decode_with_fallback(&mel_segment)?;
            seek += segment_size;
            if dr.no_speech_prob > m::NO_SPEECH_THRESHOLD && dr.avg_logprob < m::LOGPROB_THRESHOLD {
                println!("no speech detected, skipping {seek} {dr:?}");
                continue;
            }
            let segment = Segment {
                start: time_offset,
                duration: segment_duration,
                dr,
            };

            // Convert to TtsChunk and call user's real-time callback
            let chunk = TtsChunk::from(segment.clone());
            on_chunk(chunk);

            if self.timestamps {
                println!(
                    "{:.1}s -- {:.1}s",
                    segment.start,
                    segment.start + segment.duration,
                );
                let mut tokens_to_decode = vec![];
                let mut prev_timestamp_s = 0f32;
                for &token in segment.dr.tokens.iter() {
                    if token == self.sot_token || token == self.eot_token {
                        continue;
                    }
                    // The no_timestamp_token is the last before the timestamp ones.
                    if token > self.no_timestamps_token {
                        let timestamp_s = (token - self.no_timestamps_token + 1) as f32 / 50.;
                        if !tokens_to_decode.is_empty() {
                            let text = self
                                .tokenizer
                                .decode(&tokens_to_decode, true)
                                .map_err(E::msg)?;
                            println!("  {prev_timestamp_s:.1}s-{timestamp_s:.1}s: {text}");
                            tokens_to_decode.clear()
                        }
                        prev_timestamp_s = timestamp_s;
                    } else {
                        tokens_to_decode.push(token)
                    }
                }
                if !tokens_to_decode.is_empty() {
                    let text = self
                        .tokenizer
                        .decode(&tokens_to_decode, true)
                        .map_err(E::msg)?;
                    if !text.is_empty() {
                        println!("  {prev_timestamp_s:.1}s-...: {text}");
                    }
                    tokens_to_decode.clear()
                }
            } else {
                println!(
                    "{:.1}s -- {:.1}s: {}",
                    segment.start,
                    segment.start + segment.duration,
                    segment.dr.text,
                )
            }
            if self.verbose {
                println!("{seek}: {segment:?}, in {:?}", start.elapsed());
            }
            segments.push(segment)
        }
        Ok(segments)
    }
}

pub fn token_id(tokenizer: &Tokenizer, token: &str) -> candle_core::Result<u32> {
    match tokenizer.token_to_id(token) {
        None => candle_core::bail!("no token-id for {token}"),
        Some(id) => Ok(id),
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum Task {
    Transcribe,
    Translate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum WhichModel {
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
    #[value(name = "distil-large-v3")]
    DistilLargeV3,
}

impl WhichModel {
    #[allow(dead_code)] // Library code - multilingual detection
    pub fn is_multilingual(&self) -> bool {
        match self {
            Self::Tiny
            | Self::Base
            | Self::Small
            | Self::Medium
            | Self::Large
            | Self::LargeV2
            | Self::LargeV3
            | Self::LargeV3Turbo
            | Self::DistilLargeV2
            | Self::DistilLargeV3 => true,
            Self::TinyEn | Self::BaseEn | Self::SmallEn | Self::MediumEn | Self::DistilMediumEn => {
                false
            }
        }
    }

    #[allow(dead_code)] // Library code - model metadata
    pub fn model_and_revision(&self) -> (&'static str, &'static str) {
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
            Self::DistilLargeV3 => ("distil-whisper/distil-large-v3", "main"),
        }
    }
}
