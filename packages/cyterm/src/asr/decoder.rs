//! High-level Whisper decoding with temperature fallback, timestamp support, etc.

use anyhow::{Result, anyhow};
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "accelerate",
    feature = "mkl"
))]
use candle_core::{IndexOp, Tensor};
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "accelerate",
    feature = "mkl"
))]
use candle_nn::ops::softmax;
use rand::distributions::WeightedIndex;
use rand::{SeedableRng, rngs::StdRng};
use tokenizers::Tokenizer;

use crate::asr::model::WhisperModel;
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "accelerate",
    feature = "mkl"
))]
use candle_transformers::models::whisper as m;

/// A streaming Whisper decoder.
pub struct WhisperDecoder {
    model: WhisperModel,
    tokenizer: Tokenizer,
    rng: StdRng,
    temperature_fallback: &'static [f64],
    no_timestamp_token: u32,
    suppress: Tensor,

    // cached token-ids
    sot: u32,
    eot: u32,
    transcribe: u32,
    translate: u32,
}

impl WhisperDecoder {
    pub fn new(model: WhisperModel, tokenizer: Tokenizer, seed: u64) -> Result<Self> {
        let device = &model
            .encoder_forward(
                &Tensor::zeros(
                    (1, 1, 1),
                    candle_core::DType::F32,
                    &candle_core::Device::Cpu,
                )?,
                true,
            )?
            .device(); // hack to grab device

        let no_ts = token_id(&tokenizer, m::NO_TIMESTAMPS_TOKEN)?;
        // build suppress-mask
        let mut suppress = vec![0f32; model.config().vocab_size];
        for id in &model.config().suppress_tokens {
            suppress[*id as usize] = f32::NEG_INFINITY;
        }
        suppress[no_ts as usize] = f32::NEG_INFINITY;
        let suppress = Tensor::new(&suppress, device)?;

        Ok(Self {
            model,
            tokenizer,
            rng: StdRng::seed_from_u64(seed),
            temperature_fallback: &m::TEMPERATURES,
            no_timestamp_token: no_ts,
            suppress,
            sot: token_id(&tokenizer, m::SOT_TOKEN)?,
            eot: token_id(&tokenizer, m::EOT_TOKEN)?,
            transcribe: token_id(&tokenizer, m::TRANSCRIBE_TOKEN)?,
            translate: token_id(&tokenizer, m::TRANSLATE_TOKEN)?,
        })
    }

    /// One-shot decode of a mel chunk; auto-fallback on temperature.
    pub fn decode(&mut self, mel: &Tensor, translate: bool) -> Result<String> {
        for &t in self.temperature_fallback {
            match self.decode_once(mel, translate, t) {
                Ok(s) => return Ok(s),
                Err(e) if t != *self.temperature_fallback.last().unwrap() => {
                    eprintln!("fallback at {t}: {e}");
                    continue;
                }
                Err(e) => return Err(e),
            }
        }
        unreachable!()
    }

    fn decode_once(&mut self, mel: &Tensor, translate: bool, temp: f64) -> Result<String> {
        let audio_features = self.model.encoder_forward(mel, true)?;

        let mut tokens = vec![self.sot];
        tokens.push(if translate {
            self.translate
        } else {
            self.transcribe
        });
        tokens.push(self.no_timestamp_token);

        let mut text = String::new();
        for _ in 0..self.model.config().max_target_positions {
            let tok_t = Tensor::new(tokens.as_slice(), mel.device())?.unsqueeze(0)?;
            let ys = self
                .model
                .decoder_forward(&tok_t, &audio_features, tokens.len() == 1)?;
            let (_, slen, _) = ys.dims3()?;
            let logits = self
                .model
                .decoder_final_linear(&ys.i((..1, slen - 1..))?)?
                .i(0)?
                .i(0)?;

            let logits = logits.broadcast_add(&self.suppress)?;
            let next = if temp == 0.0 {
                argmax(&logits)?
            } else {
                sample(&logits, temp, &mut self.rng)?
            };
            if next == self.eot {
                break;
            }
            tokens.push(next);
            let piece = self.tokenizer.decode(&[next], true)?;
            text.push_str(&piece);
        }
        Ok(text)
    }
}

/// deterministic arg-max
fn argmax(t: &Tensor) -> Result<u32> {
    let v: Vec<f32> = t.to_vec1()?;
    Ok(v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0 as u32)
}

/// temperature sampling
fn sample(t: &Tensor, temp: f64, rng: &mut StdRng) -> Result<u32> {
    let prs = softmax(&(t / temp)?, 0)?;
    let v: Vec<f32> = prs.to_vec1()?;
    let dist = WeightedIndex::new(v)?;
    Ok(dist.sample(rng) as u32)
}

/// Helper to fetch a special token id, with nicer error.
pub fn token_id(tok: &Tokenizer, s: &str) -> Result<u32> {
    tok.token_to_id(s)
        .ok_or_else(|| anyhow!("token {s} not found"))
}
