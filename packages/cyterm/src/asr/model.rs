//! Thin wrapper around Candle Whisper variants (fp32 + gguf‐quantized).

use candle_core::{Result, Tensor};
use candle_transformers::models::whisper::{self as m, Config};

/// Either an ordinary fp32 Whisper or a quantized gguf snapshot.
pub enum WhisperModel {
    Normal(m::model::Whisper),
    Quantized(m::quantized_model::Whisper),
}

impl WhisperModel {
    #[inline]
    pub fn config(&self) -> &Config {
        match self {
            Self::Normal(m) => &m.config,
            Self::Quantized(m) => &m.config,
        }
    }

    #[inline]
    pub fn encoder_forward(&mut self, x: &Tensor, flush: bool) -> Result<Tensor> {
        match self {
            Self::Normal(m) => m.encoder.forward(x, flush),
            Self::Quantized(m) => m.encoder.forward(x, flush),
        }
    }

    #[inline]
    pub fn decoder_forward(&mut self, x: &Tensor, xa: &Tensor, flush: bool) -> Result<Tensor> {
        match self {
            Self::Normal(m) => m.decoder.forward(x, xa, flush),
            Self::Quantized(m) => m.decoder.forward(x, xa, flush),
        }
    }

    #[inline]
    pub fn decoder_final_linear(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Normal(m) => m.decoder.final_linear(x),
            Self::Quantized(m) => m.decoder.final_linear(x),
        }
    }

    #[inline]
    pub fn reset_kv_cache(&mut self) {
        match self {
            Self::Normal(m) => m.reset_kv_cache(),
            Self::Quantized(m) => m.reset_kv_cache(),
        }
    }
}
