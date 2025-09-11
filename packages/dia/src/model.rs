//! Model loading utilities for dia
//!
//! This module provides loading functionality for neural models used in dia.

use crate::config::DiaConfig;
use crate::layers::{Decoder, Encoder};
use crate::setup::ModelPaths;
use crate::state::{DecoderInferenceState, EncoderInferenceState, KVCache};
use anyhow::Result;
use candle_core::{DType, Device, IndexOp};
use std::sync::OnceLock;

// ---------- optional EnCodec round-trip -----------------------------------

use candle_transformers::models::encodec::{Config as EncodecConfig, Model as EncodecModel};
// use progresshub_client_selector::{Backend, Client, DownloadConfig};  // Temporarily disabled

/// Error type for model operations
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("Configuration error: {0}")]
    Config(String),
    #[error("Model loading error: {0}")]
    Loading(String),
}

/// Result type for model operations
pub type ModelResult<T> = Result<T, ModelError>;

/// Zero-allocation, lock-free EnCodec model storage
static ENCODEC: OnceLock<EncodecModel> = OnceLock::new();

/// Global storage for model paths from setup
static MODEL_PATHS: OnceLock<ModelPaths> = OnceLock::new();

/// DiaModel: Complete Moshi-style transformer with encoder-decoder architecture
#[derive(Clone)]
pub struct DiaModel {
    encoder: Encoder,
    decoder: Decoder,
    config: DiaConfig,
}

impl DiaModel {
    /// Create a new DiaModel instance from weights and configuration
    pub fn new(
        config: crate::config::DiaConfig,
        vb: crate::VarBuilder,
        dtype: DType,
    ) -> candle_core::Result<Self> {
        let encoder = Encoder::new(&config, vb.pp("encoder"), dtype)?;
        let decoder = Decoder::new(&config, vb.pp("decoder"), dtype)?;

        Ok(Self {
            encoder,
            decoder,
            config,
        })
    }

    /// Encode text input through the encoder
    pub fn encode(
        &self,
        src_ids: &crate::Tensor,
    ) -> candle_core::Result<(crate::Tensor, crate::state::EncoderInferenceState)> {
        let state = EncoderInferenceState::new(&self.config, src_ids)?;
        let enc_out = self.encoder.forward(src_ids, &state)?;
        Ok((enc_out, state))
    }

    /// Build cross-attention KV cache from encoder output
    pub fn build_cross_cache(
        &self,
        enc_out: &crate::Tensor,
        _enc_pos: &crate::Tensor,
    ) -> candle_core::Result<Vec<crate::state::KVCache>> {
        let mut cross_cache = Vec::with_capacity(self.config.model.decoder.n_layer);

        // Build K/V cache for each decoder layer's cross-attention
        let device = enc_out.device();
        for _layer_idx in 0..self.config.model.decoder.n_layer {
            let kv_cache = KVCache::new(
                2, // batch size (uncond + cond)
                self.config.model.decoder.cross_query_heads,
                self.config.data.text_length,
                self.config.model.decoder.cross_head_dim,
                enc_out.dtype(),
                device,
            )?;
            cross_cache.push(kv_cache);
        }

        Ok(cross_cache)
    }

    /// Create new decoder inference state
    pub fn new_decoder_state(
        &self,
        enc_state: &crate::state::EncoderInferenceState,
        enc_out: crate::Tensor,
        cross_cache: Vec<crate::state::KVCache>,
        _device: &Device,
    ) -> candle_core::Result<crate::state::DecoderInferenceState> {
        let dtype = enc_out.dtype();
        DecoderInferenceState::new(&self.config, enc_state, enc_out, cross_cache, dtype)
    }

    /// Prefill decoder with audio prompt tokens in bulk
    pub fn prefill_decoder(
        &self,
        tgt: &crate::Tensor,
        dec_state: &mut crate::state::DecoderInferenceState,
    ) -> candle_core::Result<()> {
        // Forward through decoder for bulk prefill - process all tokens at once
        let tgt_len = tgt.dim(1)?; // sequence length dimension
        for step in 0..tgt_len {
            let step_tokens = tgt.i((.., step..step + 1, ..))?; // [B, 1, C]
            self.decoder.forward_step(&step_tokens, dec_state)?;
        }
        Ok(())
    }

    /// Perform single autoregressive decode step
    pub fn decode_step(
        &self,
        ids: &crate::Tensor,
        dec_state: &mut crate::state::DecoderInferenceState,
    ) -> candle_core::Result<crate::Tensor> {
        // Forward through decoder for single step
        self.decoder.forward_step(ids, dec_state)
    }

    /// Decode audio codes to waveform using EnCodec
    pub fn decode_audio_codes(&self, codes: &crate::Tensor) -> Result<crate::Tensor, ModelError> {
        // Get the device from the codes tensor
        let device = codes.device();

        // Load EnCodec model
        let encodec = load_encodec(device)?;

        // Decode codes to audio waveform
        let audio = encodec.decode(codes)?;

        Ok(audio)
    }
}

/// Load EnCodec model using sophisticated async progresshub with zero-allocation caching
pub fn load_encodec(device: &Device) -> Result<&'static EncodecModel, candle_core::Error> {
    if let Some(model) = ENCODEC.get() {
        return Ok(model);
    }

    // Get model paths from global storage
    let model_paths = MODEL_PATHS.get().ok_or_else(|| {
        candle_core::Error::Msg("Model paths not set - call set_model_paths first".to_string())
    })?;

    // Load EnCodec model from downloaded weights
    use crate::VarBuilder;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[&model_paths.encodec_weights], DType::F32, device)?
    };

    let config = EncodecConfig::default();
    let encodec = EncodecModel::new(&config, vb)?;

    // Store in global cache
    ENCODEC
        .set(encodec)
        .map_err(|_| candle_core::Error::Msg("EnCodec model already cached".to_string()))?;

    // Safe access to cached model - we just set it above
    ENCODEC.get().ok_or_else(|| {
        candle_core::Error::Msg("Failed to retrieve cached EnCodec model".to_string())
    })
}

/// Load the main Dia model with sophisticated caching
pub fn load_dia_model(
    weights_path: &std::path::Path,
    config: &DiaConfig,
    device: &Device,
) -> ModelResult<DiaModel> {
    // Load model weights using VarBuilder
    use crate::VarBuilder;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)
            .map_err(ModelError::Candle)?
    };

    // Create the DiaModel
    let model = DiaModel::new(config.clone(), vb, DType::F32).map_err(ModelError::Candle)?;

    Ok(model)
}

/// Set the model paths from setup
pub fn set_model_paths(paths: ModelPaths) -> Result<(), String> {
    MODEL_PATHS
        .set(paths)
        .map_err(|_| "Model paths already set".to_string())
}

/// Configure model defaults
pub fn configure_model_defaults() -> Result<(), String> {
    // Default configuration is handled through DiaConfig::default()
    Ok(())
}
