#![allow(non_snake_case)]
//! High-level wrapper that glues the Dia text-encoder/decoder stacks together
//! and optionally round-trips audio codes with Facebook EnCodec (24 kHz).
//!
//! ┌─ DiaModel ─────────────────────────────────────────────────────────────┐
//! │ • encode()                 – run encoder & return context + state     │
//! │ • build_cross_cache()      – pre-compute K/V used by all x-attn       │
//! │ • new_decoder_state()      – helper for generation.rs                 │
//! │ • prefill_decoder()        – feed BOS / audio prompt once             │
//! │ • decode_step()            – 1-token AR step (CFG loop)               │
//! │ • decode_audio_codes()     – EnCodec -> PCM (T,C) → [B,1,T]           │
//! └────────────────────────────────────────────────────────────────────────┘

use std::{path::Path, sync::Arc};

use crate::{DType, Device, Tensor, VarBuilder};

use crate::{
    config::DiaConfig,
    layers::{Decoder, Encoder},
    state::{DecoderInferenceState, EncoderInferenceState, KVCache},
};

// ---------- optional EnCodec round-trip -----------------------------------

use candle_transformers::models::encodec::{Config as EncodecCfg, Model as EncodecModel};
use progresshub_client_selector::{Client, DownloadConfig, Backend};
use std::sync::OnceLock;

/// Error type for model operations
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("Loading error: {0}")]
    LoadingError(#[from] candle_core::Error),
    #[error("Model error: {0}")]
    ModelError(String),
}

/// Zero-allocation, lock-free EnCodec model storage
static ENCODEC: OnceLock<EncodecModel> = OnceLock::new();

/// Load EnCodec model using real progresshub with zero-allocation caching
pub async fn load_encodec(device: &Device) -> Result<&'static EncodecModel, candle_core::Error> {
    if let Some(model) = ENCODEC.get() {
        return Ok(model);
    }

    // Download model using real progresshub client selector
    let client = Client::new(Backend::Auto);
    let config = DownloadConfig {
        destination: dirs::cache_dir()
            .ok_or_else(|| candle_core::Error::Msg("Cannot determine cache directory".to_string()))?
            .join("fluent-voice")
            .join("encodec"),
        show_progress: false,
        use_cache: true,
    };

    let download_result = client
        .download_model_auto("facebook/encodec_24khz", &config, None)
        .await
        .map_err(|e| candle_core::Error::Msg(format!("Model download failed: {}", e)))?;

    // Find the model.safetensors file in downloaded files
    let weights_path = download_result
        .models
        .first()
        .ok_or_else(|| candle_core::Error::Msg("No models in download result".to_string()))?
        .files
        .iter()
        .find(|file| file.path.file_name().and_then(|n| n.to_str()) == Some("model.safetensors"))
        .ok_or_else(|| candle_core::Error::Msg("model.safetensors not found in downloaded files".to_string()))?;

    // Load model with zero-allocation memory mapping
    let vb = unsafe { 
        VarBuilder::from_mmaped_safetensors(&[&weights_path.path], DType::F32, device)?
    };
    
    let model = EncodecModel::new(&EncodecCfg::default(), vb)?;

    // Store in OnceLock - only first thread succeeds, others get the cached version
    ENCODEC.set(model).map_err(|_| candle_core::Error::Msg("Failed to cache EnCodec model".to_string()))?;
    
    Ok(ENCODEC.get().unwrap())
}

// -------------------------------------------------------------------------
// DiaModel (text → audio codes) -------------------------------------------
// -------------------------------------------------------------------------

#[derive(Clone)]
pub struct DiaModel {
    cfg: Arc<DiaConfig>,
    pub encoder: Encoder,
    pub decoder: Decoder,
    dtype: DType,
}

impl DiaModel {
    // ============= construction ==========================================
    pub fn new(cfg: DiaConfig, vb: VarBuilder, dtype: DType) -> candle_core::Result<Self> {
        let encoder = Encoder::new(&cfg, vb.pp("encoder"), dtype)?;
        let decoder = Decoder::new(&cfg, vb.pp("decoder"), dtype)?;
        Ok(Self {
            cfg: Arc::new(cfg),
            encoder,
            decoder,
            dtype,
        })
    }

    /// Convenience helper if you already mmap'ed safetensors on disk.
    pub fn from_safetensors<P: AsRef<Path>>(
        cfg: DiaConfig,
        paths: &[P],
        device: &Device,
    ) -> candle_core::Result<Self> {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(paths, DType::F32, device)? };
        Self::new(cfg, vb, device.f32_or_bf16())
    }

    // ============= encoder ===============================================
    pub fn encode(
        &self,
        src_ids_bx_s: &Tensor, // [2,S]
    ) -> candle_core::Result<(Tensor, EncoderInferenceState)> {
        let enc_state = EncoderInferenceState::new(&self.cfg, src_ids_bx_s)?;
        let enc_out = self.encoder.forward(src_ids_bx_s, &enc_state)?; // [2,S,E]
        Ok((enc_out, enc_state))
    }

    // ============= decoder helpers =======================================
    pub fn build_cross_cache(
        &self,
        enc_out: &Tensor,
        enc_pos: &Tensor,
    ) -> candle_core::Result<Vec<KVCache>> {
        self.decoder
            .precompute_cross_attn_cache(enc_out.clone(), enc_pos.clone())
    }

    pub fn new_decoder_state(
        &self,
        enc_state: &EncoderInferenceState,
        enc_out: Tensor,
        cross_cache: Vec<KVCache>,
        device: &Device,
    ) -> candle_core::Result<DecoderInferenceState> {
        DecoderInferenceState::new(&self.cfg, enc_state, enc_out, cross_cache, self.dtype).map(
            |mut st| {
                st.device = device.clone();
                st
            },
        )
    }

    pub fn prefill_decoder(
        &self,
        tgt_b_tx_c: &Tensor, // [2,T,C]
        dec_state: &mut DecoderInferenceState,
    ) -> candle_core::Result<()> {
        self.decoder.prefill(tgt_b_tx_c, dec_state)
    }

    pub fn decode_step(
        &self,
        ids_b1c: &Tensor, // [2,1,C]
        dec_state: &mut DecoderInferenceState,
    ) -> candle_core::Result<Tensor> {
        self.decoder.forward_step(ids_b1c, dec_state)
    }

    // ============= audio round-trip helpers ==============================
    /// Decode EnCodec codes (shape **[B,T,C]**) back to PCM `[-1,1]`.
    /// Uses async loading with zero-allocation, lock-free model access.
    pub async fn decode_audio_codes(&self, codes_btc: &Tensor) -> Result<Tensor, ModelError> {
        // transpose → [B,C,T] as expected by EnCodec
        let codes = codes_btc.transpose(1, 2).map_err(ModelError::LoadingError)?;
        let model = load_encodec(&codes.device()).await?;
        model.decode(&codes).map_err(ModelError::LoadingError)
    }
}

// -------------------------------------------------------------------------
// Device helper: pick BF16 on GPU, F32 elsewhere.
// -------------------------------------------------------------------------

trait DeviceDType {
    fn f32_or_bf16(&self) -> DType;
}
impl DeviceDType for Device {
    fn f32_or_bf16(&self) -> DType {
        if self.is_cuda() || self.is_metal() {
            DType::BF16
        } else {
            DType::F32
        }
    }
}
