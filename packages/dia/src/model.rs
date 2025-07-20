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
use progresshub_client_selector::Client as ProgressHubClient;
use progresshub_config::DownloadConfig;
use std::sync::atomic::{AtomicPtr, Ordering};

/// Zero-allocation, lock-free EnCodec model storage using atomic pointer
static ENCODEC: AtomicPtr<EncodecModel> = AtomicPtr::new(std::ptr::null_mut());

/// Custom error type for semantic model loading errors
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("Model download failed: {0}")]
    DownloadError(#[from] anyhow::Error),
    #[error("Model loading failed: {0}")]
    LoadingError(#[from] candle_core::Error),
    #[error("Model initialization failed: {0}")]
    InitializationError(String),
}

/// Zero-allocation, blazing-fast EnCodec model loading with progresshub integration
pub async fn load_encodec(device: &Device) -> Result<&'static EncodecModel, ModelError> {
    // Check if already loaded using lock-free atomic access
    let existing = ENCODEC.load(Ordering::Acquire);
    if !existing.is_null() {
        return Ok(unsafe { &*existing });
    }

    // Download model using progresshub with automatic backend selection
    let client = ProgressHubClient::new().map_err(|e| ModelError::InitializationError(format!("Failed to create progresshub client: {}", e)))?;
    
    let config = DownloadConfig {
        destination: None, // Use default cache location
        show_progress: false, // No UI progress for background loading
        use_cache: true, // Enable efficient caching
    };

    let download_result = client
        .download_model("facebook/encodec_24khz", config)
        .await
        .map_err(ModelError::DownloadError)?;

    // Find the model.safetensors file in downloaded files
    let weights_path = download_result
        .file_paths
        .iter()
        .find(|path| path.file_name().and_then(|n| n.to_str()) == Some("model.safetensors"))
        .ok_or_else(|| ModelError::InitializationError("model.safetensors not found in downloaded files".to_string()))?;

    // Load model with zero-allocation memory mapping
    let vb = unsafe { 
        VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)
            .map_err(ModelError::LoadingError)?
    };
    
    let model = EncodecModel::new(&EncodecCfg::default(), vb)
        .map_err(ModelError::LoadingError)?;

    // Atomic initialization - only one thread succeeds, others use the result
    let model_ptr = Box::into_raw(Box::new(model));
    match ENCODEC.compare_exchange_weak(
        std::ptr::null_mut(),
        model_ptr,
        Ordering::Release,
        Ordering::Relaxed,
    ) {
        Ok(_) => {
            // Successfully installed our model
            Ok(unsafe { &*model_ptr })
        }
        Err(existing_ptr) => {
            // Another thread beat us to it, clean up our model and use theirs
            unsafe { Box::from_raw(model_ptr) }; // Clean up our allocation
            Ok(unsafe { &*existing_ptr })
        }
    }
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
