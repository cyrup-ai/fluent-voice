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

#[cfg(any(feature = "cuda", feature = "metal", feature = "accelerate", feature = "mkl"))]
use candle_core::{DType, Device, Tensor};
#[cfg(any(feature = "cuda", feature = "metal", feature = "accelerate", feature = "mkl"))]
use candle_nn::VarBuilder;

use crate::{
    config::DiaConfig,
    layers::{Decoder, Encoder},
    state::{DecoderInferenceState, EncoderInferenceState, KVCache},
};

// ---------- optional EnCodec round-trip -----------------------------------

use candle_transformers::models::encodec::{Config as EncodecCfg, Model as EncodecModel};
use once_cell::sync::OnceCell;

/// Lazily-initialised global EnCodec (24 kHz, f32 weights – ~15 MB).
static ENCODEC: OnceCell<Arc<EncodecModel>> = OnceCell::new();

fn load_encodec(device: &Device) -> candle_core::Result<&'static EncodecModel> {
    ENCODEC
        .get_or_try_init(|| {
            use hf_hub::api::sync::Api;
            let api =
                Api::new().map_err(|e| candle_core::Error::Msg(format!("HF API error: {}", e)))?;
            let weights = api
                .model("facebook/encodec_24khz".to_string())
                .get("model.safetensors")
                .map_err(|e| candle_core::Error::Msg(format!("EnCodec download error: {}", e)))?;

            let vb =
                unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, device)? };
            Ok(Arc::new(EncodecModel::new(&EncodecCfg::default(), vb)?))
        })
        .map(|arc| arc.as_ref())
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
    pub fn decode_audio_codes(&self, codes_btc: &Tensor) -> candle_core::Result<Tensor> {
        // transpose → [B,C,T] as expected by EnCodec
        let codes = codes_btc.transpose(1, 2)?;
        let model = load_encodec(&codes.device())?;
        model.decode(&codes)
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
