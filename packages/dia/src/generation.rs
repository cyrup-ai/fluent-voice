//! Autoregressive generation utilities for Dia-Voice.
//!
//! This file owns **all high-level sampling logic**: encoder pass,
//! decoder pre-fill, classifier-free-guided decode loop, top-k / top-p
//! nucleus sampling and temperature scaling.  The low-level model
//! building blocks live in `layers.rs`; the “wiring” between encoder
//! and decoder lives in `model.rs`.

use crate::{DType, Device, Tensor, ops};
use anyhow::{Result, bail};
use candle_core::IndexOp;

use rand::distr::weighted::WeightedIndex;
use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::{
    audio::{delayed_view, undelayed_view},
    config::DiaConfig,
    model::DiaModel,
    state::{DecoderInferenceState, DecoderOutput},
};

/// Configuration for decoder step parameters
#[derive(Debug, Clone)]
pub struct DecoderStepConfig {
    pub cfg_scale: f64,
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
}

/// Configuration for generation parameters
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_tokens: usize,
    pub seed: u64,
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
    pub cfg_scale: f64,
}

// Import optimizations for GPU acceleration
#[cfg(any(feature = "cuda", feature = "metal"))]
use crate::optimizations::{GpuConfig, get_compute_dtype, get_optimal_config};

/// Convenience: extremely small ε to replace `-inf` when we build masks.
const NEG_INF: f32 = -1e30;

// ==========================================================================
// Top-k / Top-p (“nucleus”) filtering  –  runs fully on device
// ==========================================================================

/// Apply *top-k* and/or *top-p* (**nucleus**) filtering **in log-prob space**.
///
/// The function takes a **1-D logits tensor** of shape `[V]` and returns a
/// *masked* tensor of identical shape where every position that falls outside
/// the selected nucleus is set to `NEG_INF`.
fn topk_topp_mask(logits: &Tensor, top_k: usize, top_p: f64) -> Result<Tensor> {
    let vdim = logits.dim(0)?; // vocabulary size
    if vdim == 0 {
        bail!("empty logits tensor passed to top-k/p mask");
    }

    // Start with an unmodified clone.  We will progressively set positions to
    // `-inf` (NEG_INF) so they are ignored by the subsequent softmax.
    let device = logits.device();
    let mut masked = logits.clone();

    // ---------- Top-k ----------------------------------------------------
    if top_k > 0 && top_k < vdim {
        // In Candle 0.9 there is no built-in `topk` yet so we emulate it via
        // repeated argmax() + masking.  This is O(K·V) but K is tiny in
        // practice (<=128) so the extra cost is negligible compared to
        // attention.
        let mut keep_mask = Tensor::zeros(&[vdim], DType::U8, device)?; // bool mask
        let mut tmp = logits.clone();

        for _ in 0..top_k {
            // Find arg-max.
            let idx_t = tmp.argmax(0)?; // scalar tensor
            let idx = idx_t.to_scalar::<u32>()? as usize;

            // Mark as “keep”.
            let one = Tensor::full(1u8, &[1], device)?;
            keep_mask = keep_mask.scatter_add(
                &Tensor::from_slice(&[idx as u32], &[1], device)?,
                &one,
                0,
            )?;

            // Mask out this position so it is not found again.
            tmp = tmp.scatter_add(
                &Tensor::from_slice(&[idx as u32], &[1], device)?,
                &Tensor::full(NEG_INF, &[1], device)?,
                0,
            )?;
        }

        // Everything not marked gets NEG_INF.
        let neginf = Tensor::full(NEG_INF, &[vdim], device)?;
        masked = Tensor::where_cond(&keep_mask.to_dtype(DType::U8)?, &masked, &neginf)?;
    }

    // ---------- Top-p ----------------------------------------------------
    if (0.0..1.0).contains(&top_p) {
        // Convert to probabilities (softmax) so we can compute the cumulative
        // mass.  We work on the *already top-k masked* logits so the two
        // filters compose correctly.
        let probs = ops::softmax(&masked, 0)?; // [V]

        // Sadly there is no `sort()` yet either – emulate via argsort.
        // 1. Pull probabilities to host, build an argsort of indices.
        let pvec = probs.to_vec1::<f32>()?;
        let mut indices: Vec<usize> = (0..vdim).collect();
        indices.sort_by(|&a, &b| {
            pvec[b]
                .partial_cmp(&pvec[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        }); // descending

        // 2. Compute cumulative probability until we cross `top_p`.
        let mut cum = 0f32;
        let mut keep = vec![false; vdim];
        for &i in &indices {
            cum += pvec[i];
            keep[i] = true;
            if cum as f64 >= top_p {
                break;
            }
        }

        // 3. Build keep-mask tensor.
        let keep_u8: Vec<u8> = keep.iter().map(|&k| if k { 1 } else { 0 }).collect();
        let keep_mask = Tensor::from_vec(keep_u8, (vdim,), device)?;
        let neginf = Tensor::full(NEG_INF, &[vdim], device)?;
        masked = Tensor::where_cond(&keep_mask, &masked, &neginf)?;
    }

    Ok(masked)
}

// ==========================================================================
// Dia-TTS wrapper  (encoder → decoder  +  sampling helpers)
// ==========================================================================

/// A minimal run-time wrapper around the Dia encoder/decoder stacks plus
/// sampling utilities with optional GPU optimizations.
pub struct DiaTts {
    model: DiaModel,
    cfg: DiaConfig,
    device: Device,
    #[cfg(any(feature = "cuda", feature = "metal"))]
    _gpu_config: GpuConfig,
    #[cfg(any(feature = "cuda", feature = "metal"))]
    _compute_dtype: DType,
}

impl DiaTts {
    pub fn new(model: DiaModel, cfg: DiaConfig, device: Device) -> Self {
        #[cfg(any(feature = "cuda", feature = "metal"))]
        let gpu_config = get_optimal_config(&device);

        #[cfg(any(feature = "cuda", feature = "metal"))]
        let compute_dtype = get_compute_dtype(&device, gpu_config.mixed_precision);

        #[cfg(any(feature = "cuda", feature = "metal"))]
        if matches!(device, Device::Cuda(_) | Device::Metal(_)) {
            tracing::info!("GPU Optimizations enabled:");
            tracing::info!(device = ?device, "  - Device");
            tracing::info!(
                mixed_precision = gpu_config.mixed_precision,
                "  - Mixed precision"
            );
            tracing::info!(dtype = ?compute_dtype, "  - Compute dtype");
        }

        Self {
            model,
            cfg,
            device,
            #[cfg(any(feature = "cuda", feature = "metal"))]
            _gpu_config: gpu_config,
            #[cfg(any(feature = "cuda", feature = "metal"))]
            _compute_dtype: compute_dtype,
        }
    }

    // ---------------------------------------------------------------------
    // One *classifier-free guided* decoder step (B = 2, seq = 1)
    // ---------------------------------------------------------------------
    fn decoder_step(
        &self,
        tokens_b1c: &Tensor, // [2,1,C]
        state: &mut DecoderInferenceState,
        config: &DecoderStepConfig,
        sampler: &mut StdRng,
    ) -> Result<Tensor> {
        // Forward pass through the decoder.  Shape: [2,1,C,V]
        let logits = self.model.decode_step(tokens_b1c, state)?;
        let logits = logits.squeeze(1)?; // [2,C,V]

        // --------- CFG blend  --------------------------------------------
        let cond = logits.i((1, .., ..))?; // [C,V]
        let uncond = logits.i((0, .., ..))?; // [C,V]
        let guided = (&uncond + ((&cond - &uncond)? * config.cfg_scale)?)?; // [C,V]

        // --------- Mask tokens > EOS for non-first channels --------------
        let eos = self.cfg.data.audio_eos_value as usize;
        let mut guided = guided; // make mutable
        if eos + 1 < guided.dim(1)? {
            let before = guided.i((.., ..eos + 1))?;
            let pad = Tensor::full(
                NEG_INF,
                (guided.dim(0)?, guided.dim(1)? - (eos + 1)),
                &self.device,
            )?;
            guided = Tensor::cat(&[&before, &pad], 1)?;
        }

        // --------- Per-channel sampling ----------------------------------
        let mut next = Vec::with_capacity(self.cfg.data.channels);
        for c in 0..guided.dim(0)? {
            let mut logit = guided.i((c, ..))?; // [V]

            // Temperature scaling.
            if config.temperature > 0.0 {
                logit = (&logit / config.temperature)?;
            }

            // Top-k / Top-p nucleus filtering.
            logit = topk_topp_mask(&logit, config.top_k, config.top_p)?;

            // Softmax → probabilities.
            let probs = ops::softmax(&logit, 0)?;
            let probs_v = probs.to_vec1::<f32>()?;
            let distr = WeightedIndex::new(&probs_v)?;
            // Use the Distribution trait method correctly
            let idx = sampler.sample(&distr) as u32;

            next.push(idx);
        }

        Ok(Tensor::from_vec(
            next,
            (1, self.cfg.data.channels),
            &self.device,
        )?)
    }

    // ---------------------------------------------------------------------
    // High-level helper – BOS / prompt, encoder pass, decode loop
    // ---------------------------------------------------------------------
    pub fn generate_speech(
        &self,
        text_ids: &Tensor,             // [S] already tokenised (conditional)
        audio_prompt: Option<&Tensor>, // [T,C] optional prompt codes
        config: &GenerationConfig,
    ) -> Result<Tensor> {
        // --------- Encoder pass -----------------------------------------
        let (enc_out, enc_state) = self.model.encode(text_ids)?;
        let cross_cache = self
            .model
            .build_cross_cache(&enc_out, &enc_state.positions)?;
        let mut dec_state =
            self.model
                .new_decoder_state(&enc_state, enc_out, cross_cache, &self.device)?;

        // --------- BOS + optional prompt pre-fill -----------------------
        let bos = self.cfg.data.audio_bos_value;
        let mut generated = DecoderOutput::new(&self.cfg, &self.device)?;
        let mut cur = 0usize;

        if let Some(prompt) = audio_prompt {
            let prompt_len = prompt.dim(0)?;
            let bos_row = Tensor::full(bos, (1, self.cfg.data.channels), &self.device)?;
            let prefill = Tensor::cat(&[&bos_row, prompt], 0)?; // prepend BOS

            // Apply channel delays before prefilling
            let prefill_delayed = delayed_view(&prefill, self.cfg.data.audio_pad_value)?;

            let pref_batched = prefill_delayed.unsqueeze(0)?; // [1,T,C]
            let pref_dual = Tensor::cat(&[&pref_batched; 2], 0)?; // [2,T,C]

            self.model.prefill_decoder(&pref_dual, &mut dec_state)?;
            generated.prefill(prefill, prompt_len + 1)?;
            cur = prompt_len + 1;
        } else {
            let bos_tok = Tensor::full(bos, (1, 1, self.cfg.data.channels), &self.device)?; // [1,1,C]
            let bos_dual = Tensor::cat(&[&bos_tok; 2], 0)?; // [2,1,C]
            self.model.prefill_decoder(&bos_dual, &mut dec_state)?;
        }

        // --------- Autoregressive loop ----------------------------------
        let mut rng = StdRng::seed_from_u64(config.seed);
        while cur < config.max_tokens && cur < self.cfg.data.audio_length {
            dec_state.prepare_step(cur);

            // Last generated token (or BOS at cur==0) duplicated for CFG.
            let prev = if cur == 0 {
                Tensor::full(bos, (1, 1, self.cfg.data.channels), &self.device)?
            } else {
                generated.get_tokens_at(cur - 1, cur)?.unsqueeze(0)?
            };
            let cond_uncond = Tensor::cat(&[&prev; 2], 0)?; // [2,1,C]

            // Update cross-attention mask based on current tokens
            dec_state.update_cross_attention_mask(&cond_uncond, &self.cfg, &enc_state)?;

            let step_config = DecoderStepConfig {
                cfg_scale: config.cfg_scale,
                temperature: config.temperature,
                top_p: config.top_p,
                top_k: config.top_k,
            };
            let next = self.decoder_step(&cond_uncond, &mut dec_state, &step_config, &mut rng)?; // [1,C]
            generated.update_one(next.squeeze(0)?, cur, false)?;
            cur += 1;

            // Early-stop if **every** channel produced EOS.
            let eos_mask = generated
                .get_tokens_at(cur - 1, cur)?
                .eq(bos + 1)? // EOS value = BOS+1 by spec
                .sum_all()?
                .to_scalar::<u32>()?;
            if eos_mask as usize == self.cfg.data.channels {
                break;
            }
        }

        // Remove channel delays before returning tokens for decoding
        let final_tokens =
            undelayed_view(&generated.generated_tokens, self.cfg.data.audio_pad_value)?;

        Ok(final_tokens)
    }
}
