//! Runtime data-structures used **during inference**
//! (KV-cache, encoder / decoder state, etc.)

use crate::{CandleResult, DType, Device, Tensor};
use candle_core::IndexOp;

use crate::config::DiaConfig;

// ==========================================================================
// helpers
// ==========================================================================

/// Build an attention mask that mimics the original JAX “segment-ID” logic.
///
/// * `q_pad`, `k_pad` – padding masks *(true = real token)*.
/// * if `causal` is **true**, a causal triangle is AND-ed in.
pub fn make_mask(q_pad: &Tensor, k_pad: &Tensor, causal: bool) -> CandleResult<Tensor> {
    let (_b, tq) = (q_pad.dim(0)?, q_pad.dim(1)?);
    let tk = k_pad.dim(1)?;

    // broadcast  -> [B, Tq, Tk]
    let q = q_pad.unsqueeze(2)?; // [B, Tq, 1]
    let k = k_pad.unsqueeze(1)?; // [B, 1 , Tk]
    // Using tensor operations instead of ! operator
    // For boolean tensors in candle, we use multiplication for logical AND
    // and addition for logical OR
    let q_and_k = q.mul(&k)?;

    // For logical NOT, use tensor.eq(&tensor.zeros_like()?) as documented in FIXES.md
    let not_q = q.eq(&q.zeros_like()?)?;
    let not_k = k.eq(&k.zeros_like()?)?;

    // !q & !k: both false elements using multiplication for AND
    let not_q_and_not_k = not_q.mul(&not_k)?;

    // (q & k) | (!q & !k): either both true or both false using addition for OR
    let mask = q_and_k.add(&not_q_and_not_k)?; // compatible padding

    if causal {
        if tq != tk {
            // Fix error type to match candle_core::Result
            return Err(candle_core::Error::Msg(
                "causal mask requires square (Tq==Tk)".to_string(),
            ));
        }
        // lower-triangular causal part using row/column indices comparison
        let tril_indices = Tensor::arange(0, tk as u32, q_pad.device())?.unsqueeze(0)?;
        let tril_indices = tril_indices.expand(&[tq, tk])?;
        let row_indices = Tensor::arange(0, tq as u32, q_pad.device())?.unsqueeze(1)?;
        let row_indices = row_indices.expand(&[tq, tk])?;
        let tril = tril_indices.le(&row_indices)?; // This creates a boolean mask

        // Use mul for logical AND operation - clone tril to make it owned (Tensor instead of &Tensor)
        let masked = mask.mul(&tril)?;
        Ok(masked.unsqueeze(1)?) // [B, 1, Tq, Tk]
    } else {
        Ok(mask.unsqueeze(1)?) // [B, 1, Tq, Tk]
    }
}

// ==========================================================================
// Encoder-side state
// ==========================================================================

pub struct EncoderInferenceState {
    /// `[B, S]` absolute positions (uncond *and* cond ⇒ B = 2)
    pub positions: Tensor,
    /// padding mask `[B, S]` (*true = token present*)
    pub padding_mask: Tensor,
    /// attention mask `[B, 1, S, S]`
    pub attn_mask: Tensor,
}

impl EncoderInferenceState {
    /// Build once per request (after text pre-processing).
    pub fn new(cfg: &DiaConfig, src_cond: &Tensor) -> CandleResult<Self> {
        let dev = src_cond.device();
        let s = cfg.data.text_length;

        // positions 0‥S-1  (broadcast to both uncond/cond)
        let positions = Tensor::arange(0, s as u32, dev)?.expand(&[2, s])?; // <-- no “broadcast_in_dim” flag

        // padding mask – cond row is real; uncond all-zeros
        let padding_mask = src_cond.ne(cfg.data.text_pad_value)?.expand(&[2, s])?;

        let attn_mask = make_mask(&padding_mask, &padding_mask, false)?.unsqueeze(1)?; // [B,1,S,S]

        Ok(Self {
            positions,
            padding_mask,
            attn_mask,
        })
    }
}

// ==========================================================================
// Key/value cache (per transformer layer)
// ==========================================================================

#[derive(Clone, Debug)]
pub struct KVCache {
    /// K/V: `[B, heads, S_max, head_dim]`
    pub k: Tensor,
    pub v: Tensor,
    cur: usize,
}

impl KVCache {
    pub fn new(
        b: usize,
        heads: usize,
        s_max: usize,
        head_dim: usize,
        dtype: DType,
        dev: &Device,
    ) -> CandleResult<Self> {
        let zeros = Tensor::zeros(&[b, heads, s_max, head_dim], dtype, dev)?;
        Ok(Self {
            k: zeros.clone(),
            v: zeros,
            cur: 0,
        })
    }

    /// Append one step (called at decode time).
    pub fn update(&mut self, k: Tensor, v: Tensor) -> candle_core::Result<(Tensor, Tensor)> {
        let len = k.dim(2)?;

        // Advanced approach: update cache by directly modifying the slice at current position
        // Use narrow to update the specific range in the cache
        let start = self.cur;
        let end = start + len;

        // Extract the relevant slice from the cache and replace it
        if end <= self.k.dim(2)? {
            // Create new tensors by concatenating: [before] + [new_data] + [after]
            let before_k = if start > 0 {
                Some(self.k.narrow(2, 0, start)?)
            } else {
                None
            };
            let after_k = if end < self.k.dim(2)? {
                Some(self.k.narrow(2, end, self.k.dim(2)? - end)?)
            } else {
                None
            };

            let before_v = if start > 0 {
                Some(self.v.narrow(2, 0, start)?)
            } else {
                None
            };
            let after_v = if end < self.v.dim(2)? {
                Some(self.v.narrow(2, end, self.v.dim(2)? - end)?)
            } else {
                None
            };

            // Concatenate the parts for k
            let new_k = match (before_k, after_k) {
                (Some(before), Some(after)) => Tensor::cat(&[&before, &k, &after], 2)?,
                (Some(before), None) => Tensor::cat(&[&before, &k], 2)?,
                (None, Some(after)) => Tensor::cat(&[&k, &after], 2)?,
                (None, None) => k.clone(),
            };

            // Concatenate the parts for v
            let new_v = match (before_v, after_v) {
                (Some(before), Some(after)) => Tensor::cat(&[&before, &v, &after], 2)?,
                (Some(before), None) => Tensor::cat(&[&before, &v], 2)?,
                (None, Some(after)) => Tensor::cat(&[&v, &after], 2)?,
                (None, None) => v.clone(),
            };

            self.k = new_k;
            self.v = new_v;
        } else {
            // If we're at the end, just concatenate
            self.k = Tensor::cat(&[&self.k, &k], 2)?;
            self.v = Tensor::cat(&[&self.v, &v], 2)?;
        }

        self.cur += len;
        Ok((self.k.clone(), self.v.clone()))
    }

    /// Bulk pre-fill (encoder-decoder cross-attn or pre-roll for streaming).
    pub fn prefill(&mut self, k: Tensor, v: Tensor) -> candle_core::Result<()> {
        let len = k.dim(2)?;

        // Advanced approach: just replace the beginning of the cache with the new data
        if len <= self.k.dim(2)? {
            // Replace the first 'len' positions with the new k,v data
            let remaining_k = if len < self.k.dim(2)? {
                Some(self.k.narrow(2, len, self.k.dim(2)? - len)?)
            } else {
                None
            };
            let remaining_v = if len < self.v.dim(2)? {
                Some(self.v.narrow(2, len, self.v.dim(2)? - len)?)
            } else {
                None
            };

            self.k = if let Some(remaining) = remaining_k {
                Tensor::cat(&[&k, &remaining], 2)?
            } else {
                k
            };

            self.v = if let Some(remaining) = remaining_v {
                Tensor::cat(&[&v, &remaining], 2)?
            } else {
                v
            };
        } else {
            // If incoming data is larger than cache, just take the first part
            self.k = k.narrow(2, 0, self.k.dim(2)?)?;
            self.v = v.narrow(2, 0, self.v.dim(2)?)?;
        }

        self.cur = len;
        Ok(())
    }

    /// Create a cache from pre-computed key and value tensors (typically for
    /// cross-attention). The sequence cursor starts at zero so that the first
    /// `update()` will write from the beginning of the cache.
    pub fn from_tensors(k: Tensor, v: Tensor) -> Self {
        Self { k, v, cur: 0 }
    }
}

// ==========================================================================
// Decoder-side state
// ==========================================================================

pub struct DecoderInferenceState {
    pub device: Device,
    pub dtype: DType,

    // ---------- encoder context ----------
    pub enc_out: Tensor,       // [B, S, E]
    pub enc_positions: Tensor, // [B, S]

    // ---------- decoding positions ----------
    pub dec_positions: Tensor, // [B, 1]

    // ---------- masks ----------
    pub dec_cross_attn_mask: Tensor, // [B,1,1,S]

    // ---------- per-layer caches ----------
    pub self_attn_cache: Vec<KVCache>,
    pub cross_attn_cache: Vec<KVCache>,
}

impl DecoderInferenceState {
    /// Build once after the encoder pass.
    pub fn new(
        cfg: &DiaConfig,
        enc_state: &EncoderInferenceState,
        enc_out: Tensor,
        cross_cache: Vec<KVCache>,
        dtype: DType,
    ) -> CandleResult<Self> {
        let dev = enc_out.device().clone();

        // start at position 0
        let dec_positions = Tensor::zeros(&[2, 1], DType::U32, &dev)?;

        // dummy “all real” mask for the first token
        // For CFG: row 0 (unconditional) = all padding, row 1 (conditional) = real BOS
        let tgt_pad = Self::generate_initial_decoder_mask(cfg, &dev)?;
        let dec_cross = make_mask(&tgt_pad, &enc_state.padding_mask, false)?.unsqueeze(1)?; // [B,1,1,S]

        // empty self-attn cache for each decoder layer
        let mut self_cache = Vec::with_capacity(cfg.model.decoder.n_layer);
        for _ in 0..cfg.model.decoder.n_layer {
            self_cache.push(KVCache::new(
                2,
                cfg.model.decoder.kv_heads,
                cfg.data.audio_length,
                cfg.model.decoder.gqa_head_dim,
                dtype,
                &dev,
            )?);
        }

        Ok(Self {
            device: dev,
            dtype,
            enc_out,
            enc_positions: enc_state.positions.clone(),
            dec_positions,
            dec_cross_attn_mask: dec_cross,
            self_attn_cache: self_cache,
            cross_attn_cache: cross_cache,
        })
    }

    /// Generate initial decoder attention mask for cross-attention.
    /// CFG batch: [2,1] where row 0=unconditional(padding), row 1=conditional(real)
    fn generate_initial_decoder_mask(_cfg: &DiaConfig, device: &Device) -> CandleResult<Tensor> {
        // For initial state, we assume BOS token which is always "real" for conditional
        // but "padding" for unconditional (CFG pattern)
        let mask = Tensor::new(&[[0u8], [1u8]], device)?; // [2,1]: uncond=pad, cond=real
        Ok(mask)
    }

    /// Generate decoder attention mask based on current tokens.
    /// Validates tokens against audio special values (pad, bos, eos).
    fn generate_decoder_attention_mask(
        decoder_tokens: &Tensor, // [2,1,C] or [2,C] 
        cfg: &DiaConfig,
        device: &Device,
    ) -> CandleResult<Tensor> {
        let batch_size = decoder_tokens.dim(0)?;
        let seq_len = if decoder_tokens.rank() == 3 {
            decoder_tokens.dim(1)? // [B,T,C] case
        } else {
            1 // [B,C] case - single token
        };
        
        // Create mask based on token validity
        let mut mask_data = vec![0u8; batch_size * seq_len];
        
        for batch_idx in 0..batch_size {
            for seq_idx in 0..seq_len {
                // Extract token value - handle both [B,C] and [B,T,C] cases
                let token_slice = if decoder_tokens.rank() == 3 {
                    decoder_tokens.i((batch_idx, seq_idx, 0))? // Use first channel for validation
                } else {
                    decoder_tokens.i((batch_idx, 0))? // [B,C] case
                };
                let token_value = token_slice.to_scalar::<u32>()?;
                
                // Token is "real" if it's not padding and is valid
                let is_real = token_value != cfg.data.audio_pad_value && 
                             (token_value == cfg.data.audio_bos_value || 
                              token_value == cfg.data.audio_eos_value ||
                              token_value < cfg.data.audio_pad_value); // Valid audio tokens
                
                // For CFG: batch_idx 0 = unconditional (should be padding for most cases)
                // batch_idx 1 = conditional (use actual token validity)
                mask_data[batch_idx * seq_len + seq_idx] = if batch_idx == 0 {
                    0u8 // Unconditional = padding
                } else {
                    if is_real { 1u8 } else { 0u8 }
                };
            }
        }
        
        Tensor::from_vec(mask_data, (batch_size, seq_len), device)
    }

    /// Update cross-attention mask based on current decoder tokens.
    pub fn update_cross_attention_mask(
        &mut self,
        decoder_tokens: &Tensor,
        cfg: &DiaConfig,
        enc_state: &EncoderInferenceState,
    ) -> CandleResult<()> {
        let tgt_pad = Self::generate_decoder_attention_mask(decoder_tokens, cfg, &self.device)?;
        self.dec_cross_attn_mask = make_mask(&tgt_pad, &enc_state.padding_mask, false)?.unsqueeze(1)?;
        Ok(())
    }

    /// Call **before** every decode step (`step` is absolute).
    pub fn prepare_step(&mut self, step: usize) {
        // Fix arange to use the proper signature (start, end, device)
        self.dec_positions = Tensor::arange(step as u32, (step + 1) as u32, &self.device)
            .map_err(|e| format!("Failed to create arange tensor: {e}"))
            .and_then(|t| {
                t.expand(&[2, 1])
                    .map_err(|e| format!("Failed to expand tensor: {e}"))
            })
            .unwrap_or_else(|e| {
                tracing::error!("Tensor operation failed in prepare_step: {}", e);
                // Return a fallback tensor to prevent panic - if this fails too, we're in serious trouble
                Tensor::zeros(&[2, 1], candle_core::DType::U32, &self.device).unwrap_or_else(
                    |fallback_err| {
                        tracing::error!(
                            "Critical: fallback tensor creation failed: {}",
                            fallback_err
                        );
                        // Last resort: return a minimal tensor that won't crash downstream
                        match Tensor::new(&[[0u32], [0u32]], &self.device) {
                            Ok(tensor) => tensor,
                            Err(critical_err) => {
                                tracing::error!(
                                    "FATAL: Cannot create any tensor: {}",
                                    critical_err
                                );
                                // We must return something - create the simplest possible tensor
                                Tensor::zeros(&[1, 1], candle_core::DType::U32, &self.device)
                                    .unwrap_or_else(|_| {
                                        // Absolute last resort - system is fundamentally broken at this point
                                        panic!(
                                            "Cannot create any tensor - device or memory failure"
                                        )
                                    })
                            }
                        }
                    },
                )
            });
    }
}

// ==========================================================================
// Generated-token buffer (one per request)
// ==========================================================================

pub struct DecoderOutput {
    pub generated_tokens: Tensor, // [-1 = empty / masked]
    pub prefill_step: usize,
}

impl DecoderOutput {
    pub fn new(cfg: &DiaConfig, dev: &Device) -> CandleResult<Self> {
        // Use i64 instead of i32 as it implements WithDType
        Ok(Self {
            generated_tokens: Tensor::full(-1i64, (cfg.data.audio_length, cfg.data.channels), dev)?,
            prefill_step: 0,
        })
    }

    /// Slice helpers -------------------------------------------------------
    pub fn get_tokens_at(&self, from: usize, to: usize) -> CandleResult<Tensor> {
        // Production-grade error handling for tensor indexing
        self.generated_tokens.i((.., from..to)).map_err(|e| {
            candle_core::Error::Msg(format!(
                "Token slice indexing failed at range {from}..{to}: {e}"
            ))
        })
    }

    pub fn update_one(&mut self, tok_c: Tensor, step: usize, masked: bool) -> CandleResult<()> {
        if masked {
            // Use i64 instead of i32 to match the tensor dtype we used in 'new'
            let neg_one =
                Tensor::full(-1i64, (1,), self.generated_tokens.device()).map_err(|e| {
                    candle_core::Error::Msg(format!("Failed to create negative one tensor: {e}"))
                })?;
            // Indexing using proper IndexOp syntax
            let tokens_at_step = self.generated_tokens.i((step,)).map_err(|e| {
                candle_core::Error::Msg(format!("Failed to index tokens at step {step}: {e}"))
            })?;
            let mask = tokens_at_step.eq(&neg_one).map_err(|e| {
                candle_core::Error::Msg(format!("Failed to create mask comparison: {e}"))
            })?;
            let merged = Tensor::where_cond(&mask, &tok_c, &tokens_at_step).map_err(|e| {
                candle_core::Error::Msg(format!("Failed to merge tokens conditionally: {e}"))
            })?;
            // Handle tensor updates with scatter operation - a safer approach in candle
            // Create a new tensor with updated values at the specified step
            let all_indices = Tensor::arange(
                0,
                self.generated_tokens.dim(0)? as u32,
                self.generated_tokens.device(),
            )
            .map_err(|e| candle_core::Error::Msg(format!("Failed to create index range: {e}")))?;

            // Create a mask where only the step index is 1, others are 0
            let step_tensor =
                Tensor::new(&[step as u32], self.generated_tokens.device()).map_err(|e| {
                    candle_core::Error::Msg(format!("Failed to create step tensor: {e}"))
                })?;
            let mask = all_indices
                .eq(&step_tensor)
                .map_err(|e| candle_core::Error::Msg(format!("Failed to create step mask: {e}")))?;

            // Use where_cond to update only at the masked position
            let step_value = merged.unsqueeze(0).map_err(|e| {
                candle_core::Error::Msg(format!("Failed to unsqueeze merged tensor: {e}"))
            })?; // make sure dimensions match

            // Apply where_cond across the full tensor
            let mask_broadcasted = mask
                .unsqueeze(1)
                .map_err(|e| candle_core::Error::Msg(format!("Failed to unsqueeze mask: {e}")))?; // Broadcast the mask to match all dimensions
            let step_broadcasted = step_value
                .broadcast_as(self.generated_tokens.shape())
                .map_err(|e| {
                    candle_core::Error::Msg(format!("Failed to broadcast step value: {e}"))
                })?;
            let updated =
                Tensor::where_cond(&mask_broadcasted, &step_broadcasted, &self.generated_tokens)
                    .map_err(|e| {
                        candle_core::Error::Msg(format!("Failed to apply conditional update: {e}"))
                    })?;

            self.generated_tokens = updated;
        } else {
            // Use the same where_cond approach as the masked case, but without a conditional mask
            let all_indices = Tensor::arange(
                0,
                self.generated_tokens.dim(0)? as u32,
                self.generated_tokens.device(),
            )
            .map_err(|e| candle_core::Error::Msg(format!("Failed to create index range: {e}")))?;

            // Create a mask where only the step index is 1, others are 0
            let step_tensor =
                Tensor::new(&[step as u32], self.generated_tokens.device()).map_err(|e| {
                    candle_core::Error::Msg(format!("Failed to create step tensor: {e}"))
                })?;
            let mask = all_indices
                .eq(&step_tensor)
                .map_err(|e| candle_core::Error::Msg(format!("Failed to create step mask: {e}")))?;

            // Prepare the update value with correct shape
            let step_value = tok_c
                .unsqueeze(0)
                .map_err(|e| candle_core::Error::Msg(format!("Failed to unsqueeze token: {e}")))?;

            // Apply where_cond across the full tensor
            let mask_broadcasted = mask
                .unsqueeze(1)
                .map_err(|e| candle_core::Error::Msg(format!("Failed to unsqueeze mask: {e}")))?; // Broadcast the mask to match all dimensions
            let step_broadcasted = step_value
                .broadcast_as(self.generated_tokens.shape())
                .map_err(|e| {
                    candle_core::Error::Msg(format!("Failed to broadcast step value: {e}"))
                })?;
            let updated =
                Tensor::where_cond(&mask_broadcasted, &step_broadcasted, &self.generated_tokens)
                    .map_err(|e| {
                        candle_core::Error::Msg(format!("Failed to apply conditional update: {e}"))
                    })?;

            self.generated_tokens = updated;
        }
        Ok(())
    }

    pub fn prefill(&mut self, toks: Tensor, prefill_step: usize) -> CandleResult<()> {
        let total = self.generated_tokens.dim(0).map_err(|e| {
            candle_core::Error::Msg(format!("Failed to get tensor dimension 0: {e}"))
        })?;
        let channels = self.generated_tokens.dim(1).map_err(|e| {
            candle_core::Error::Msg(format!("Failed to get tensor dimension 1: {e}"))
        })?;
        let tok_len = toks.dim(0).map_err(|e| {
            candle_core::Error::Msg(format!("Failed to get token tensor dimension 0: {e}"))
        })?;

        // Pad the incoming tokens to full length with -1 sentinel values.
        let pad_len = total.saturating_sub(tok_len);
        let padded = if pad_len > 0 {
            let pad = Tensor::full(-1i64, (pad_len, channels), self.generated_tokens.device())
                .map_err(|e| {
                    candle_core::Error::Msg(format!("Failed to create padding tensor: {e}"))
                })?;
            Tensor::cat(&[toks, pad], 0).map_err(|e| {
                candle_core::Error::Msg(format!("Failed to concatenate tensors: {e}"))
            })?
        } else {
            toks
        };

        self.generated_tokens = padded
            .to_dtype(self.generated_tokens.dtype())
            .map_err(|e| candle_core::Error::Msg(format!("Failed to convert tensor dtype: {e}")))?;
        self.prefill_step = prefill_step;
        Ok(())
    }
}
