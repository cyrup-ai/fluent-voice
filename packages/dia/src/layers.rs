//! Encoder & Decoder building-blocks (Rotary-PE, DenseGeneral, Attention, MLP)
//! (corrected rotary mix **and** now applies causal / padding masks)

use crate::config::DiaConfig;
use crate::state::{DecoderInferenceState, EncoderInferenceState, KVCache};

use crate::{DType, Module, Tensor, VarBuilder};
use candle_core::IndexOp;
use candle_nn::ops;

// Import optimizations when GPU features are enabled
#[cfg(any(feature = "cuda", feature = "metal"))]
use crate::optimizations::{attention_optimized, matmul_optimized};

/// Convenience: small −∞ used for masking logits.
const NEG_INF: f32 = -1e30;

/// Configuration for transformer layer forward pass
#[derive(Debug)]
pub struct TransformerLayerConfig<'a> {
    pub dec_positions: &'a Tensor,
    pub enc_out: &'a Tensor,
    pub enc_positions: &'a Tensor,
    pub dec_cross_mask: &'a Tensor,
    pub self_cache: &'a mut KVCache,
    pub cross_cache: &'a KVCache,
    pub prefill: bool,
}

// ───────────────────────────── RmsNorm wrapper ────────────────────────────

use candle_nn::layer_norm as ln;

#[derive(Clone, Debug)]
pub struct RmsNorm {
    inner: ln::RmsNorm,
}
impl RmsNorm {
    pub fn new(size: usize, eps: f64, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            inner: ln::rms_norm(size, eps, vb)?,
        })
    }
}
impl candle_core::Module for RmsNorm {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        self.inner.forward(xs)
    }
}

// ───────────────────── util split helper ─────────────────────────────────

#[inline]
fn split_last(t: &Tensor, last: usize) -> candle_core::Result<(Tensor, Tensor)> {
    let half = last / 2;
    Ok((
        t.narrow(t.rank() - 1, 0, half)?,
        t.narrow(t.rank() - 1, half, half)?,
    ))
}

// ───────────────────── DenseGeneral ──────────────────────────────────────

#[derive(Clone)]
pub struct DenseGeneral {
    w: Tensor,
    out_rank: usize,
}
impl DenseGeneral {
    pub fn new(
        vb: VarBuilder,
        in_dims: &[usize],
        out_dims: &[usize],
        _dtype: DType,
    ) -> candle_core::Result<Self> {
        let shape: Vec<_> = in_dims.iter().chain(out_dims).copied().collect();
        Ok(Self {
            w: vb.get(shape.as_slice(), "weight")?,
            out_rank: out_dims.len(),
        })
    }
    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x2 = x.flatten_all()?;

        // Use optimized matmul for GPU when available
        #[cfg(any(feature = "cuda", feature = "metal"))]
        let y = if matches!(
            x.device(),
            candle_core::Device::Cuda(_) | candle_core::Device::Metal(_)
        ) {
            matmul_optimized(&x2, &self.w)?
        } else {
            x2.matmul(&self.w)?
        };

        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let y = x2.matmul(&self.w)?;

        let output_shape: Vec<_> = x.dims()[..x.rank() - 1]
            .iter()
            .chain(&self.w.dims()[1..])
            .copied()
            .collect();

        // Validate output rank matches expected
        assert_eq!(
            self.w.dims()[1..].len(),
            self.out_rank,
            "Output dimensions rank mismatch: expected {}, got {}",
            self.out_rank,
            self.w.dims()[1..].len()
        );

        y.reshape(output_shape)
    }
}

// ───────────────────── Rotary Embedding ──────────────────────────────────

#[derive(Clone)]
pub struct Rotary {
    inv_freq: Tensor,
}
impl Rotary {
    pub fn new(
        head_dim: usize,
        min: f32,
        max: f32,
        dev: &candle_core::Device,
    ) -> candle_core::Result<Self> {
        let hd2 = head_dim as f32 / 2.0;
        let inv_freq: Vec<f32> = (0..head_dim / 2)
            .map(|i| min * ((max / min).powf(i as f32 / hd2)))
            .collect();
        Ok(Self {
            inv_freq: Tensor::from_vec(inv_freq, (1, head_dim / 2), dev)?,
        })
    }

    pub fn apply(&self, x: Tensor, pos: &Tensor) -> candle_core::Result<Tensor> {
        use candle_core::D;
        let freq = (pos.unsqueeze(D::Minus1)? * &self.inv_freq)?;
        let sinus = freq.sin()?;
        let cosinus = freq.cos()?;
        let (x1, x2) = split_last(&x, x.dims()[x.rank() - 1])?;
        // corrected rotary mix
        let first_part = ((&x1 * &cosinus)? - (&x2 * &sinus)?)?;
        let second_part = ((&x2 * &cosinus)? + (&x1 * &sinus)?)?;
        Tensor::cat(&[&first_part, &second_part], D::Minus1)
    }
}

// ───────────────────── Scaled-Dot-Product Attention with MASK ────────────

fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> candle_core::Result<Tensor> {
    // Use optimized attention for GPU when available
    #[cfg(any(feature = "cuda", feature = "metal"))]
    if matches!(
        q.device(),
        candle_core::Device::Cuda(_) | candle_core::Device::Metal(_)
    ) {
        return attention_optimized(q, k, v, mask, false);
    }

    // Standard implementation for CPU
    use candle_core::D;
    let dim = q.dim(D::Minus1)?;
    let scale = 1f64 / (dim as f64).sqrt();
    let mut scores = (q.matmul(&k.t()?)? * scale)?; // [B,H,Tq,Tk]
    if let Some(m) = mask {
        // mask == 0 → set to −inf so softmax ~ 0
        let neginf = Tensor::full(NEG_INF, scores.dims(), scores.device())?;
        scores = Tensor::where_cond(&m.to_dtype(DType::U8)?, &scores, &neginf)?;
    }
    let attn = candle_nn::ops::softmax(&scores, D::Minus1)?;
    attn.matmul(v)
}

// ───────────────────── Attention Module ──────────────────────────────────

#[derive(Clone)]
pub struct Attention {
    q_proj: DenseGeneral,
    k_proj: DenseGeneral,
    v_proj: DenseGeneral,
    o_proj: DenseGeneral,
    rotary: Rotary,
    n_q: usize,
    n_kv: usize,
    head_dim: usize,
    groups: usize,
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vb: VarBuilder,
        cfg: &DiaConfig,
        q_emb: usize,
        kv_emb: usize,
        n_q: usize,
        n_kv: usize,
        head_dim: usize,
        dtype: DType,
        _is_cross: bool,
    ) -> candle_core::Result<Self> {
        Ok(Self {
            q_proj: DenseGeneral::new(vb.pp("q"), &[q_emb], &[n_q, head_dim], dtype)?,
            k_proj: DenseGeneral::new(vb.pp("k"), &[kv_emb], &[n_kv, head_dim], dtype)?,
            v_proj: DenseGeneral::new(vb.pp("v"), &[kv_emb], &[n_kv, head_dim], dtype)?,
            o_proj: DenseGeneral::new(vb.pp("o"), &[n_q, head_dim], &[q_emb], dtype)?,
            rotary: Rotary::new(
                head_dim,
                cfg.model.rope_min_timescale as _,
                cfg.model.rope_max_timescale as _,
                vb.device(),
            )?,
            n_q,
            n_kv,
            head_dim,
            groups: n_q / n_kv,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x_q: &Tensor,
        x_kv: &Tensor,
        q_pos: &Tensor,
        kv_pos: &Tensor,
        mask: Option<&Tensor>,
        cache: Option<&mut KVCache>,
        _causal: bool,
    ) -> candle_core::Result<Tensor> {
        // Validate input dimensions match expected attention configuration
        debug_assert_eq!(
            x_q.dim(x_q.rank() - 1)?,
            self.q_proj.w.dim(0)?,
            "Query input dimension mismatch"
        );
        debug_assert_eq!(
            x_kv.dim(x_kv.rank() - 1)?,
            self.k_proj.w.dim(0)?,
            "Key-Value input dimension mismatch"
        );

        let q = self.rotr_proj(&self.q_proj, x_q, q_pos)?;
        let (k, v) = self.kv_proj(x_kv, kv_pos, cache)?;

        // Validate attention head dimensions
        debug_assert_eq!(
            q.dim(q.rank() - 1)? / self.n_q,
            self.head_dim,
            "Query head dimension mismatch"
        );
        debug_assert_eq!(
            k.dim(k.rank() - 1)? / self.n_kv,
            self.head_dim,
            "Key head dimension mismatch"
        );
        debug_assert_eq!(
            self.n_q % self.n_kv,
            0,
            "n_q must be divisible by n_kv for grouped query attention"
        );
        debug_assert_eq!(
            self.groups,
            self.n_q / self.n_kv,
            "Groups calculation mismatch"
        );
        let attn = sdpa(&q, &k, &v, mask)?;
        let attn = attn.transpose(1, 2)?; // B,T,N,H
        self.o_proj.forward(&attn)
    }

    fn rotr_proj(
        &self,
        proj: &DenseGeneral,
        x: &Tensor,
        pos: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let q = proj.forward(x)?; // B,T,N,H
        self.rotary.apply(q, pos)?.transpose(1, 2) // B,N,T,H
    }

    fn kv_proj(
        &self,
        x_kv: &Tensor,
        pos: &Tensor,
        cache: Option<&mut KVCache>,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let mut k = self.k_proj.forward(x_kv)?;
        let mut v = self.v_proj.forward(x_kv)?;
        k = self.rotary.apply(k, pos)?.transpose(1, 2)?; // B,N,T,H
        v = v.transpose(1, 2)?;
        if let Some(cache) = cache {
            cache.update(k, v)
        } else {
            Ok((k, v))
        }
    }
}

// ---------- MLP ------------------------------------------------------------

#[derive(Clone)]
pub struct Mlp {
    wi_fused: DenseGeneral,
    wo: DenseGeneral,
}

impl Mlp {
    pub fn new(
        vb: VarBuilder,
        embed: usize,
        hidden: usize,
        dtype: DType,
    ) -> candle_core::Result<Self> {
        let wi_fused = DenseGeneral::new(vb.pp("wi"), &[embed], &[2, hidden], dtype)?;
        let wo = DenseGeneral::new(vb.pp("wo"), &[hidden], &[embed], dtype)?;
        Ok(Self { wi_fused, wo })
    }

    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let fused = self.wi_fused.forward(x)?;

        // The multi-argument `.i(.., .., idx)` form has been removed in Candle
        // 0.9.  Indexing is now performed using a single tuple that matches
        // the tensor rank.
        let gate = fused.i((.., .., 0))?; // [..., hidden]
        let up = fused.i((.., .., 1))?;

        let silu = ops::silu(&gate)?;
        self.wo.forward(&(silu * up)?)
    }
}

// ---------- Encoder Layer / Stack -----------------------------------------

#[derive(Clone)]
pub struct EncoderLayer {
    sa_norm: RmsNorm,
    sa: Attention,
    ffn_norm: RmsNorm,
    mlp: Mlp,
}

impl EncoderLayer {
    pub fn new(cfg: &DiaConfig, vb: VarBuilder, dtype: DType) -> candle_core::Result<Self> {
        let enc = &cfg.model.encoder;
        let sa_norm = RmsNorm::new(
            enc.n_embd,
            cfg.model.normalization_layer_epsilon as f64,
            vb.pp("sa_norm"),
        )?;
        let sa = Attention::new(
            vb.pp("sa"),
            cfg,
            enc.n_embd,
            enc.n_embd,
            enc.n_head,
            enc.n_head,
            enc.head_dim,
            dtype,
            false,
        )?;
        let ffn_norm = RmsNorm::new(
            enc.n_embd,
            cfg.model.normalization_layer_epsilon as f64,
            vb.pp("ffn_norm"),
        )?;
        let mlp = Mlp::new(vb.pp("mlp"), enc.n_embd, enc.n_hidden, dtype)?;
        Ok(Self {
            sa_norm,
            sa,
            ffn_norm,
            mlp,
        })
    }

    pub fn forward(&self, x: Tensor, state: &EncoderInferenceState) -> candle_core::Result<Tensor> {
        // Self-attention
        let h = self.sa_norm.forward(&x)?;
        let h = self.sa.forward(
            &h,
            &h,
            &state.positions,
            &state.positions,
            Some(&state.attn_mask),
            None,
            false,
        )?;
        let x = (x + h)?;

        // Feed-forward
        let h = self.ffn_norm.forward(&x)?;
        x + self.mlp.forward(&h)?
    }
}

#[derive(Clone)]
pub struct Encoder {
    embed: Tensor,
    layers: Vec<EncoderLayer>,
    norm: RmsNorm,
}

impl Encoder {
    pub fn new(cfg: &DiaConfig, vb: VarBuilder, dtype: DType) -> candle_core::Result<Self> {
        let enc = &cfg.model.encoder;
        // Retrieve the token-embedding matrix.  In Candle ≥0.9 the `get` helper
        // takes only the *shape* and a *name* – the dtype is configured on the
        // `VarBuilder` itself.
        let embed = vb
            .pp("embed")
            .get(&[cfg.model.src_vocab_size, enc.n_embd], "weight")?;
        let mut layers = Vec::with_capacity(enc.n_layer);
        for l in 0..enc.n_layer {
            layers.push(EncoderLayer::new(cfg, vb.pp(format!("layers.{l}")), dtype)?);
        }
        let norm = RmsNorm::new(
            enc.n_embd,
            cfg.model.normalization_layer_epsilon as f64,
            vb.pp("norm"),
        )?;
        Ok(Self {
            embed,
            layers,
            norm,
        })
    }

    pub fn forward(
        &self,
        ids: &Tensor,
        state: &EncoderInferenceState,
    ) -> candle_core::Result<Tensor> {
        use candle_core::D;
        let mut x = self.embed.gather(ids, D::Minus1)?.to_dtype(ids.dtype())?;
        for layer in &self.layers {
            x = layer.forward(x, state)?;
        }
        self.norm.forward(&x)
    }
}

// ---------- Decoder Layer / Stack -----------------------------------------

#[derive(Clone)]
pub struct DecoderLayer {
    self_norm: RmsNorm,
    self_attn: Attention,
    cross_norm: RmsNorm,
    cross_attn: Attention,
    ffn_norm: RmsNorm,
    mlp: Mlp,
}

impl DecoderLayer {
    pub fn new(cfg: &DiaConfig, vb: VarBuilder, dtype: DType) -> candle_core::Result<Self> {
        let dec = &cfg.model.decoder;
        let enc = &cfg.model.encoder;
        let self_norm = RmsNorm::new(
            dec.n_embd,
            cfg.model.normalization_layer_epsilon as f64,
            vb.pp("self_norm"),
        )?;
        let cross_norm = RmsNorm::new(
            dec.n_embd,
            cfg.model.normalization_layer_epsilon as f64,
            vb.pp("cross_norm"),
        )?;
        let ffn_norm = RmsNorm::new(
            dec.n_embd,
            cfg.model.normalization_layer_epsilon as f64,
            vb.pp("ffn_norm"),
        )?;

        let self_attn = Attention::new(
            vb.pp("self"),
            cfg,
            dec.n_embd,
            dec.n_embd,
            dec.gqa_query_heads,
            dec.kv_heads,
            dec.gqa_head_dim,
            dtype,
            false,
        )?;
        let cross_attn = Attention::new(
            vb.pp("cross"),
            cfg,
            dec.n_embd,
            enc.n_embd,
            dec.cross_query_heads,
            dec.cross_query_heads,
            dec.cross_head_dim,
            dtype,
            true,
        )?;

        let mlp = Mlp::new(vb.pp("mlp"), dec.n_embd, dec.n_hidden, dtype)?;
        Ok(Self {
            self_norm,
            self_attn,
            cross_norm,
            cross_attn,
            ffn_norm,
            mlp,
        })
    }

    pub fn forward(
        &self,
        x: Tensor,
        config: &mut TransformerLayerConfig,
    ) -> candle_core::Result<Tensor> {
        // Self-attention pass
        let h = self.self_norm.forward(&x)?;
        let h = self.self_attn.forward(
            &h,
            &h,
            config.dec_positions,
            config.dec_positions,
            None,
            Some(config.self_cache),
            config.prefill, // causal when pre-filling
        )?;
        let x = (x + h)?;

        let h = self.cross_norm.forward(&x)?;
        let h = self.cross_attn.forward(
            &h,
            config.enc_out,
            config.dec_positions,
            config.enc_positions,
            Some(config.dec_cross_mask),
            Some(&mut config.cross_cache.clone()),
            false,
        )?;
        let x = (x + h)?;

        let h = self.ffn_norm.forward(&x)?;
        x + self.mlp.forward(&h)?
    }
}

#[derive(Clone)]
pub struct Decoder {
    embeds: Vec<Tensor>,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    logits: DenseGeneral,
}

impl Decoder {
    pub fn new(cfg: &DiaConfig, vb: VarBuilder, dtype: DType) -> candle_core::Result<Self> {
        let dec = &cfg.model.decoder;
        let mut embeds = Vec::with_capacity(cfg.data.channels);
        for ch in 0..cfg.data.channels {
            embeds.push(
                vb.pp(format!("emb_{ch}"))
                    .get(&[cfg.model.tgt_vocab_size, dec.n_embd], "weight")?,
            );
        }
        let mut layers = Vec::with_capacity(dec.n_layer);
        for l in 0..dec.n_layer {
            layers.push(DecoderLayer::new(cfg, vb.pp(format!("layers.{l}")), dtype)?);
        }
        let norm = RmsNorm::new(
            dec.n_embd,
            cfg.model.normalization_layer_epsilon as f64,
            vb.pp("norm"),
        )?;
        let logits = DenseGeneral::new(
            vb.pp("logits"),
            &[dec.n_embd],
            &[cfg.data.channels, cfg.model.tgt_vocab_size],
            dtype,
        )?;
        Ok(Self {
            embeds,
            layers,
            norm,
            logits,
        })
    }

    pub fn forward_step(
        &self,
        ids_b1c: &Tensor, // [B,1,C]
        state: &mut DecoderInferenceState,
    ) -> candle_core::Result<Tensor> {
        use candle_core::D;
        let mut x = Tensor::zeros_like(&self.embeds[0])?;
        for (i, emb) in self.embeds.iter().enumerate() {
            x = (&x + &emb.gather(&ids_b1c.i((.., .., i))?, D::Minus1)?)?;
        }

        for (l, layer) in self.layers.iter().enumerate() {
            let dec_positions = &state.dec_positions;
            let enc_out = &state.enc_out;
            let enc_positions = &state.enc_positions;
            let dec_cross_mask = &state.dec_cross_attn_mask;

            let self_cache = &mut state.self_attn_cache[l];
            let cross_cache = &state.cross_attn_cache[l];

            let mut config = TransformerLayerConfig {
                dec_positions,
                enc_out,
                enc_positions,
                dec_cross_mask,
                self_cache,
                cross_cache,
                prefill: false,
            };
            x = layer.forward(x, &mut config)?;
        }

        let h = self.norm.forward(&x)?;
        self.logits.forward(&h)
    }

    pub fn prefill(
        &self,
        tgt_bt_c: &Tensor,
        state: &mut DecoderInferenceState,
    ) -> candle_core::Result<()> {
        use candle_core::D;
        let mut x = Tensor::zeros_like(&self.embeds[0])?;
        for (i, emb) in self.embeds.iter().enumerate() {
            x = (&x + &emb.gather(&tgt_bt_c.i((.., .., i))?, D::Minus1)?)?;
        }
        for (l, layer) in self.layers.iter().enumerate() {
            let dec_positions = &state.dec_positions;
            let enc_out = &state.enc_out;
            let enc_positions = &state.enc_positions;
            let dec_cross_mask = &state.dec_cross_attn_mask;

            let self_cache = &mut state.self_attn_cache[l];
            let cross_cache = &state.cross_attn_cache[l];
            let mut config = TransformerLayerConfig {
                dec_positions,
                enc_out,
                enc_positions,
                dec_cross_mask,
                self_cache,
                cross_cache,
                prefill: true,
            };
            x = layer.forward(x, &mut config)?;
        }
        Ok(())
    }

    /// Pre-compute the K/V that will be shared by all cross-attention layers.
    pub fn precompute_cross_attn_cache(
        &self,
        enc_out: Tensor, // [B,S,E]
        enc_pos: Tensor, // [B,S]
    ) -> candle_core::Result<Vec<KVCache>> {
        let mut caches = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let k = layer.cross_attn.k_proj.forward(&enc_out)?;
            let v = layer.cross_attn.v_proj.forward(&enc_out)?;

            // Apply rotary embeddings to keys
            let k = layer.cross_attn.rotary.apply(k, &enc_pos)?;

            // Reshape for attention
            let k = k.transpose(1, 2)?; // B,N,T,H
            let v = v.transpose(1, 2)?;

            caches.push(KVCache::from_tensors(k, v));
        }
        Ok(caches)
    }
}
