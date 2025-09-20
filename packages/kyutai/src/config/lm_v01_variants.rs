//! v0.1 model variants and streaming configurations

use super::conditioners::DepFormerConfig;
use super::lm_base::LmConfig;
use crate::transformer::{Config as TransformerConfig, NormType, PositionalEmbedding};

impl LmConfig {
    /// v0.1 vision model with streaming support
    pub fn v0_1_vision_streaming(num_slices: usize) -> Self {
        let mut cfg = Self::v0_1_vision();
        cfg.audio_codebooks = 16;
        if let Some(depformer) = cfg.depformer.as_mut() {
            depformer.num_slices = num_slices;
            depformer.transformer.context = num_slices;
        }
        cfg
    }

    /// v0.1 model with streaming support
    pub fn v0_1_streaming(num_slices: usize) -> Self {
        let mut cfg = Self::v0_1();
        cfg.audio_codebooks = 16;
        if let Some(depformer) = cfg.depformer.as_mut() {
            depformer.num_slices = num_slices;
            depformer.transformer.context = num_slices;
        }
        cfg
    }

    /// v0.1 model for ASR (Automatic Speech Recognition)
    pub fn v0_1_asr() -> Self {
        let mut cfg = Self::v0_1();
        cfg.audio_codebooks = 8;
        if let Some(depformer) = cfg.depformer.as_mut() {
            depformer.num_slices = 0;
            depformer.transformer.context = 0;
        }
        cfg
    }

    /// Speech-to-Speech v0.1 model configuration
    pub fn s2s_v0_1() -> Self {
        let lm_cfg = TransformerConfig {
            d_model: 2048,
            num_heads: 16,
            num_layers: 16,
            dim_feedforward: 4096 * 2,
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: 3000,
            max_period: 10000,
            use_conv_block: false,
            use_conv_bias: true,
            cross_attention: None,
            gating: Some(candle_nn::Activation::Silu),
            norm: NormType::RmsNorm,
            positional_embedding: PositionalEmbedding::Rope,
            conv_layout: false,
            conv_kernel_size: 3,
            kv_repeat: 1,
            max_seq_len: 4096,
            shared_cross_attn: false,
        };
        Self {
            transformer: lm_cfg,
            depformer: Some(DepFormerConfig {
                transformer: TransformerConfig {
                    d_model: 1024,
                    num_heads: 16,
                    num_layers: 6,
                    dim_feedforward: 1024 * 4,
                    causal: true,
                    norm_first: true,
                    bias_ff: false,
                    bias_attn: false,
                    layer_scale: None,
                    context: 16,
                    max_period: 10000,
                    use_conv_block: false,
                    use_conv_bias: true,
                    cross_attention: None,
                    gating: Some(candle_nn::Activation::Silu),
                    norm: NormType::RmsNorm,
                    positional_embedding: PositionalEmbedding::None,
                    conv_layout: false,
                    conv_kernel_size: 3,
                    kv_repeat: 1,
                    max_seq_len: 4096,
                    shared_cross_attn: false,
                },
                num_slices: 16,
                low_rank_embeddings: None,
                shared: true,
                multi_linear: true,
                weights_per_step: true,
                pos_emb: "none".to_string(),
                weights_per_step_schedule: None,
            }),
            fuser: None,
            conditioners: None,
            audio_vocab_size: 2049,
            text_in_vocab_size: 48001,
            text_out_vocab_size: 48000,
            audio_codebooks: 16,
        }
    }

    /// Speech-to-Speech v0.1 with streaming support
    pub fn s2s_v0_1_streaming(num_slices: usize) -> Self {
        let mut cfg = Self::s2s_v0_1();
        cfg.audio_codebooks = 16;
        if let Some(depformer) = cfg.depformer.as_mut() {
            depformer.num_slices = num_slices;
            depformer.transformer.context = num_slices;
        }
        cfg
    }

    /// ASR v0.1 1B parameter model
    pub fn asr_v0_1_1b() -> Self {
        let lm_cfg = TransformerConfig {
            d_model: 2048,
            num_heads: 16,
            num_layers: 16,
            dim_feedforward: 2048 * 4,
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: 750,
            max_period: 100_000,
            use_conv_block: false,
            use_conv_bias: true,
            cross_attention: None,
            gating: Some(candle_nn::Activation::Silu),
            norm: NormType::RmsNorm,
            positional_embedding: PositionalEmbedding::Rope,
            conv_layout: false,
            conv_kernel_size: 3,
            kv_repeat: 1,
            max_seq_len: 4096,
            shared_cross_attn: false,
        };
        Self {
            transformer: lm_cfg,
            depformer: None,
            fuser: None,
            conditioners: None,
            audio_vocab_size: 2049,
            text_in_vocab_size: 48001,
            text_out_vocab_size: 48000,
            audio_codebooks: 8,
        }
    }
}
