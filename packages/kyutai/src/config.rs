// src/config.rs

use super::transformer::{
    Config as TransformerConfig, CrossAttentionGating, NormType, PositionalEmbedding,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct LutConfig {
    pub n_bins: usize,
    pub dim: usize,
    pub tokenizer: String,
    pub possible_values: Vec<String>,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct TensorConfig {
    pub dim: usize,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(tag = "type")]
pub enum ConditionerConfig {
    Lut(LutConfig),
    Tensor(TensorConfig),
}

pub type ConditionersConfig = HashMap<String, ConditionerConfig>;

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct FuserConfig {
    pub cross_attention_pos_emb: bool,
    pub cross_attention_pos_emb_scale: f32,
    pub sum: Vec<String>,
    pub prepend: Vec<String>,
    pub cross: Vec<String>,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct DepFormerConfig {
    pub transformer: TransformerConfig,
    pub num_slices: usize,
    pub low_rank_embeddings: Option<usize>,
    pub shared: bool,
    pub multi_linear: bool,
    pub weights_per_step: bool,
    pub pos_emb: String,
    pub weights_per_step_schedule: Option<Vec<usize>>,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct LmConfig {
    pub transformer: TransformerConfig,
    pub depformer: Option<DepFormerConfig>,
    pub fuser: Option<FuserConfig>,
    pub conditioners: Option<ConditionersConfig>,
    pub audio_vocab_size: usize,
    pub text_in_vocab_size: usize,
    pub text_out_vocab_size: usize,
    pub audio_codebooks: usize,
}

impl LmConfig {
    pub fn tts_1_6b_en_fr() -> Self {
        let lm_cfg = TransformerConfig {
            d_model: 2048,
            num_heads: 16,
            num_layers: 16,
            dim_feedforward: 8448,
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: 500,
            max_period: 10000,
            use_conv_block: false,
            use_conv_bias: true,
            cross_attention: Some((CrossAttentionGating::Normal, NormType::RmsNorm, None)),
            gating: Some(candle_nn::Activation::Silu),
            norm: NormType::RmsNorm,
            positional_embedding: PositionalEmbedding::Rope,
            conv_layout: false,
            conv_kernel_size: 3,
            kv_repeat: 1,
            max_seq_len: 4096,
            shared_cross_attn: false,
        };
        let mut conditioners = HashMap::new();
        conditioners.insert(
            "speaker_wavs".to_string(),
            ConditionerConfig::Tensor(TensorConfig { dim: 512 }),
        );
        conditioners.insert(
            "cfg".to_string(),
            ConditionerConfig::Lut(LutConfig {
                n_bins: 7,
                dim: 16,
                tokenizer: "noop".to_string(),
                possible_values: vec![
                    "1.0".into(),
                    "1.5".into(),
                    "2.0".into(),
                    "2.5".into(),
                    "3.0".into(),
                    "3.5".into(),
                    "4.0".into(),
                ],
            }),
        );
        conditioners.insert(
            "control".to_string(),
            ConditionerConfig::Lut(LutConfig {
                n_bins: 1,
                dim: 2048,
                tokenizer: "noop".to_string(),
                possible_values: vec!["ok".into()],
            }),
        );
        Self {
            transformer: lm_cfg,
            depformer: Some(DepFormerConfig {
                transformer: TransformerConfig {
                    d_model: 1024,
                    num_heads: 16,
                    num_layers: 4,
                    dim_feedforward: 3072,
                    causal: true,
                    norm_first: true,
                    bias_ff: false,
                    bias_attn: false,
                    layer_scale: None,
                    context: 32,
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
                num_slices: 32,
                low_rank_embeddings: Some(128),
                shared: true,
                multi_linear: true,
                weights_per_step: true,
                pos_emb: "none".to_string(),
                weights_per_step_schedule: Some(vec![
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10,
                    10, 10, 10, 10, 10, 10,
                ]),
            }),
            fuser: Some(FuserConfig {
                cross_attention_pos_emb: true,
                cross_attention_pos_emb_scale: 1.0,
                sum: vec!["control".into(), "cfg".into()],
                prepend: vec![],
                cross: vec!["speaker_wavs".into()],
            }),
            conditioners: Some(conditioners),
            audio_vocab_size: 2049,
            text_in_vocab_size: 8001,
            text_out_vocab_size: 8000,
            audio_codebooks: 32,
        }
    }

    pub fn stt_2_6b_en() -> Self {
        let lm_cfg = TransformerConfig {
            d_model: 2048,
            num_heads: 32,
            num_layers: 48,
            dim_feedforward: 8448,
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: 375,
            max_period: 100000,
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
                    dim_feedforward: 4096,
                    causal: true,
                    norm_first: true,
                    bias_ff: false,
                    bias_attn: false,
                    layer_scale: None,
                    context: 0,
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
                num_slices: 0,
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
            text_in_vocab_size: 4001,
            text_out_vocab_size: 4000,
            audio_codebooks: 32,
        }
    }

    pub fn v0_1() -> Self {
        let lm_cfg = TransformerConfig {
            d_model: 4096,
            num_heads: 32,
            num_layers: 32,
            dim_feedforward: 4096 * 4,
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
                    context: 8,
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
                num_slices: 8,
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
            text_in_vocab_size: 32001,
            text_out_vocab_size: 32000,
            audio_codebooks: 8,
        }
    }

    pub fn v0_1_vision() -> Self {
        let lm_cfg = TransformerConfig {
            d_model: 4096,
            num_heads: 32,
            num_layers: 32,
            dim_feedforward: 4096 * 4,
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: 3000,
            max_period: 10000,
            use_conv_block: false,
            use_conv_bias: true,
            cross_attention: Some((
                CrossAttentionGating::ConditionalGatedSigmoid,
                NormType::RmsNorm,
                None,
            )),
            gating: Some(candle_nn::Activation::Silu),
            norm: NormType::RmsNorm,
            positional_embedding: PositionalEmbedding::Rope,
            conv_layout: false,
            conv_kernel_size: 3,
            kv_repeat: 1,
            max_seq_len: 4096,
            shared_cross_attn: true,
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
                    context: 8,
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
                num_slices: 8,
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
            text_in_vocab_size: 32001,
            text_out_vocab_size: 32000,
            audio_codebooks: 8,
        }
    }

    pub fn v0_1_vision_streaming(num_slices: usize) -> Self {
        let mut cfg = Self::v0_1_vision();
        cfg.audio_codebooks = 16;
        if let Some(depformer) = cfg.depformer.as_mut() {
            depformer.num_slices = num_slices;
            depformer.transformer.context = num_slices;
        }
        cfg
    }

    pub fn v0_1_streaming(num_slices: usize) -> Self {
        let mut cfg = Self::v0_1();
        cfg.audio_codebooks = 16;
        if let Some(depformer) = cfg.depformer.as_mut() {
            depformer.num_slices = num_slices;
            depformer.transformer.context = num_slices;
        }
        cfg
    }

    pub fn v0_1_asr() -> Self {
        let mut cfg = Self::v0_1();
        cfg.audio_codebooks = 8;
        if let Some(depformer) = cfg.depformer.as_mut() {
            depformer.num_slices = 0;
            depformer.transformer.context = 0;
        }
        cfg
    }

    pub fn tts_v0_1() -> Self {
        let lm_cfg = TransformerConfig {
            d_model: 2048,
            num_heads: 32,
            num_layers: 48,
            dim_feedforward: 4096 * 2,
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: 4096,
            max_period: 10000,
            use_conv_block: false,
            use_conv_bias: true,
            cross_attention: Some((CrossAttentionGating::Normal, NormType::LayerNorm, None)),
            gating: None,
            norm: NormType::LayerNorm,
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
            audio_vocab_size: 2050,
            text_in_vocab_size: 32001,
            text_out_vocab_size: 32001,
            audio_codebooks: 16,
        }
    }

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

    pub fn s2s_v0_1_streaming(num_slices: usize) -> Self {
        let mut cfg = Self::s2s_v0_1();
        cfg.audio_codebooks = 16;
        if let Some(depformer) = cfg.depformer.as_mut() {
            depformer.num_slices = num_slices;
            depformer.transformer.context = num_slices;
        }
        cfg
    }

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

    pub fn asr_300m_202501() -> Self {
        let lm_cfg = TransformerConfig {
            d_model: 1024,
            num_heads: 8,
            num_layers: 16,
            dim_feedforward: 1024 * 4,
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
            audio_codebooks: 32,
        }
    }

    pub fn tts_202501() -> Self {
        let lm_cfg = TransformerConfig {
            d_model: 2048,
            num_heads: 32,
            num_layers: 48,
            dim_feedforward: 2048 * 4,
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: 500,
            max_period: 10000,
            use_conv_block: false,
            use_conv_bias: true,
            cross_attention: Some((CrossAttentionGating::Normal, NormType::LayerNorm, None)),
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
                    context: 32,
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
                num_slices: 32,
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
            text_in_vocab_size: 8001,
            text_out_vocab_size: 8000,
            audio_codebooks: 32,
        }
    }

    pub fn s2s_2b_16rvq_202501() -> Self {
        let lm_cfg = TransformerConfig {
            d_model: 2560,
            num_heads: 20,
            num_layers: 24,
            dim_feedforward: 2560 * 4,
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: 3000,
            max_period: 100000,
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
            audio_codebooks: 32,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TtsConfig {
    pub acoustic_delay: usize,
    pub text_pad_token: u32,
    pub text_bos_token: u32,
    pub text_eos_token: u32,
    pub text_eop_token: u32,
    pub text_start_token: u32,
    pub text_audio_delay_in_tokens: usize,
    pub max_consecutive_pads: usize,
    pub speaker_cond_duration_s: f64,
    pub speaker_cond_dim: usize,
    pub speaker_cond_n_speakers: usize,
    pub second_stream_ahead: usize,
}

impl TtsConfig {
    pub fn v202501() -> Self {
        Self {
            acoustic_delay: 2,
            text_eop_token: 0,
            text_bos_token: 1,
            text_eos_token: 2,
            text_pad_token: 3,
            text_start_token: 8000,
            text_audio_delay_in_tokens: 16,
            max_consecutive_pads: 10,
            speaker_cond_duration_s: 10.,
            speaker_cond_dim: 512,
            speaker_cond_n_speakers: 5,
            second_stream_ahead: 2,
        }
    }
}

/// Main configuration struct (compatible with old interface)
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    /// Model dimension
    pub d_model: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Feedforward dimension
    pub dim_feedforward: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Audio configuration
    pub audio: AudioConfig,
    /// Transformer configuration
    pub transformer: TransformerConfig,
    /// Conditioning configuration
    pub conditioning: ConditioningConfig,
    /// Streaming configuration
    pub streaming: super::streaming::StreamingConfig,
    /// Language model configuration
    pub lm_config: LmConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            d_model: 768,
            num_heads: 12,
            num_layers: 12,
            dim_feedforward: 3072,
            max_seq_len: 4096,
            vocab_size: 32000,
            audio: AudioConfig::default(),
            transformer: TransformerConfig::default(),
            conditioning: ConditioningConfig::default(),
            streaming: super::streaming::StreamingConfig::default(),
            lm_config: LmConfig::v0_1(),
        }
    }
}

impl Config {
    /// Validate the configuration
    pub fn validate(&self) -> crate::Result<()> {
        if self.d_model == 0 {
            return Err(crate::error::MoshiError::Config(
                "d_model must be > 0".to_string(),
            ));
        }
        if self.num_heads == 0 {
            return Err(crate::error::MoshiError::Config(
                "num_heads must be > 0".to_string(),
            ));
        }
        if self.d_model % self.num_heads != 0 {
            return Err(crate::error::MoshiError::Config(
                "d_model must be divisible by num_heads".to_string(),
            ));
        }
        if self.vocab_size == 0 {
            return Err(crate::error::MoshiError::Config(
                "vocab_size must be > 0".to_string(),
            ));
        }
        if self.lm_config.audio_vocab_size == 0 {
            return Err(crate::error::MoshiError::Config(
                "audio_vocab_size must be > 0".to_string(),
            ));
        }
        if self.lm_config.audio_codebooks == 0 || self.lm_config.audio_codebooks > 64 {
            return Err(crate::error::MoshiError::Config(
                "audio_codebooks must be between 1 and 64".to_string(),
            ));
        }
        Ok(())
    }
}

/// Audio processing configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AudioConfig {
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of audio channels
    pub channels: u32,
    /// Frame size for processing
    pub frame_size: usize,
    /// Hop length for windowing
    pub hop_length: usize,
    /// Number of mel frequency bins
    pub n_mels: usize,
    /// FFT window size
    pub n_fft: usize,
    /// Minimum frequency for mel scale
    pub f_min: f32,
    /// Maximum frequency for mel scale
    pub f_max: Option<f32>,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 24000,
            channels: 1,
            frame_size: 1024,
            hop_length: 256,
            n_mels: 80,
            n_fft: 1024,
            f_min: 0.0,
            f_max: None,
        }
    }
}

/// Configuration for conditioning
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ConditioningConfig {
    /// Lookup table conditioners
    pub lut_conditioners: HashMap<String, LutConfig>,
    /// Tensor conditioners  
    pub tensor_conditioners: HashMap<String, TensorConfig>,
    /// Global conditioning dimension
    pub global_dim: Option<usize>,
}

impl Default for ConditioningConfig {
    fn default() -> Self {
        Self {
            lut_conditioners: HashMap::new(),
            tensor_conditioners: HashMap::new(),
            global_dim: None,
        }
    }
}
