use candle_transformers::generation::Sampling;

/// Configuration for sampling strategies in language model generation
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// The sampling strategy to use
    pub sampling: Sampling,
    /// Repetition penalty factor (1.0 = no penalty, >1.0 = penalize repetition)
    pub repetition_penalty: Option<f32>,
    /// Number of recent tokens to consider for repetition penalty
    pub repetition_context_size: usize,
    /// Random seed for deterministic sampling
    pub seed: u64,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            sampling: Sampling::TopKThenTopP {
                k: 50,
                p: 0.9,
                temperature: 1.0,
            },
            repetition_penalty: Some(1.1),
            repetition_context_size: 64,
            seed: 42,
        }
    }
}

impl SamplingConfig {
    /// Create a greedy sampling configuration (deterministic, no randomness)
    pub fn greedy() -> Self {
        Self {
            sampling: Sampling::ArgMax,
            repetition_penalty: None,
            repetition_context_size: 0,
            seed: 0,
        }
    }

    /// Create a creative sampling configuration (high temperature, nucleus sampling)
    pub fn creative() -> Self {
        Self {
            sampling: Sampling::TopP {
                p: 0.95,
                temperature: 1.2,
            },
            repetition_penalty: Some(1.05),
            repetition_context_size: 128,
            seed: rand::random(),
        }
    }

    /// Create a balanced sampling configuration (moderate temperature, top-k + top-p)
    pub fn balanced() -> Self {
        Self {
            sampling: Sampling::TopKThenTopP {
                k: 40,
                p: 0.85,
                temperature: 0.8,
            },
            repetition_penalty: Some(1.1),
            repetition_context_size: 64,
            seed: rand::random(),
        }
    }

    /// Create a focused sampling configuration (low temperature, top-k only)
    pub fn focused() -> Self {
        Self {
            sampling: Sampling::TopK {
                k: 20,
                temperature: 0.6,
            },
            repetition_penalty: Some(1.15),
            repetition_context_size: 32,
            seed: rand::random(),
        }
    }

    /// Create a sampling configuration with custom parameters
    pub fn custom(
        sampling: Sampling,
        repetition_penalty: Option<f32>,
        repetition_context_size: usize,
        seed: u64,
    ) -> Self {
        Self {
            sampling,
            repetition_penalty,
            repetition_context_size,
            seed,
        }
    }
}
