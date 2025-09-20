//! Text generation methods for the language model

use super::core::LmModel;
use crate::{conditioner::Condition, sampling_config::SamplingConfig};
use candle_core::{Result, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::utils::apply_repeat_penalty;

impl LmModel {
    /// Generate text using the model with default sampling configuration
    pub fn generate(
        &mut self,
        input_ids: &Tensor,
        max_length: usize,
        temperature: f64,
        top_k: Option<usize>,
        conditions: Option<&[Condition]>,
    ) -> Result<Tensor> {
        // Convert legacy parameter format to modern SamplingConfig for backward compatibility
        let sampling = match top_k {
            Some(k) => Sampling::TopK { k, temperature },
            None => Sampling::All { temperature },
        };
        let config = SamplingConfig::custom(sampling, Some(1.1), 64, 42);
        self.generate_with_config(input_ids, max_length, &config, conditions)
    }

    /// Generate text using the model with advanced sampling configuration
    pub fn generate_with_config(
        &mut self,
        input_ids: &Tensor,
        max_length: usize,
        config: &SamplingConfig,
        conditions: Option<&[Condition]>,
    ) -> Result<Tensor> {
        let mut generated = input_ids.clone();
        let mut context: Vec<u32> = Vec::new();

        // Extract initial context from input_ids if possible
        if let Ok(input_vec) = input_ids.to_vec2::<u32>() {
            if !input_vec.is_empty() {
                context = input_vec[0].clone();
            }
        }

        for _ in 0..max_length {
            // Forward pass
            let logits = self.forward(&generated, conditions)?;

            // Get last token logits
            let seq_len = logits.dim(1)?;
            let last_logits = logits.narrow(1, seq_len - 1, 1)?.squeeze(1)?;

            // Sample next token using advanced sampling
            let next_token_id = self.sample_from_logits(&last_logits, config, &context)?;

            // Update context for repetition penalty
            context.push(next_token_id);

            // Convert token ID to tensor and concatenate
            let next_token = Tensor::new(&[next_token_id], &self.device)?
                .unsqueeze(0)?
                .unsqueeze(1)?;
            generated = Tensor::cat(&[&generated, &next_token], 1)?;
        }

        Ok(generated)
    }
    /// Sample from logits using advanced sampling strategies with repetition penalty
    pub(super) fn sample_from_logits(
        &self,
        logits: &Tensor,
        config: &SamplingConfig,
        context: &[u32],
    ) -> Result<u32> {
        // Apply repetition penalty if configured
        let mut processed_logits = if let Some(penalty) = config.repetition_penalty {
            let context_slice = if context.len() > config.repetition_context_size {
                &context[context.len() - config.repetition_context_size..]
            } else {
                context
            };
            apply_repeat_penalty(logits, penalty, context_slice)?
        } else {
            logits.clone()
        };

        // Apply top-k filtering if using TopK sampling
        if let Sampling::TopK { k, .. } = &config.sampling {
            processed_logits = self.top_k_filter(&processed_logits, *k)?;
        }

        // Use LogitsProcessor for advanced sampling
        let mut processor = LogitsProcessor::from_sampling(config.seed, config.sampling.clone());
        processor.sample(&processed_logits)
    }

    pub(super) fn top_k_filter(&self, logits: &Tensor, k: usize) -> Result<Tensor> {
        let vocab_size = logits.dim(1)?;
        if k >= vocab_size {
            return Ok(logits.clone());
        }

        // Implement proper top-k sampling using Candle operations
        let batch_size = logits.dim(0)?;
        let mut filtered_logits = Vec::new();

        for batch_idx in 0..batch_size {
            let batch_logits = logits.get(batch_idx)?;

            // Get the top-k values and indices using Candle operations
            // First convert to vec, then sort to find top-k
            let logits_vec = batch_logits.to_vec1::<f32>()?;
            let mut indexed_logits: Vec<(usize, f32)> = logits_vec
                .iter()
                .enumerate()
                .map(|(i, &v)| (i, v))
                .collect();

            // Sort by logits value in descending order
            indexed_logits
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Take top-k
            let top_k_pairs: Vec<(usize, f32)> = indexed_logits.into_iter().take(k).collect();
            let top_indices_vec: Vec<u32> = top_k_pairs.iter().map(|(i, _)| *i as u32).collect();
            let top_values_vec: Vec<f32> = top_k_pairs.iter().map(|(_, v)| *v).collect();

            // Create a mask tensor filled with negative infinity
            let neg_inf = f32::NEG_INFINITY;
            let mut mask = vec![neg_inf; vocab_size];

            // Set the top-k positions to their original values
            for (idx, &token_idx) in top_indices_vec.iter().enumerate() {
                mask[token_idx as usize] = top_values_vec[idx];
            }

            // Convert back to tensor
            let filtered_batch = Tensor::from_vec(mask, (vocab_size,), logits.device())?;
            filtered_logits.push(filtered_batch);
        }

        // Stack the filtered logits back into a batch
        Tensor::stack(&filtered_logits, 0)
    }
}
