use crate::error::Result;
use crate::lm::LmModel;
use crate::mimi::Mimi;
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use std::sync::{Arc, Mutex};

/// Generator trait for Moshi models.
pub trait Generator: std::fmt::Debug {
    fn generate(&mut self, prompt: &Tensor, max_length: usize) -> Result<Tensor>;
    fn reset(&mut self) -> Result<()>;
}

/// Basic generator implementation.
pub struct BasicGenerator {
    model: Arc<Mutex<LmModel>>,
    mimi: Arc<Mutex<Mimi>>,
    device: Device,
    logits_processor: LogitsProcessor,
}

impl std::fmt::Debug for BasicGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BasicGenerator")
            .field("model", &self.model)
            .field("mimi", &self.mimi)
            .field("device", &self.device)
            .field("logits_processor", &"LogitsProcessor { ... }")
            .finish()
    }
}

impl BasicGenerator {
    pub fn new(
        model: Arc<Mutex<LmModel>>,
        mimi: Arc<Mutex<Mimi>>,
        device: Device,
        seed: u64,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, None, None);
        Self {
            model,
            mimi,
            device,
            logits_processor,
        }
    }
}

impl Generator for BasicGenerator {
    fn generate(&mut self, prompt: &Tensor, max_length: usize) -> Result<Tensor> {
        let mut generated = prompt.clone();

        for _ in 0..max_length {
            let logits = {
                let model = self.model.lock().map_err(|e| {
                    crate::error::MoshiError::MutexPoisoned(format!(
                        "Model mutex poisoned during generation step: {}",
                        e
                    ))
                })?;
                model.forward(Some(generated.clone()), vec![])?.0
            };
            // Get the last token's logits from the sequence
            let last_token_logits = logits.narrow(1, logits.dim(1)? - 1, 1)?.squeeze(1)?;
            let token_id = self
                .logits_processor
                .sample(&last_token_logits)
                .map_err(|e| {
                    crate::error::MoshiError::Generation(format!("Sampling error: {}", e))
                })?;
            let token_tensor = Tensor::new(&[token_id], &self.device)?.unsqueeze(0)?;
            generated = Tensor::cat(&[generated, token_tensor], 1)?;
        }

        Ok(generated)
    }

    fn reset(&mut self) -> Result<()> {
        // Reset model state with error handling
        {
            let mut model = self.model.lock().map_err(|e| {
                crate::error::MoshiError::MutexPoisoned(format!(
                    "Model mutex poisoned during reset: {}",
                    e
                ))
            })?;
            model.reset_state()?;
        }

        // Reset MIMI state with error handling
        {
            let mut mimi = self.mimi.lock().map_err(|e| {
                crate::error::MoshiError::MutexPoisoned(format!(
                    "MIMI mutex poisoned during reset: {}",
                    e
                ))
            })?;
            mimi.reset_state();
        }

        Ok(())
    }
}
