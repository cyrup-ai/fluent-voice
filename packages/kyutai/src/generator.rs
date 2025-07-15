use crate::error::Result;
use crate::lm::LmModel;
use crate::mimi::Mimi;
use candle::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use std::sync::Arc;

/// Generator trait for Moshi models.
pub trait Generator {
    fn generate(&mut self, prompt: &Tensor, max_length: usize) -> Result<Tensor>;
    fn reset(&mut self);
}

/// Basic generator implementation.
pub struct BasicGenerator {
    model: Arc<LmModel>,
    mimi: Arc<Mimi>,
    device: Device,
    logits_processor: LogitsProcessor,
}

impl BasicGenerator {
    pub fn new(model: Arc<LmModel>, mimi: Arc<Mimi>, device: Device, seed: u64) -> Self {
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
            let logits = self.model.forward(Some(generated.clone()), vec![])?.0;
            let token = self
                .logits_processor
                .sample(&logits.i((0, logits.dim(1)? - 1))?)?;
            let token_tensor = Tensor::new(token, &self.device)?
                .unsqueeze(0)?
                .unsqueeze(0)?;
            generated = Tensor::cat(&[generated, token_tensor], 1)?;
        }

        Ok(generated)
    }

    fn reset(&mut self) {
        self.model.reset_state();
        self.mimi.reset_state();
    }
}
