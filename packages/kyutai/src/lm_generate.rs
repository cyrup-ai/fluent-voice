use super::generator::Generator;
use super::lm::{Config, LmModel};
use super::mimi::Mimi;
use candle::{Result, Tensor};
use candle_transformers::generation::LogitsProcessor;

/// Language model generation state management
#[derive(Debug)]
pub struct LmGenerate {
    pub model: LmModel,
    pub mimi: Mimi,
    pub generator: Box<dyn Generator>,
    pub logits_processor: LogitsProcessor,
}

impl LmGenerate {
    pub fn new(
        model: LmModel,
        mimi: Mimi,
        generator: Box<dyn Generator>,
        logits_processor: LogitsProcessor,
    ) -> Self {
        Self {
            model,
            mimi,
            generator,
            logits_processor,
        }
    }

    pub fn generate_step(&mut self, input: &Tensor) -> Result<Tensor> {
        // Generate one step using the language model
        let logits = self.model.forward(input)?;
        let processed_logits = self.logits_processor.sample(&logits)?;
        Ok(processed_logits)
    }

    pub fn reset(&mut self) -> Result<()> {
        self.generator.reset()
    }
}
