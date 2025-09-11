use super::generator::Generator;
use super::lm::LmModel;
use super::mimi::Mimi;
use crate::error::Result;
use candle_core::Tensor;
use candle_transformers::generation::LogitsProcessor;

/// Language model generation state management
pub struct LmGenerate {
    pub model: LmModel,
    pub mimi: Mimi,
    pub generator: Box<dyn Generator>,
    pub logits_processor: LogitsProcessor,
}

impl std::fmt::Debug for LmGenerate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LmGenerate")
            .field("model", &self.model)
            .field("mimi", &self.mimi)
            .field("generator", &self.generator)
            .field("logits_processor", &"LogitsProcessor { ... }")
            .finish()
    }
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
        let (logits, _hidden) = self.model.forward(Some(input.clone()), vec![])?;
        let token_id = self
            .logits_processor
            .sample(&logits)
            .map_err(|e| crate::error::MoshiError::Generation(format!("Sampling error: {}", e)))?;
        // Convert token ID to tensor
        let result = Tensor::new(&[token_id], input.device())?;
        Ok(result)
    }

    pub fn reset(&mut self) -> Result<()> {
        self.generator.reset();
        Ok(())
    }
}
