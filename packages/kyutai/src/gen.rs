use crate::error::Result;
use crate::generator::Generator;
use crate::lm::LmModel;
use crate::mimi::Mimi;
use candle_core::Tensor;
use candle_transformers::generation::LogitsProcessor;

/// General generation functions for Moshi models.
pub fn generate_audio(
    model: &mut LmModel,
    mimi: &mut Mimi,
    prompt: &Tensor,
    max_length: usize,
    temperature: f32,
    top_k: usize,
    seed: u64,
) -> Result<Tensor> {
    let mut logits_processor = LogitsProcessor::new(seed, Some(temperature), Some(top_k));

    let mut generated = vec![prompt.clone()];

    for _ in 0..max_length {
        let input = generated.last().ok_or(crate::error::MoshiError::Custom("No input generated".into()))?.clone();
        let logits = model.forward(&input, None)?.0;
        let token = logits_processor.sample(&logits.i((0, logits.dim(1)? - 1))?)?;
        let token_tensor = Tensor::new(token, logits.device())?.unsqueeze(0)?.unsqueeze(0)?;
        generated.push(token_tensor.clone());

        let audio = mimi.decode(&token_tensor)?;
        // Process audio if needed
    }

    Tensor::cat(&generated, 1).map_err(crate::error::MoshiError::from)
}
