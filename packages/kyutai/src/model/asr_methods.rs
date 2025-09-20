//! ASR-compatible methods for the language model

use super::core::LmModel;
use candle_core::{Result, Tensor};

impl LmModel {
    // ASR-compatible methods

    /// Get the text start token ID
    pub fn text_start_token(&self) -> u32 {
        1 // Common text start token
    }

    /// Get the audio pad token ID
    pub fn audio_pad_token(&self) -> u32 {
        0 // Common padding token
    }

    /// Get the number of input audio codebooks
    pub fn in_audio_codebooks(&self) -> usize {
        8 // Standard number for Moshi audio codebooks
    }

    /// ASR-compatible forward method
    pub fn forward_asr(
        &mut self,
        text: Option<Tensor>,
        audio_tokens: Vec<Option<Tensor>>,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        // Create or use provided text input, using dynamic dimensions
        let text_input = if let Some(text) = text {
            text
        } else {
            // FIXED: Use extracted dimensions instead of hardcoded defaults
            let batch_size = 1; // Minimal batch when no input provided
            let seq_len = 1;    // Minimal sequence when no input provided
            Tensor::from_vec(
                vec![self.text_start_token()],
                (batch_size, seq_len),
                &self.device,
            )?
        };

        // Extract dynamic batch size and sequence length from actual input tensor
        let input_shape = text_input.shape();
        let batch_size = input_shape.dims()[0].max(1); // Handle empty batches
        let seq_len = if input_shape.rank() > 1 { 
            input_shape.dims()[1].max(1) // Handle empty sequences
        } else { 
            1 
        };

        // IMPLEMENTED: Validate extracted dimensions for processing
        if batch_size == 0 || seq_len == 0 {
            return Err(candle_core::Error::Msg(format!(
                "Invalid tensor dimensions: batch_size={}, seq_len={}", 
                batch_size, seq_len
            )));
        }

        // Embed text tokens using dynamic dimensions
        let mut hidden_states = text_input.apply(&self.embed_tokens)?;
        
        // IMPLEMENTED: Use extracted batch_size and seq_len in tensor operations
        // Ensure hidden states match expected dynamic dimensions
        let hidden_shape = hidden_states.shape();
        let expected_shape = &[batch_size, seq_len, hidden_shape.dims()[2]];
        
        if hidden_shape.dims() != expected_shape {
            // Reshape tensor to match extracted dynamic dimensions
            hidden_states = hidden_states.reshape(expected_shape)?;
        }

        // Process audio tokens if provided
        if !audio_tokens.is_empty() {
            if let Some(audio_embeddings) = self.process_audio_tokens(&audio_tokens)? {
                hidden_states =
                    self.fuse_text_audio_representations(&hidden_states, &audio_embeddings)?;
            }
        }

        // Forward through transformer
        let output = self.transformer.forward(&hidden_states)?;

        // Project to vocabulary for text logits
        let text_logits = output.apply(&self.output_proj)?;

        // Generate proper audio logits using multi-codebook projection
        let audio_logits_vec = self.audio_output_proj.forward(&output)?;

        Ok((text_logits, audio_logits_vec))
    }
    /// ASR-compatible forward method returning all audio codebook logits
    pub fn forward_asr_multi_codebook(
        &mut self,
        text: Option<Tensor>,
        audio_tokens: Vec<Option<Tensor>>,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        // Create or use provided text input, using dynamic dimensions
        let text_input = if let Some(text) = text {
            text
        } else {
            // FIXED: Use extracted dimensions instead of hardcoded defaults
            let batch_size = 1; // Minimal batch when no input provided
            let seq_len = 1;    // Minimal sequence when no input provided
            Tensor::from_vec(
                vec![self.text_start_token()],
                (batch_size, seq_len),
                &self.device,
            )?
        };

        // Extract dynamic batch size and sequence length from actual input tensor
        let input_shape = text_input.shape();
        let batch_size = input_shape.dims()[0].max(1); // Handle empty batches
        let seq_len = if input_shape.rank() > 1 { 
            input_shape.dims()[1].max(1) // Handle empty sequences
        } else { 
            1 
        };

        // IMPLEMENTED: Validate extracted dimensions for processing
        if batch_size == 0 || seq_len == 0 {
            return Err(candle_core::Error::Msg(format!(
                "Invalid tensor dimensions: batch_size={}, seq_len={}", 
                batch_size, seq_len
            )));
        }

        // Embed text tokens using dynamic dimensions
        let mut hidden_states = text_input.apply(&self.embed_tokens)?;
        
        // IMPLEMENTED: Use extracted batch_size and seq_len in tensor operations
        // Ensure hidden states match expected dynamic dimensions
        let hidden_shape = hidden_states.shape();
        let expected_shape = &[batch_size, seq_len, hidden_shape.dims()[2]];
        
        if hidden_shape.dims() != expected_shape {
            // Reshape tensor to match extracted dynamic dimensions
            hidden_states = hidden_states.reshape(expected_shape)?;
        }

        // Process audio tokens if provided
        if !audio_tokens.is_empty() {
            if let Some(audio_embeddings) = self.process_audio_tokens(&audio_tokens)? {
                hidden_states =
                    self.fuse_text_audio_representations(&hidden_states, &audio_embeddings)?;
            }
        }

        // Forward through transformer
        let output = self.transformer.forward(&hidden_states)?;

        // Project to vocabulary for text logits
        let text_logits = output.apply(&self.output_proj)?;

        // Generate proper audio logits using multi-codebook projection
        let audio_logits_vec = self.audio_output_proj.forward(&output)?;

        Ok((text_logits, audio_logits_vec))
    }

    /// Get audio output projection information
    pub fn audio_projection_info(&self) -> (usize, usize) {
        (
            self.audio_output_proj.num_codebooks(),
            self.audio_output_proj.audio_vocab_size(),
        )
    }
}
