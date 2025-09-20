//! Helper methods for the language model

use super::core::LmModel;
use candle_core::{Result, Tensor};

impl LmModel {
    /// Process audio tokens with proper multi-codebook handling
    pub(super) fn process_audio_tokens(
        &self,
        audio_tokens: &[Option<Tensor>],
    ) -> Result<Option<Tensor>> {
        if audio_tokens.is_empty() {
            return Ok(None);
        }

        let mut codebook_embeddings = Vec::new();
        let mut max_seq_len = 0;
        let batch_size = 1; // From current implementation

        // Process each codebook
        for (codebook_idx, maybe_tokens) in audio_tokens.iter().enumerate() {
            if let Some(tokens) = maybe_tokens {
                // Embed tokens for this codebook
                let embedded = tokens.apply(&self.embed_tokens)?; // Reuse existing embedding
                max_seq_len = max_seq_len.max(tokens.dim(1)?);
                codebook_embeddings.push((codebook_idx, embedded));
            }
        }

        if codebook_embeddings.is_empty() {
            return Ok(None);
        }

        // Combine all codebook embeddings (sum like Mimi decoder)
        let d_model = self.config.d_model;
        let mut combined_embedding = Tensor::zeros(
            (batch_size, max_seq_len, d_model),
            codebook_embeddings[0].1.dtype(),
            &self.device,
        )?;

        for (_idx, embedding) in codebook_embeddings {
            // Pad/truncate to max_seq_len if needed
            let seq_len = embedding.dim(1)?;
            let padded_embedding = if seq_len < max_seq_len {
                let padding = Tensor::zeros(
                    (batch_size, max_seq_len - seq_len, d_model),
                    embedding.dtype(),
                    &self.device,
                )?;
                Tensor::cat(&[&embedding, &padding], 1)?
            } else if seq_len > max_seq_len {
                embedding.narrow(1, 0, max_seq_len)?
            } else {
                embedding
            };

            // Sum embeddings (following Mimi decode pattern)
            combined_embedding = (combined_embedding + padded_embedding)?;
        }

        Ok(Some(combined_embedding))
    }

    /// Fuse text and audio representations
    pub(super) fn fuse_text_audio_representations(
        &self,
        text_hidden: &Tensor,
        audio_hidden: &Tensor,
    ) -> Result<Tensor> {
        // Strategy 1: Addition (current approach)
        text_hidden.broadcast_add(audio_hidden)
    }
}
