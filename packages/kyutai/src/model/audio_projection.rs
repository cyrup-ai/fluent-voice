//! Audio output projection for multi-codebook audio generation

use candle_core::{Result, Tensor};
use candle_nn::{Linear, VarBuilder};

/// Audio output projection for multi-codebook audio generation
#[derive(Debug)]
pub struct AudioOutputProjection {
    /// Audio codebook projections (one per codebook)
    codebook_projections: Vec<Linear>,
    /// Number of audio codebooks
    num_codebooks: usize,
    /// Audio vocabulary size per codebook
    audio_vocab_size: usize,
}

impl AudioOutputProjection {
    /// Create a new audio output projection
    pub fn new(
        d_model: usize,
        audio_vocab_size: usize,
        num_codebooks: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut codebook_projections = Vec::with_capacity(num_codebooks);

        for i in 0..num_codebooks {
            let proj = candle_nn::linear(
                d_model,
                audio_vocab_size,
                vb.pp(&format!("audio_proj_{}", i)),
            )?;
            codebook_projections.push(proj);
        }

        Ok(Self {
            codebook_projections,
            num_codebooks,
            audio_vocab_size,
        })
    }

    /// Forward pass through all codebook projections
    pub fn forward(&self, hidden_states: &Tensor) -> Result<Vec<Tensor>> {
        let mut audio_logits = Vec::with_capacity(self.num_codebooks);

        for projection in &self.codebook_projections {
            let logits = hidden_states.apply(projection)?;
            audio_logits.push(logits);
        }

        Ok(audio_logits)
    }

    /// Get the number of codebooks
    pub fn num_codebooks(&self) -> usize {
        self.num_codebooks
    }

    /// Get the audio vocabulary size
    pub fn audio_vocab_size(&self) -> usize {
        self.audio_vocab_size
    }
}
