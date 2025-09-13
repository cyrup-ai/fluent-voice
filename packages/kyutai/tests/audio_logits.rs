//! Unit tests for audio logits generation

use candle_core::{Device, Result, Tensor};
use candle_nn::VarBuilder;
use fluent_voice_kyutai::model::AudioOutputProjection;

#[test]
fn test_audio_output_projection_creation() -> Result<()> {
    let device = Device::Cpu;
    let d_model = 512;
    let audio_vocab_size = 2049;
    let num_codebooks = 8;

    // Create a dummy VarBuilder for testing
    let dummy_tensors = std::collections::HashMap::new();
    let vb = VarBuilder::from_tensors(dummy_tensors, candle_core::DType::F32, &device);

    let projection = AudioOutputProjection::new(d_model, audio_vocab_size, num_codebooks, vb)?;

    assert_eq!(projection.num_codebooks(), num_codebooks);
    assert_eq!(projection.audio_vocab_size(), audio_vocab_size);

    Ok(())
}

#[test]
fn test_audio_output_projection_forward() -> Result<()> {
    let device = Device::Cpu;
    let d_model = 512;
    let audio_vocab_size = 2049;
    let num_codebooks = 8;
    let batch_size = 2;
    let seq_len = 10;

    // Create random tensors for weights and biases
    let mut tensors = std::collections::HashMap::new();

    for i in 0..num_codebooks {
        let weight_key = format!("audio_proj_{}.weight", i);
        let bias_key = format!("audio_proj_{}.bias", i);

        let weight = Tensor::randn(0f32, 1.0, (audio_vocab_size, d_model), &device)?;
        let bias = Tensor::randn(0f32, 1.0, (audio_vocab_size,), &device)?;

        tensors.insert(weight_key, weight);
        tensors.insert(bias_key, bias);
    }

    let vb = VarBuilder::from_tensors(tensors, candle_core::DType::F32, &device);
    let projection = AudioOutputProjection::new(d_model, audio_vocab_size, num_codebooks, vb)?;

    // Create input tensor
    let hidden_states = Tensor::randn(0f32, 1.0, (batch_size, seq_len, d_model), &device)?;

    // Forward pass
    let audio_logits = projection.forward(&hidden_states)?;

    // Verify output shape and properties
    assert_eq!(audio_logits.len(), num_codebooks);

    for (i, logits) in audio_logits.iter().enumerate() {
        let shape = logits.shape();
        assert_eq!(
            shape.dims(),
            &[batch_size, seq_len, audio_vocab_size],
            "Codebook {} has incorrect shape: {:?}",
            i,
            shape.dims()
        );

        // Verify logits are not all zeros (proper projection occurred)
        let sum = logits.sum_all()?.to_scalar::<f32>()?;
        assert!(sum.abs() > 1e-6, "Codebook {} logits are all zeros", i);
    }

    Ok(())
}

#[test]
fn test_different_codebook_counts() -> Result<()> {
    let device = Device::Cpu;
    let d_model = 256;
    let audio_vocab_size = 1024;

    for num_codebooks in [1, 4, 8, 16, 32] {
        let mut tensors = std::collections::HashMap::new();

        for i in 0..num_codebooks {
            let weight_key = format!("audio_proj_{}.weight", i);
            let bias_key = format!("audio_proj_{}.bias", i);

            let weight = Tensor::randn(0f32, 1.0, (audio_vocab_size, d_model), &device)?;
            let bias = Tensor::randn(0f32, 1.0, (audio_vocab_size,), &device)?;

            tensors.insert(weight_key, weight);
            tensors.insert(bias_key, bias);
        }

        let vb = VarBuilder::from_tensors(tensors, candle_core::DType::F32, &device);
        let projection = AudioOutputProjection::new(d_model, audio_vocab_size, num_codebooks, vb)?;

        assert_eq!(projection.num_codebooks(), num_codebooks);

        let hidden_states = Tensor::randn(0f32, 1.0, (1, 1, d_model), &device)?;
        let audio_logits = projection.forward(&hidden_states)?;

        assert_eq!(audio_logits.len(), num_codebooks);
    }

    Ok(())
}
