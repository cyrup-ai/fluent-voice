use candle_core::{Device, Result, Tensor};
use candle_nn::VarBuilder;
use kyutai::{Config, LmModel};

/// Test helper to create a test model
fn create_test_model() -> Result<LmModel> {
    let device = Device::Cpu;
    let config = Config::default();
    let var_map = candle_nn::VarMap::new();
    let vb = VarBuilder::from_varmap(&var_map, candle_core::DType::F32, &device);
    LmModel::new(&config, vb)
}

/// Test helper to create test audio tokens for all codebooks
fn create_test_audio_tokens(num_codebooks: usize, seq_len: usize) -> Result<Vec<Option<Tensor>>> {
    let device = Device::Cpu;
    let mut audio_tokens = Vec::new();
    for codebook in 0..num_codebooks {
        let tokens: Vec<u32> = (0..seq_len).map(|i| 1000 + (codebook * 100) + i).collect();
        let tensor = Tensor::from_vec(tokens, (1, seq_len), &device)?;
        audio_tokens.push(Some(tensor));
    }
    Ok(audio_tokens)
}

/// Test helper to create test text tensor
fn create_test_text_tensor() -> Result<Tensor> {
    let device = Device::Cpu;
    Tensor::from_vec(vec![1u32, 2u32, 3u32], (1, 3), &device)
}

/// Test helper to check if tensor is all zeros
fn is_zero_tensor(tensor: &Tensor) -> Result<bool> {
    let zero_tensor = Tensor::zeros_like(tensor)?;
    let diff = (tensor - &zero_tensor)?;
    let sum = diff.abs()?.sum_all()?;
    let sum_val: f32 = sum.to_scalar()?;
    Ok(sum_val < 1e-6)
}

/// Test helper to assert tensors are close
fn assert_tensor_close(a: &Tensor, b: &Tensor, tolerance: f64) -> Result<()> {
    let diff = (a - b)?;
    let max_diff = diff.abs()?.max(0)?.to_scalar::<f32>()?;
    assert!(
        max_diff < tolerance as f32,
        "Tensors not close: max_diff = {}",
        max_diff
    );
    Ok(())
}

/// Test helper to assert tensors are not equal
fn assert_ne_tensors(a: &Tensor, b: &Tensor) -> Result<()> {
    let diff = (a - b)?;
    let sum_diff = diff.abs()?.sum_all()?.to_scalar::<f32>()?;
    assert!(
        sum_diff > 1e-6,
        "Tensors are equal when they should be different"
    );
    Ok(())
}

#[test]
fn test_audio_token_processing_full_codebooks() -> Result<()> {
    let model = create_test_model()?;

    // Create audio tokens for all 8 codebooks
    let mut audio_tokens = Vec::new();
    for _ in 0..8 {
        let tokens = Tensor::from_vec(vec![100u32, 200u32, 300u32], (1, 3), model.device())?;
        audio_tokens.push(Some(tokens));
    }

    let result = model.process_audio_tokens(&audio_tokens)?;
    assert!(result.is_some());

    let embedded = result.unwrap();
    assert_eq!(embedded.dims(), &[1, 3, model.config().d_model]); // batch=1, seq=3, d_model
    assert!(!is_zero_tensor(&embedded)?); // Should not be zeros!

    Ok(())
}

#[test]
fn test_audio_token_processing_partial_codebooks() -> Result<()> {
    let model = create_test_model()?;

    // Create audio tokens for only some codebooks (sparse)
    let mut audio_tokens = vec![None; 8];
    audio_tokens[0] = Some(Tensor::from_vec(
        vec![100u32, 200u32],
        (1, 2),
        model.device(),
    )?);
    audio_tokens[3] = Some(Tensor::from_vec(
        vec![300u32, 400u32],
        (1, 2),
        model.device(),
    )?);
    audio_tokens[7] = Some(Tensor::from_vec(
        vec![500u32, 600u32],
        (1, 2),
        model.device(),
    )?);

    let result = model.process_audio_tokens(&audio_tokens)?;
    assert!(result.is_some());

    let embedded = result.unwrap();
    assert_eq!(embedded.dims(), &[1, 2, model.config().d_model]);
    assert!(!is_zero_tensor(&embedded)?); // Should not be zeros!

    Ok(())
}

#[test]
fn test_audio_token_processing_empty() -> Result<()> {
    let model = create_test_model()?;

    // Empty audio tokens
    let audio_tokens = vec![];
    let result = model.process_audio_tokens(&audio_tokens)?;
    assert!(result.is_none());

    // All None audio tokens
    let audio_tokens = vec![None; 8];
    let result = model.process_audio_tokens(&audio_tokens)?;
    assert!(result.is_none());

    Ok(())
}

#[test]
fn test_text_audio_fusion() -> Result<()> {
    let model = create_test_model()?;

    let text_hidden = Tensor::ones((1, 10, model.config().d_model), model.device())?;
    let audio_hidden = Tensor::ones((1, 10, model.config().d_model), model.device())? * 2.0;

    let fused = model.fuse_text_audio_representations(&text_hidden, &audio_hidden)?;

    assert_eq!(fused.dims(), &[1, 10, model.config().d_model]);
    // Should be text + audio = 1.0 + 2.0 = 3.0
    let expected = Tensor::ones((1, 10, model.config().d_model), model.device())? * 3.0;
    assert_tensor_close(&fused, &expected, 1e-6)?;

    Ok(())
}

#[test]
fn test_forward_asr_with_audio_tokens() -> Result<()> {
    let mut model = create_test_model()?;
    let text = Some(create_test_text_tensor()?);

    // Create realistic audio tokens
    let mut audio_tokens = Vec::new();
    for codebook in 0..8 {
        let tokens = Tensor::from_vec(
            vec![1000 + codebook, 1100 + codebook, 1200 + codebook],
            (1, 3),
            model.device(),
        )?;
        audio_tokens.push(Some(tokens));
    }

    let (text_logits, audio_logits) = model.forward_asr(text, audio_tokens)?;

    // Verify audio tokens influenced the output
    let (text_logits_no_audio, _) = model.forward_asr(Some(create_test_text_tensor()?), vec![])?;
    assert_ne_tensors(&text_logits, &text_logits_no_audio)?; // Should be different with audio

    assert_eq!(text_logits.dims()[2], model.config().vocab_size);
    let (num_codebooks, audio_vocab_size) = model.audio_projection_info();
    assert_eq!(audio_logits.dims()[2], audio_vocab_size);

    Ok(())
}

#[test]
fn test_forward_asr_multi_codebook_consistency() -> Result<()> {
    let mut model = create_test_model()?;
    let text = Some(create_test_text_tensor()?);
    let audio_tokens = create_test_audio_tokens(8, 5)?; // 8 codebooks, 5 tokens each

    let (text_logits1, audio_logits1) = model.forward_asr(text.clone(), audio_tokens.clone())?;
    let (text_logits2, audio_logits_vec2) = model.forward_asr_multi_codebook(text, audio_tokens)?;

    // Text logits should be identical
    assert_tensor_close(&text_logits1, &text_logits2, 1e-6)?;

    // First audio logits should match
    assert_tensor_close(&audio_logits1, &audio_logits_vec2[0], 1e-6)?;

    // Should have logits for all codebooks
    let (num_codebooks, _) = model.audio_projection_info();
    assert_eq!(audio_logits_vec2.len(), num_codebooks);

    Ok(())
}

#[test]
fn test_variable_sequence_lengths() -> Result<()> {
    let model = create_test_model()?;

    // Create audio tokens with different sequence lengths per codebook
    let mut audio_tokens = Vec::new();
    audio_tokens.push(Some(Tensor::from_vec(
        vec![100u32, 200u32],
        (1, 2),
        model.device(),
    )?)); // len=2
    audio_tokens.push(Some(Tensor::from_vec(
        vec![300u32, 400u32, 500u32],
        (1, 3),
        model.device(),
    )?)); // len=3  
    audio_tokens.push(Some(Tensor::from_vec(
        vec![600u32],
        (1, 1),
        model.device(),
    )?)); // len=1
    audio_tokens.push(None); // Missing codebook

    let result = model.process_audio_tokens(&audio_tokens)?;
    assert!(result.is_some());

    let embedded = result.unwrap();
    assert_eq!(embedded.dims(), &[1, 3, model.config().d_model]); // Should use max seq len = 3

    Ok(())
}

#[test]
fn test_audio_processing_performance() -> Result<()> {
    let model = create_test_model()?;

    // Large audio token sequences
    let mut audio_tokens = Vec::new();
    for _ in 0..8 {
        // 8 codebooks (reduced from 32 for reasonable test time)
        let large_seq: Vec<u32> = (0..100).collect(); // 100 tokens (reduced from 1000)
        let tokens = Tensor::from_vec(large_seq, (1, 100), model.device())?;
        audio_tokens.push(Some(tokens));
    }

    let start = std::time::Instant::now();
    let _result = model.process_audio_tokens(&audio_tokens)?;
    let elapsed = start.elapsed();

    // Should complete in reasonable time (< 1000ms for this size on CPU)
    assert!(
        elapsed.as_millis() < 1000,
        "Audio processing took too long: {:?}",
        elapsed
    );

    Ok(())
}
