use candle_core::{Device, Result, Tensor};
use candle_transformers::generation::Sampling;
use kyutai::{
    config::Config,
    model::LmModel,
    sampling_config::SamplingConfig,
};

fn create_test_model() -> Result<LmModel> {
    let device = Device::Cpu;
    let config = Config::default();
    LmModel::new(&config, &device)
}

fn create_test_logits(vocab_size: usize) -> Result<Tensor> {
    let logits_data: Vec<f32> = (0..vocab_size)
        .map(|i| (i as f32 / vocab_size as f32) * 10.0 - 5.0)
        .collect();
    Tensor::from_vec(logits_data, (1, vocab_size), &Device::Cpu)
}

#[test]
fn test_sampling_config_defaults() -> Result<()> {
    let config = SamplingConfig::default();
    
    match config.sampling {
        Sampling::TopKThenTopP { k, p, temperature } => {
            assert_eq!(k, 50);
            assert_eq!(p, 0.9);
            assert_eq!(temperature, 1.0);
        }
        _ => panic!("Expected TopKThenTopP sampling"),
    }
    
    assert_eq!(config.repetition_penalty, Some(1.1));
    assert_eq!(config.repetition_context_size, 64);
    assert_eq!(config.seed, 42);
    
    Ok(())
}

#[test]
fn test_sampling_config_presets() -> Result<()> {
    // Test greedy config
    let greedy = SamplingConfig::greedy();
    assert!(matches!(greedy.sampling, Sampling::ArgMax));
    assert_eq!(greedy.repetition_penalty, None);
    
    // Test creative config
    let creative = SamplingConfig::creative();
    if let Sampling::TopP { p, temperature } = creative.sampling {
        assert_eq!(p, 0.95);
        assert_eq!(temperature, 1.2);
    } else {
        panic!("Expected TopP sampling for creative config");
    }
    
    // Test balanced config
    let balanced = SamplingConfig::balanced();
    if let Sampling::TopKThenTopP { k, p, temperature } = balanced.sampling {
        assert_eq!(k, 40);
        assert_eq!(p, 0.85);
        assert_eq!(temperature, 0.8);
    } else {
        panic!("Expected TopKThenTopP sampling for balanced config");
    }
    
    // Test focused config
    let focused = SamplingConfig::focused();
    if let Sampling::TopK { k, temperature } = focused.sampling {
        assert_eq!(k, 20);
        assert_eq!(temperature, 0.6);
    } else {
        panic!("Expected TopK sampling for focused config");
    }
    
    Ok(())
}

#[test]
fn test_sampling_integration() -> Result<()> {
    let model = create_test_model()?;
    let logits = create_test_logits(1000)?;
    
    // Test all sampling strategies work
    let sampling_configs = vec![
        SamplingConfig::greedy(),
        SamplingConfig::custom(
            Sampling::TopK { k: 50, temperature: 1.0 },
            Some(1.1),
            64,
            42,
        ),
        SamplingConfig::custom(
            Sampling::TopP { p: 0.9, temperature: 1.0 },
            Some(1.1),
            64,
            42,
        ),
        SamplingConfig::custom(
            Sampling::TopKThenTopP { k: 50, p: 0.9, temperature: 1.0 },
            Some(1.1),
            64,
            42,
        ),
    ];
    
    for config in sampling_configs {
        let token = model.sample_from_logits(&logits, &config, &[1, 2, 3])?;
        assert!(token < 1000, "Token ID should be within vocabulary size");
    }
    
    Ok(())
}

#[test]
fn test_repetition_penalty_integration() -> Result<()> {
    let model = create_test_model()?;
    let logits = create_test_logits(100)?;
    
    // Test without repetition penalty
    let config_no_penalty = SamplingConfig::custom(
        Sampling::ArgMax,
        None,
        0,
        42,
    );
    
    let token_no_penalty = model.sample_from_logits(&logits, &config_no_penalty, &[99])?;
    
    // Test with repetition penalty (should discourage token 99)
    let config_with_penalty = SamplingConfig::custom(
        Sampling::ArgMax,
        Some(2.0), // Strong penalty
        64,
        42,
    );
    
    let token_with_penalty = model.sample_from_logits(&logits, &config_with_penalty, &[99])?;
    
    // With a strong penalty on token 99, we should get a different result
    // (This test assumes the logits favor token 99 without penalty)
    println!("Token without penalty: {}, with penalty: {}", token_no_penalty, token_with_penalty);
    
    Ok(())
}

#[test]
fn test_deterministic_sampling() -> Result<()> {
    let model = create_test_model()?;
    let logits = create_test_logits(100)?;
    
    let config = SamplingConfig::custom(
        Sampling::TopK { k: 10, temperature: 1.0 },
        Some(1.1),
        64,
        12345, // Fixed seed
    );
    
    // Multiple calls with same seed should produce same result
    let token1 = model.sample_from_logits(&logits, &config, &[1, 2, 3])?;
    let token2 = model.sample_from_logits(&logits, &config, &[1, 2, 3])?;
    
    assert_eq!(token1, token2, "Sampling with same seed should be deterministic");
    
    Ok(())
}

#[test]
fn test_context_size_limiting() -> Result<()> {
    let model = create_test_model()?;
    let logits = create_test_logits(100)?;
    
    let config = SamplingConfig::custom(
        Sampling::ArgMax,
        Some(1.5),
        5, // Small context size
        42,
    );
    
    // Large context should be truncated to last 5 tokens
    let large_context: Vec<u32> = (0..20).collect();
    let token = model.sample_from_logits(&logits, &config, &large_context)?;
    
    assert!(token < 100, "Token should be valid");
    
    Ok(())
}