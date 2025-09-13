use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use kyutai::{Config, LmConfig, LmModel};
use std::collections::HashMap;

/// Create a test VarBuilder for testing model creation
fn create_test_var_builder() -> Result<VarBuilder<'static>> {
    let device = Device::Cpu;
    let dummy_tensor = Tensor::zeros((1, 1), candle_core::DType::F32, &device)?;
    let mut vars = HashMap::new();
    vars.insert("test".to_string(), dummy_tensor);
    Ok(VarBuilder::from_tensors(
        vars,
        candle_core::DType::F32,
        &device,
    ))
}

#[test]
fn test_config_default_includes_lm_config() {
    let config = Config::default();

    // Verify lm_config is included and has expected values
    assert_eq!(config.lm_config.audio_vocab_size, 2049);
    assert_eq!(config.lm_config.audio_codebooks, 8); // v0_1 default
}

#[test]
fn test_tts_1_6b_config() -> Result<()> {
    let lm_config = LmConfig::tts_1_6b_en_fr();
    let mut config = Config::default();
    config.lm_config = lm_config;

    // Validate configuration
    config.validate()?;

    // Verify audio configuration values
    assert_eq!(config.lm_config.audio_vocab_size, 2049);
    assert_eq!(config.lm_config.audio_codebooks, 32); // From LmConfig, not hardcoded 8

    Ok(())
}

#[test]
fn test_stt_2_6b_config() -> Result<()> {
    let lm_config = LmConfig::stt_2_6b_en();
    let mut config = Config::default();
    config.lm_config = lm_config;

    // Validate configuration
    config.validate()?;

    // Verify audio configuration values
    assert_eq!(config.lm_config.audio_vocab_size, 2049);
    assert_eq!(config.lm_config.audio_codebooks, 32);

    Ok(())
}

#[test]
fn test_v0_1_config() -> Result<()> {
    let lm_config = LmConfig::v0_1();
    let mut config = Config::default();
    config.lm_config = lm_config;

    // Validate configuration
    config.validate()?;

    // Verify audio configuration values
    assert_eq!(config.lm_config.audio_vocab_size, 2049);
    assert_eq!(config.lm_config.audio_codebooks, 8);

    Ok(())
}

#[test]
fn test_tts_202501_config() -> Result<()> {
    let lm_config = LmConfig::tts_202501();
    let mut config = Config::default();
    config.lm_config = lm_config;

    // Validate configuration
    config.validate()?;

    // Verify audio configuration values
    assert_eq!(config.lm_config.audio_vocab_size, 2049);
    assert_eq!(config.lm_config.audio_codebooks, 32);

    Ok(())
}

#[test]
fn test_s2s_2b_16rvq_202501_config() -> Result<()> {
    let lm_config = LmConfig::s2s_2b_16rvq_202501();
    let mut config = Config::default();
    config.lm_config = lm_config;

    // Validate configuration
    config.validate()?;

    // Verify audio configuration values
    assert_eq!(config.lm_config.audio_vocab_size, 2049);
    assert_eq!(config.lm_config.audio_codebooks, 32);

    Ok(())
}

#[test]
fn test_invalid_audio_vocab_size() {
    let mut lm_config = LmConfig::v0_1();
    lm_config.audio_vocab_size = 0; // Invalid

    let mut config = Config::default();
    config.lm_config = lm_config;

    let result = config.validate();
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("audio_vocab_size must be > 0"));
}

#[test]
fn test_invalid_audio_codebooks_zero() {
    let mut lm_config = LmConfig::v0_1();
    lm_config.audio_codebooks = 0; // Invalid

    let mut config = Config::default();
    config.lm_config = lm_config;

    let result = config.validate();
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        err.to_string()
            .contains("audio_codebooks must be between 1 and 64")
    );
}

#[test]
fn test_invalid_audio_codebooks_too_large() {
    let mut lm_config = LmConfig::v0_1();
    lm_config.audio_codebooks = 65; // Invalid (too large)

    let mut config = Config::default();
    config.lm_config = lm_config;

    let result = config.validate();
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        err.to_string()
            .contains("audio_codebooks must be between 1 and 64")
    );
}

#[test]
fn test_valid_audio_codebooks_boundary_values() -> Result<()> {
    // Test minimum valid value
    let mut lm_config = LmConfig::v0_1();
    lm_config.audio_codebooks = 1; // Valid minimum

    let mut config = Config::default();
    config.lm_config = lm_config;

    config.validate()?; // Should not error

    // Test maximum valid value
    let mut lm_config = LmConfig::v0_1();
    lm_config.audio_codebooks = 64; // Valid maximum

    let mut config = Config::default();
    config.lm_config = lm_config;

    config.validate()?; // Should not error

    Ok(())
}

#[test]
fn test_all_predefined_configs_are_valid() -> Result<()> {
    let configs = vec![
        LmConfig::tts_1_6b_en_fr(),
        LmConfig::stt_2_6b_en(),
        LmConfig::v0_1(),
        LmConfig::v0_1_vision(),
        LmConfig::tts_v0_1(),
        LmConfig::s2s_v0_1(),
        LmConfig::asr_v0_1_1b(),
        LmConfig::asr_300m_202501(),
        LmConfig::tts_202501(),
        LmConfig::s2s_2b_16rvq_202501(),
    ];

    for lm_config in configs {
        let mut config = Config::default();
        config.lm_config = lm_config;

        // Each predefined config should be valid
        config
            .validate()
            .map_err(|e| anyhow::anyhow!("Config validation failed: {}", e))?;

        // Verify audio configuration is reasonable
        assert!(config.lm_config.audio_vocab_size > 0);
        assert!(config.lm_config.audio_codebooks > 0);
        assert!(config.lm_config.audio_codebooks <= 64);
    }

    Ok(())
}

#[test]
fn test_config_with_streaming_variants() -> Result<()> {
    let configs = vec![
        LmConfig::v0_1_streaming(16),
        LmConfig::v0_1_vision_streaming(16),
        LmConfig::s2s_v0_1_streaming(16),
    ];

    for lm_config in configs {
        let mut config = Config::default();
        config.lm_config = lm_config;

        // Each streaming config should be valid
        config
            .validate()
            .map_err(|e| anyhow::anyhow!("Streaming config validation failed: {}", e))?;

        // Streaming variants typically have 16 codebooks
        assert_eq!(config.lm_config.audio_codebooks, 16);
        assert_eq!(config.lm_config.audio_vocab_size, 2049);
    }

    Ok(())
}
