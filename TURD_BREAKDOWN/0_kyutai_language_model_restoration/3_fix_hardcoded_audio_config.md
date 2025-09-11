# Task: Fix Hardcoded Audio Configuration Values

**Priority**: 🚨 CRITICAL  
**File**: [`packages/kyutai/src/model.rs:103-104`](../../packages/kyutai/src/model.rs#L103)  
**Milestone**: 0_kyutai_language_model_restoration  

## Problem Description

The audio output projection uses hardcoded configuration values instead of reading from the existing LmConfig:

```rust
// CURRENT (BROKEN):
let audio_vocab_size = 2049; // Standard audio vocab size for Moshi
let num_codebooks = 8; // Standard number of codebooks for Moshi
```

**But LmConfig already has these fields**:
```rust
pub struct LmConfig {
    pub audio_vocab_size: usize,    // ← Should use this
    pub audio_codebooks: usize,     // ← Should use this
    // ...
}
```

**Impact**: 
- Configuration system completely bypassed
- Model architecture hardcoded to single variant
- Cannot support different Kyutai model sizes (1.6B, 2.6B, etc.)
- Breaks deployment flexibility and model loading

## Success Criteria

- [ ] Remove hardcoded audio_vocab_size and num_codebooks values
- [ ] Integrate LmConfig into Config struct or create proper mapping
- [ ] Update LmModel::new() to use configuration values from LmConfig
- [ ] Add validation for audio configuration parameters
- [ ] Support all existing Kyutai model variants (tts_1_6b_en_fr, stt_2_6b_en, etc.)
- [ ] Comprehensive testing with different model configurations

## Technical Solution Overview

Integrate LmConfig with the existing Config system:

```rust
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    // ... existing fields ...
    pub lm_config: LmConfig,  // Add LmConfig integration
}

impl LmModel {
    pub fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        // Use config values instead of hardcoded
        let audio_vocab_size = config.lm_config.audio_vocab_size;
        let num_codebooks = config.lm_config.audio_codebooks;
        // ...
    }
}
```

## Dependencies

**Internal Dependencies**:
- Must complete after audio logits generation (task 0)
- Can run in parallel with tokenizer and sampling tasks

**External Dependencies**:
- None - uses existing configuration system

**Required Files**:
- Modify: `packages/kyutai/src/config.rs` (integrate LmConfig)
- Modify: `packages/kyutai/src/model.rs` (remove hardcoded values)
- Add tests: `packages/kyutai/tests/config_integration.rs`

## Implementation Steps

1. **Update Config struct** (30 minutes)
   - Add LmConfig field to main Config struct
   - Update Default implementation with reasonable LmConfig defaults

2. **Modify LmModel constructor** (45 minutes)
   - Replace hardcoded values with config.lm_config fields
   - Add validation for audio configuration parameters

3. **Add configuration validation** (30 minutes)
   - Validate audio_vocab_size > 0
   - Validate num_codebooks in reasonable range (1-64)
   - Validate d_model compatibility

4. **Update model loading** (1 hour)
   - Ensure all existing model variants work correctly
   - Test with tts_1_6b_en_fr, stt_2_6b_en, v0_1, etc.

5. **Add comprehensive tests** (1.5 hours)
   - Test all predefined model configurations
   - Test custom configurations
   - Test validation error cases

## Validation Requirements

**Configuration Tests**:
```rust
#[test]
fn test_tts_1_6b_config() {
    let lm_config = LmConfig::tts_1_6b_en_fr();
    let config = Config {
        lm_config,
        ..Default::default()
    };
    
    let vb = create_test_var_builder()?;
    let model = LmModel::new(&config, vb)?;
    
    let (num_codebooks, vocab_size) = model.audio_projection_info();
    assert_eq!(num_codebooks, 32); // From LmConfig, not hardcoded 8
    assert_eq!(vocab_size, 2049);
}
```

**Validation Tests**:
```rust
#[test]
fn test_invalid_audio_config() {
    let mut lm_config = LmConfig::tts_1_6b_en_fr();
    lm_config.audio_vocab_size = 0; // Invalid
    
    let config = Config {
        lm_config,
        ..Default::default()
    };
    
    let vb = create_test_var_builder()?;
    let result = LmModel::new(&config, vb);
    assert!(result.is_err());
}
```

## Model Variants to Support

All existing LmConfig variants must work correctly:
- `LmConfig::tts_1_6b_en_fr()` - 32 codebooks, 2049 vocab
- `LmConfig::stt_2_6b_en()` - 32 codebooks, 2049 vocab  
- `LmConfig::v0_1()` - 8 codebooks, 2049 vocab
- `LmConfig::tts_202501()` - 32 codebooks, 2049 vocab
- `LmConfig::s2s_2b_16rvq_202501()` - 32 codebooks, 2049 vocab

## Risk Assessment

**Risk Level**: CRITICAL - Breaks configuration system  
**Estimated Effort**: 3-4 hours  
**Complexity**: Medium (configuration integration)

## Completion Definition

Task is complete when:
1. `cargo check --package kyutai` passes without warnings
2. All model variants load correctly with proper audio configuration
3. No hardcoded audio configuration values remain
4. Configuration validation prevents invalid setups
5. All tests pass including edge cases