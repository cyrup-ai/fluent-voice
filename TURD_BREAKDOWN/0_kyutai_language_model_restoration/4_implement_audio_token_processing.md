# Task: Implement Proper Audio Token Processing

**Priority**: 🚨 CRITICAL  
**File**: [`packages/kyutai/src/model.rs:320-330, 380-390`](../../packages/kyutai/src/model.rs#L320)  
**Milestone**: 0_kyutai_language_model_restoration  

## Problem Description

Audio token processing is completely stubbed with placeholder zeros instead of proper multi-codebook handling:

```rust
// CURRENT (BROKEN):
if !audio_tokens.is_empty() {
    // For now, we'll create a simple audio conditioning tensor
    // In a full implementation, this would properly handle multi-codebook audio
    let audio_condition = Tensor::zeros(
        (batch_size, seq_len, audio_dim),
        hidden_states.dtype(),
        &self.device,
    )?;
    hidden_states = hidden_states.broadcast_add(&audio_condition)?;
}
```

**Impact**: 
- Audio input completely ignored during processing
- Multi-codebook audio tokens not utilized
- ASR and TTS functionality severely degraded
- Model cannot condition on audio context

## Success Criteria

- [ ] Remove placeholder zero tensor creation
- [ ] Implement proper multi-codebook audio token embedding
- [ ] Support variable-length audio token sequences
- [ ] Add audio token positional encoding
- [ ] Implement proper fusion of text and audio representations
- [ ] Handle missing/partial audio tokens gracefully
- [ ] Comprehensive testing with real audio token sequences

## Technical Solution Overview

Implement proper audio token processing with multi-codebook support:

```rust
impl LmModel {
    fn process_audio_tokens(&self, audio_tokens: &[Option<Tensor>]) -> Result<Option<Tensor>>
    fn embed_audio_codebooks(&self, audio_tokens: &[Tensor]) -> Result<Tensor>
    fn fuse_text_audio_representations(&self, text_hidden: &Tensor, audio_hidden: &Tensor) -> Result<Tensor>
}
```

## Dependencies

**Internal Dependencies**:
- Must complete after hardcoded config fix (task 3)
- Requires proper audio configuration from LmConfig

**External Dependencies**:
- Existing Candle tensor operations
- Audio embedding layers (may need to add to model)

**Required Files**:
- Modify: `packages/kyutai/src/model.rs` (replace stubbed implementation)
- Add: `packages/kyutai/src/audio_processing.rs` (new module)
- Add tests: `packages/kyutai/tests/audio_processing.rs`

## Implementation Steps

1. **Add audio embedding layers** (1.5 hours)
   - Create embedding layers for each audio codebook
   - Proper initialization with VarBuilder
   - Support for different codebook vocabulary sizes

2. **Implement audio token embedding** (2 hours)
   - Process Vec<Option<Tensor>> input format
   - Handle missing codebook tokens gracefully
   - Combine multi-codebook embeddings properly

3. **Add positional encoding for audio** (1 hour)
   - Audio-specific positional encoding
   - Alignment with text sequence positions
   - Configurable encoding schemes

4. **Implement text-audio fusion** (2 hours)
   - Proper fusion strategies (concatenation, addition, cross-attention)
   - Dimension compatibility handling
   - Configurable fusion methods

5. **Add comprehensive error handling** (30 minutes)
   - Validate audio token dimensions
   - Handle empty or malformed audio inputs
   - Meaningful error messages

6. **Optimize for performance** (1 hour)
   - Efficient tensor operations
   - Memory usage optimization
   - Batch processing support

7. **Write extensive tests** (2 hours)
   - Test with various audio token configurations
   - Test missing/partial codebook scenarios
   - Integration tests with full model pipeline

## Validation Requirements

**Unit Tests**:
```rust
#[test]
fn test_audio_token_embedding() {
    let model = create_test_model()?;
    let audio_tokens = create_test_audio_tokens(8, 100)?; // 8 codebooks, 100 tokens
    
    let embedded = model.embed_audio_codebooks(&audio_tokens)?;
    
    assert_eq!(embedded.dims(), &[1, 100, model.config.d_model]);
    assert!(!is_zero_tensor(&embedded)); // Should not be zeros
}
```

**Integration Tests**:
```rust
#[test]
fn test_forward_asr_with_audio() {
    let mut model = create_test_model()?;
    let text = create_test_text_tensor()?;
    let audio_tokens = create_test_audio_tokens(8, 50)?;
    
    let (text_logits, audio_logits) = model.forward_asr(Some(text), audio_tokens)?;
    
    // Verify audio tokens influenced the output
    let (text_logits_no_audio, _) = model.forward_asr(Some(text), vec![])?;
    assert_ne!(text_logits, text_logits_no_audio); // Should be different
}
```

**Missing Token Tests**:
```rust
#[test]
fn test_partial_audio_tokens() {
    let mut model = create_test_model()?;
    let mut audio_tokens = vec![None; 8];
    audio_tokens[0] = Some(create_test_tensor(50)?); // Only first codebook
    audio_tokens[3] = Some(create_test_tensor(50)?); // Only fourth codebook
    
    let result = model.forward_asr(None, audio_tokens);
    assert!(result.is_ok()); // Should handle gracefully
}
```

## Audio Token Format

Expected input format for audio_tokens parameter:
```rust
// Vec<Option<Tensor>> where:
// - Length = number of codebooks (typically 8 or 32)
// - Each Option<Tensor> represents one codebook's tokens
// - None indicates missing codebook data
// - Tensor shape: [batch_size, sequence_length] with token IDs
```

## Fusion Strategies

Support multiple text-audio fusion approaches:
1. **Concatenation**: Concat text and audio along sequence dimension
2. **Addition**: Element-wise addition after projection to same dimension
3. **Cross-attention**: Audio as keys/values, text as queries
4. **Interleaving**: Alternate text and audio tokens in sequence

## Risk Assessment

**Risk Level**: CRITICAL - Core functionality completely broken  
**Estimated Effort**: 8-10 hours  
**Complexity**: High (multi-modal fusion + tensor operations)

## Completion Definition

Task is complete when:
1. `cargo check --package kyutai` passes without warnings
2. Audio tokens are properly processed and embedded
3. Text-audio fusion produces meaningful representations
4. All tests pass including edge cases with missing tokens
5. Performance meets real-time requirements for inference