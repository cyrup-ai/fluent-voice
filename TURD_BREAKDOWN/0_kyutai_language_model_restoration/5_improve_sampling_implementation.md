# Task: Integrate Existing Advanced Sampling with Model Interface

**Priority**: ⚠️ MODERATE  
**File**: [`packages/kyutai/src/model.rs:460-465`](../../packages/kyutai/src/model.rs#L460)  
**Milestone**: 0_kyutai_language_model_restoration  

## Problem Description

**CRITICAL DISCOVERY**: Advanced sampling functionality already exists in the codebase via `candle_transformers::generation::LogitsProcessor`, but the `LmModel::sample_from_probs()` method uses basic greedy sampling instead of leveraging this existing infrastructure:

```rust
// CURRENT (BASIC) - in model.rs:
fn sample_from_probs(&self, probs: &Tensor) -> Result<Tensor> {
    // Simple greedy sampling for now - can be extended with proper sampling
    let indices = probs.argmax_keepdim(D::Minus1)?;
    Ok(indices)
}

// BUT EXISTING ADVANCED SAMPLING - already used in gen.rs, generator.rs, lm_generate.rs:
let mut logits_processor = LogitsProcessor::new(seed, Some(temperature), Some(top_k));
let token = logits_processor.sample(&logits)?;
```

**Impact**: 
- Inconsistent sampling across the codebase
- Model interface doesn't expose advanced sampling capabilities
- Generation quality varies depending on which code path is used

## Existing Advanced Sampling Infrastructure

### LogitsProcessor Capabilities (Already Available)
**Source**: [`./tmp/candle/candle-transformers/src/generation/mod.rs`](../../tmp/candle/candle-transformers/src/generation/mod.rs)

```rust
#[derive(Clone, PartialEq, Debug)]
pub enum Sampling {
    ArgMax,                                           // Greedy sampling
    All { temperature: f64 },                        // Multinomial with temperature
    TopK { k: usize, temperature: f64 },            // Top-k sampling
    TopP { p: f64, temperature: f64 },              // Nucleus (top-p) sampling
    TopKThenTopP { k: usize, p: f64, temperature: f64 }, // Combined top-k + top-p
    GumbelSoftmax { temperature: f64 },             // Gumbel-Softmax sampling
}

pub struct LogitsProcessor {
    rng: rand::rngs::StdRng,
    sampling: Sampling,
}
```

### Repetition Penalty (Already Available)
**Source**: [`./tmp/candle/candle-transformers/src/utils.rs`](../../tmp/candle/candle-transformers/src/utils.rs)

```rust
pub fn apply_repeat_penalty(logits: &Tensor, penalty: f32, context: &[u32]) -> Result<Tensor>
```

### Current Usage Examples
**Files using LogitsProcessor correctly**:
- [`packages/kyutai/src/gen.rs`](../../packages/kyutai/src/gen.rs#L16) - `LogitsProcessor::new(seed, Some(temperature), Some(top_k))`
- [`packages/kyutai/src/generator.rs`](../../packages/kyutai/src/generator.rs#L30) - `LogitsProcessor::new(seed, None, None)`  
- [`packages/kyutai/src/lm_generate.rs`](../../packages/kyutai/src/lm_generate.rs#L15) - Uses LogitsProcessor for generation

## Success Criteria

- [ ] Replace basic `sample_from_probs()` with LogitsProcessor integration
- [ ] Add configurable sampling strategies to LmModel interface
- [ ] Integrate repetition penalty using existing `apply_repeat_penalty`
- [ ] Unify sampling approach across all generation code paths
- [ ] Add SamplingConfig for consistent configuration
- [ ] Comprehensive testing of integrated sampling
- [ ] Performance validation (should be same or better)

## Technical Solution Overview

**Integration approach** (NOT reimplementation):

```rust
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::utils::apply_repeat_penalty;

#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub sampling: Sampling,
    pub repetition_penalty: Option<f32>,
    pub repetition_context_size: usize,
    pub seed: u64,
}

impl LmModel {
    // Replace basic sampling with LogitsProcessor integration
    fn sample_from_logits(&self, logits: &Tensor, config: &SamplingConfig, context: &[u32]) -> Result<u32> {
        let processed_logits = if let Some(penalty) = config.repetition_penalty {
            apply_repeat_penalty(logits, penalty, context)?
        } else {
            logits.clone()
        };
        
        let mut processor = LogitsProcessor::from_sampling(config.seed, config.sampling.clone());
        processor.sample(&processed_logits)
    }
}
```

## Dependencies

**Internal Dependencies**:
- Existing `candle_transformers::generation::LogitsProcessor` ✅ Already available
- Existing `candle_transformers::utils::apply_repeat_penalty` ✅ Already available
- Current generation infrastructure in gen.rs, generator.rs, lm_generate.rs ✅ Already working

**External Dependencies**:
- None - all functionality already exists in candle-transformers

**Required Files**:
- Modify: `packages/kyutai/src/model.rs` (integrate LogitsProcessor)
- Add: `packages/kyutai/src/sampling_config.rs` (configuration struct)
- Modify: Existing generation files to use unified approach
- Add tests: `packages/kyutai/tests/sampling_integration.rs`

## Implementation Steps

1. **Create SamplingConfig struct** (30 minutes)
   - Wrapper around existing Sampling enum
   - Add repetition penalty configuration
   - Default configurations for common use cases

2. **Replace sample_from_probs method** (1 hour)
   - Remove basic greedy implementation
   - Integrate LogitsProcessor with repetition penalty
   - Maintain backward compatibility

3. **Update LmModel interface** (1 hour)
   - Add sampling configuration to generate() method
   - Update forward_asr methods to use new sampling
   - Consistent sampling across all model methods

4. **Unify generation code paths** (2 hours)
   - Update gen.rs, generator.rs, lm_generate.rs to use unified approach
   - Remove duplicate LogitsProcessor instantiation
   - Consistent configuration across all generation

5. **Add comprehensive tests** (2 hours)
   - Test all existing Sampling variants
   - Test repetition penalty integration
   - Regression tests to ensure no functionality loss

## Validation Requirements

**Integration Tests**:
```rust
#[test]
fn test_sampling_integration() {
    let model = create_test_model()?;
    let logits = create_test_logits(1000)?;
    
    // Test all sampling strategies work
    for sampling in [
        Sampling::ArgMax,
        Sampling::TopK { k: 50, temperature: 1.0 },
        Sampling::TopP { p: 0.9, temperature: 1.0 },
        Sampling::TopKThenTopP { k: 50, p: 0.9, temperature: 1.0 },
    ] {
        let config = SamplingConfig {
            sampling,
            repetition_penalty: Some(1.1),
            repetition_context_size: 64,
            seed: 42,
        };
        
        let token = model.sample_from_logits(&logits, &config, &[1, 2, 3])?;
        assert!(token < 1000); // Valid token ID
    }
}
```

**Regression Tests**:
```rust
#[test]
fn test_generation_consistency() {
    // Ensure existing generation code produces same results
    let mut model = create_test_model()?;
    let prompt = create_test_prompt()?;
    
    // Test that gen.rs, generator.rs, lm_generate.rs all work consistently
    let result1 = generate_audio(&mut model, &mut mimi, &prompt, 10, 1.0, 50, 42)?;
    let result2 = BasicGenerator::new(model, mimi, device, 42).generate(&prompt, 10)?;
    
    // Should produce deterministic results with same seed
    assert_eq!(result1.dims(), result2.dims());
}
```

**Performance Tests**:
```rust
#[test]
fn test_sampling_performance() {
    let model = create_test_model()?;
    let logits = create_large_logits(50000)?; // Large vocabulary
    let config = SamplingConfig::default();
    
    let start = std::time::Instant::now();
    for _ in 0..100 {
        let _token = model.sample_from_logits(&logits, &config, &[])?;
    }
    let duration = start.elapsed();
    
    assert!(duration.as_millis() < 100); // Should be fast
}
```

## Configuration Examples

**Default Configurations**:
```rust
impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            sampling: Sampling::TopKThenTopP { k: 50, p: 0.9, temperature: 1.0 },
            repetition_penalty: Some(1.1),
            repetition_context_size: 64,
            seed: 42,
        }
    }
}

impl SamplingConfig {
    pub fn greedy() -> Self {
        Self {
            sampling: Sampling::ArgMax,
            repetition_penalty: None,
            repetition_context_size: 0,
            seed: 0,
        }
    }
    
    pub fn creative() -> Self {
        Self {
            sampling: Sampling::TopP { p: 0.95, temperature: 1.2 },
            repetition_penalty: Some(1.05),
            repetition_context_size: 128,
            seed: rand::random(),
        }
    }
}
```

## Reference Implementation Patterns

**From existing candle-transformers tests**: [`./tmp/candle/candle-transformers/tests/generation_tests.rs`](../../tmp/candle/candle-transformers/tests/generation_tests.rs)

```rust
// Temperature sampling
let mut logits_process = LogitsProcessor::new(42, Some(0.9), None);
let token = logits_process.sample(&logits)?;

// Top-k sampling  
let mut logits_process = LogitsProcessor::from_sampling(
    42,
    Sampling::TopK { k: 2, temperature: 1.0 }
);
let token = logits_process.sample(&logits)?;

// Top-p sampling
let mut logits_process = LogitsProcessor::new(42, Some(1.0), Some(0.5));
let token = logits_process.sample(&logits)?;
```

**From existing repetition penalty usage**: [`./tmp/candle/candle-wasm-examples/llama2-c/src/bin/m.rs`](../../tmp/candle/candle-wasm-examples/llama2-c/src/bin/m.rs)

```rust
let logits = if self.repeat_penalty == 1. || tokens.is_empty() {
    logits
} else {
    candle_transformers::utils::apply_repeat_penalty(
        &logits,
        self.repeat_penalty,
        tokens
    )?
};
```

## Risk Assessment

**Risk Level**: LOW - Integration of existing functionality  
**Estimated Effort**: 4-6 hours (much less than original estimate)  
**Complexity**: Low-Medium (configuration + integration, not implementation)

## Completion Definition

Task is complete when:
1. `cargo check --package kyutai` passes without warnings
2. All existing generation functionality works unchanged
3. Model interface exposes configurable sampling strategies
4. Repetition penalty integrated using existing utilities
5. All tests pass including regression and performance tests
6. Consistent sampling approach across entire codebase