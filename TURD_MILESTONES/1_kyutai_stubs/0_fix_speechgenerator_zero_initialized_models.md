# Fix SpeechGenerator Zero-Initialized Models

## Problem
The `SpeechGenerator::new()` method creates models with zero-initialized weights instead of loading actual trained weights, making it non-functional for real use.

## Location
**File**: `packages/kyutai/src/speech_generator.rs`  
**Lines**: 684-688

## Current Code
```rust
let lm_vb = VarBuilder::zeros(config.dtype, &config.device);
let mimi_vb = VarBuilder::zeros(config.dtype, &config.device);

let tts_model = TtsModel::new(tts_config, lm_vb, mimi_vb)
    .map_err(|e| SpeechGenerationError::ModelInitialization(e.to_string()))?;
```

## Issue Details
- `VarBuilder::zeros()` creates tensors filled with zeros
- This results in models with no learned parameters
- The `new()` constructor becomes unusable for actual speech generation
- Only the `load_from_files()` method properly loads trained weights

## Solution Required
1. **Remove the broken `new()` constructor** - it should not exist if it can't load real weights
2. **Make `load_from_files()` the primary constructor** 
3. **Add validation** to ensure model files exist before attempting to load
4. **Update documentation** to clarify that models must be loaded from files

## Implementation Steps
1. Remove or deprecate `SpeechGenerator::new()` method
2. Rename `load_from_files()` to `new()` or make it the default constructor
3. Update `SpeechGeneratorBuilder::build()` to require model file paths
4. Update all usage examples and documentation
5. Add proper error messages for missing model files

## Priority
**HIGH** - This is a critical functionality issue that makes the primary constructor unusable.

## Acceptance Criteria
- [x] `SpeechGenerator` can only be created with actual model weights
- [x] No zero-initialized model constructors exist
- [x] Clear error messages when model files are missing
- [x] Updated documentation reflects the requirement for model files

## Implementation Completed

### Changes Made
1. **Fixed `SpeechGenerator::new()` constructor** - Now requires model file paths and loads real weights
2. **Added comprehensive file validation** - Validates safetensors format before loading
3. **Implemented robust error handling** - Clear error messages for missing/invalid files
4. **Updated builder pattern** - All constructors now require model files
5. **Deprecated old methods** - `load_from_files()` now delegates to `new()`
6. **Cleaned up imports** - Removed unused `VarBuilder` import

### Key Implementation Details
- Constructor signature: `SpeechGenerator::new<P: AsRef<Path>>(lm_model_path: P, mimi_model_path: P, config: GeneratorConfig)`
- File validation includes existence check, readability check, and safetensors format validation
- Uses `TtsModel::load()` with `VarBuilder::from_mmaped_safetensors()` for proper model loading
- Comprehensive error messages specify which model file failed and why

### Validation Results
- ✅ Kyutai package compiles successfully
- ✅ No zero-initialization code remains
- ✅ File validation working correctly
- ✅ Error handling provides clear feedback

**Status: COMPLETED** ✅