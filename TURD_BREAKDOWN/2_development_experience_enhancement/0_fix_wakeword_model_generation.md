# Task: Fix Wake-Word Model Generation and Training Integration

**Priority**: 🔧 MEDIUM (Development Experience)  
**File**: [`packages/cyterm/build.rs:26-29`](../../packages/cyterm/build.rs#L26)  
**Milestone**: 2_development_experience_enhancement  

## Problem Description

Build script creates zero-byte placeholder model files that cause silent failures:

```rust
// CURRENT (CREATES INVALID FILES):
std::fs::write(&out, []).unwrap(); // zero-byte placeholder
```

**Impact**: 
- Wake-word detection silently fails to load models
- Build succeeds but runtime functionality is broken
- Confusing developer experience with no clear training path

## Success Criteria

- [ ] Replace zero-byte placeholder with proper model validation
- [ ] Create minimal functional model for development builds
- [ ] Integrate rustpotter-cli training workflow
- [ ] Add comprehensive model format validation
- [ ] Provide clear training instructions and error messages
- [ ] Support both custom models and rustpotter .rpw format
- [ ] Graceful build process with helpful developer guidance

## Technical Solution Overview

Enhanced build script with proper model handling:

```rust
fn handle_wake_word_model(model_path: &Path) -> Result<ModelStatus, Box<dyn std::error::Error>>
```

## Dependencies

**Internal Dependencies**:
- INDEPENDENT - Can run in parallel with all other milestones
- No blocking dependencies on language model or video processing

**External Dependencies**:
- `rustpotter-cli` (auto-installed during build)
- `rustpotter = "3.0"` for model training binaries

**Required Files**:
- Modify: `packages/cyterm/build.rs` (replace zero-byte creation)
- Add: `packages/cyterm/src/bin/train_wake_word.rs`
- Add: `packages/cyterm/src/bin/record_samples.rs`
- Add: `packages/cyterm/src/bin/test_wake_word.rs`
- Modify: `packages/cyterm/Cargo.toml` (add training binaries)

## Implementation Steps

1. **Add model validation functions** (1 hour)
   - Validate existing models for proper format
   - Distinguish between rustpotter .rpw and custom format
   - Comprehensive error detection and reporting

2. **Create minimal model generation** (1.5 hours)
   - Generate small functional model with reasonable weights
   - Deterministic generation for consistent builds
   - Proper bias and weight initialization

3. **Enhance build script logic** (1 hour)
   - Replace zero-byte creation with validation workflow
   - Helpful error messages and training instructions
   - Build re-run triggers for model changes

4. **Add training binary integration** (2 hours)
   - train-wake-word binary with rustpotter integration
   - record-samples binary for audio collection
   - test-wake-word binary for validation

5. **Implement comprehensive error handling** (45 minutes)
   - Clear build failure messages with solutions
   - Training workflow guidance
   - Platform-specific instructions

6. **Add developer documentation** (30 minutes)
   - Training guide integration
   - Quick start instructions
   - Troubleshooting common issues

7. **Testing and validation** (1 hour)
   - Build script testing with various model states
   - Training workflow validation
   - Cross-platform build testing

## Validation Requirements

**Build Script Tests**:
```rust
#[test]
fn test_model_validation() {
    let valid_model = create_test_model_data();
    assert!(validate_custom_model(&valid_model).unwrap());
    
    let invalid_model = vec![]; // Empty
    assert!(!validate_custom_model(&invalid_model).unwrap());
}
```

**Training Integration Tests**:
```rust
#[test]
fn test_minimal_model_creation() {
    let model_path = tempfile::NamedTempFile::new()?;
    create_minimal_model(model_path.path())?;
    
    let model_data = std::fs::read(model_path.path())?;
    assert!(!model_data.is_empty());
    assert!(validate_custom_model(&model_data)?);
}
```

**Build Integration Tests**:
```rust
#[test]
fn test_build_script_workflow() {
    let temp_dir = tempfile::tempdir()?;
    let model_path = temp_dir.path().join("wake-word.rpw");
    
    // Should create minimal model when missing
    let status = handle_wake_word_model(&model_path)?;
    assert!(matches!(status, ModelStatus::Valid));
    assert!(model_path.exists());
}
```

## Reference Implementation

**Source**: [`./tmp/rustpotter/README.md`](../../tmp/rustpotter/README.md)  
**Integration**: Existing [`packages/cyterm/src/wake_word/model.rs`](../../packages/cyterm/src/wake_word/model.rs)

**Training Workflow**:
- Sample recording with microphone
- Rustpotter model training
- Validation and testing integration

## Training Binary Implementation

```rust
// packages/cyterm/src/bin/train_wake_word.rs
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Wake-word Model Training");
    
    // Check for training samples
    let samples_dir = PathBuf::from("training_samples");
    if !samples_dir.exists() {
        eprintln!("❌ No training samples found!");
        eprintln!("Run: cargo run --bin record-samples");
        std::process::exit(1);
    }
    
    // Configure and train model
    let config = RustpotterConfig::default();
    let mut detector = Rustpotter::new(&config)?;
    
    // Training implementation...
    
    println!("✅ Model training completed!");
    Ok(())
}
```

## Developer Experience Enhancements

**Clear Error Messages**:
```
❌ WAKE-WORD MODEL REQUIRED

🔧 QUICK START (Development):
   cargo run --bin record-samples
   cargo run --bin train-wake-word
   cargo run --bin test-wake-word

📖 For detailed instructions, see CYRUP_WAKE_GUIDE.md
```

**Build Process Integration**:
- Automatic rustpotter-cli installation
- Helpful warnings with next steps
- Graceful fallbacks for development

## Risk Assessment

**Risk Level**: MEDIUM - Affects development experience and wake-word functionality  
**Estimated Effort**: 6-8 hours  
**Complexity**: Medium (build integration + training workflow)

## Completion Definition

Task is complete when:
1. `cargo check --package cyterm` passes without warnings
2. No zero-byte model files are created during build
3. Minimal functional model is generated for development
4. Training binaries work correctly with sample data
5. Build script provides helpful error messages and guidance
6. Developer documentation is comprehensive and actionable