# TASK4: Wire Model Management System

## Status: INTEGRATION - Connect ProgressHub to Fluent-Voice

### Objective
Wire the existing `models.rs` model downloading system into the fluent-voice engine initialization to handle Kyutai model management.

### THE OTHER END OF THE WIRE: Fluent-Voice Engine Traits to Implement

**Primary Traits**: From `/Volumes/samsung_t9/fluent-voice/packages/fluent-voice/src/stt_conversation.rs`

```rust
// Engine registration trait
pub trait SttEngine: Send + Sync {
    type Conv: SttConversationBuilder;
    fn conversation(&self) -> Self::Conv;
}
```

**TTS Engine Trait**: From `/Volumes/samsung_t9/fluent-voice/packages/domain/src/tts_engine.rs`

```rust
pub trait TtsEngine: Send + Sync {
    type Conv: TtsConversationBuilder;
    fn conversation(&self) -> Self::Conv;
}
```

**Entry Point Trait**: From `/Volumes/samsung_t9/fluent-voice/packages/fluent-voice/src/fluent_voice.rs`

```rust
pub trait FluentVoice {
    fn tts() -> impl TtsConversationBuilder;
    fn stt() -> impl SttConversationBuilder;
}
```

**Implementation Target**: `KyutaiEngine` in `src/engine.rs` must implement:
1. `SttEngine` - Provides STT conversation builders
2. `TtsEngine` - Provides TTS conversation builders  
3. Static methods for `FluentVoice::tts()` and `FluentVoice::stt()`

### Existing Working Implementation
**File: `src/models.rs`**
- Complete `KyutaiModelManager` with progresshub integration (lines 47-239)
- Downloads `"kyutai/moshika-pytorch-bf16"` and `"kyutai/tts-voices"`
- Returns `KyutaiModelPaths` with validated file locations
- Proper error handling and file validation

### Correct ProgressHub API Usage
**Reference**: `/Volumes/samsung_t9/progresshub/packages/examples/default-cli/src/main.rs`
```rust
let results = ProgressHub::builder()
    .model(&model_id)
    .with_cli_progress()
    .download()
    .await?;
```

### Integration Points

#### 1. Engine Initialization
**Wire into**: `KyutaiEngine::new()` in `engine.rs`
```rust
impl KyutaiEngine {
    pub async fn new() -> Result<Self, MoshiError> {
        // Use existing model manager
        let manager = KyutaiModelManager::new();
        let model_paths = manager.download_models().await?;
        
        // Store paths for builder use
        Ok(Self {
            model_paths,
            // ... other fields
        })
    }
}
```

#### 2. Builder Access to Models
**Provide to builders**: Model paths for TTS and STT initialization
```rust
impl KyutaiEngine {
    pub fn tts(&self) -> KyutaiTtsConversationBuilder {
        KyutaiTtsConversationBuilder {
            model_paths: self.model_paths.clone(),
            // ... other fields
        }
    }
    
    pub fn stt(&self) -> KyutaiSttConversationBuilder {
        KyutaiSttConversationBuilder {
            model_paths: self.model_paths.clone(),
            // ... other fields
        }
    }
}
```

#### 3. Custom Configuration Support
**Use existing**: `KyutaiModelConfig` for custom model repositories
```rust
impl KyutaiEngine {
    pub async fn with_config(config: KyutaiModelConfig) -> Result<Self, MoshiError> {
        let manager = KyutaiModelManager::with_config(config);
        let model_paths = manager.download_models().await?;
        // ...
    }
}
```

### Implementation Steps

1. **Modify Engine Constructor**:
   - Add async model downloading to `KyutaiEngine::new()`
   - Store `KyutaiModelPaths` in engine instance
   - Handle download errors gracefully

2. **Pass Paths to Builders**:
   - Provide model paths to TTS and STT builders
   - Ensure builders can access required model files
   - Handle path validation

3. **Configuration Support**:
   - Support custom model repositories via `KyutaiModelConfig`
   - Allow force redownload option
   - Enable different model variants

4. **Error Handling**:
   - Wrap progresshub errors in `MoshiError`
   - Provide clear error messages for download failures
   - Handle network connectivity issues

### Key Integration Code

```rust
// In engine.rs
pub struct KyutaiEngine {
    model_paths: KyutaiModelPaths,
    // ... other fields
}

impl KyutaiEngine {
    pub async fn new() -> Result<Self, MoshiError> {
        let manager = KyutaiModelManager::new();
        let model_paths = manager.download_models().await?;
        
        Ok(Self {
            model_paths,
        })
    }
    
    pub async fn with_custom_models(config: KyutaiModelConfig) -> Result<Self, MoshiError> {
        let manager = KyutaiModelManager::with_config(config);
        let model_paths = manager.download_models().await?;
        
        Ok(Self {
            model_paths,
        })
    }
}

// In builders - use paths from engine
impl KyutaiTtsConversationBuilder {
    pub async fn await(self) -> Result<AudioStream, MoshiError> {
        // Load TTS model using paths
        let model = tts::Model::load(
            &self.model_paths.lm_model_path,
            &self.model_paths.mimi_model_path,
            DType::F32,
            &Device::Cpu,
        )?;
        
        // Use model for generation...
    }
}
```

### Files to Modify
- `src/engine.rs` (KyutaiEngine implementation)

### Files to Use (DO NOT MODIFY)
- `src/models.rs` (complete working model management)

### Testing
- Test model downloading on first run
- Verify cached models work on subsequent runs
- Test with custom model configurations
- Confirm error handling for network issues

### Acceptance Criteria
- ✅ Models download automatically on engine initialization
- ✅ Uses existing `KyutaiModelManager` implementation
- ✅ Supports custom model configurations
- ✅ Provides model paths to TTS/STT builders
- ✅ Handles errors gracefully with clear messages