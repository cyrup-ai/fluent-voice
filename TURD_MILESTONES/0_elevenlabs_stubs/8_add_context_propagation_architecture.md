# Add Context Propagation Architecture

## Description
Design and implement proper context propagation from TTS builders to timestamp generation methods.

## Current Problem
The `save_with_timestamps` method has no access to the original TTS builder context (voice_id, model_id, text, etc.), making it impossible to populate metadata correctly.

## Required Solution
Design a context propagation system that passes TTS synthesis parameters through to timestamp generation without breaking existing APIs.

## Implementation Steps

### 1. Define Synthesis Context Structure
```rust
#[derive(Debug, Clone)]
pub struct SynthesisContext {
    pub voice_id: String,
    pub model_id: String,
    pub text: String,
    pub voice_settings: Option<VoiceSettings>,
    pub output_format: OutputFormat,
    pub language_code: Option<String>,
    pub enable_logging: bool,
    pub seed: Option<i32>,
}

impl SynthesisContext {
    pub fn from_tts_builder(builder: &TtsBuilder) -> Self {
        // Extract context from builder state
    }
}
```

### 2. Update Method Signatures
```rust
// Current problematic signature:
pub async fn save_with_timestamps(
    mut self,
    audio_path: impl AsRef<std::path::Path>,
    timestamps_path: impl AsRef<std::path::Path>,
) -> Result<()>

// New signature with context:
pub async fn save_with_timestamps(
    mut self,
    audio_path: impl AsRef<std::path::Path>,
    timestamps_path: impl AsRef<std::path::Path>,
    context: SynthesisContext,
) -> Result<()>
```

### 3. Update Builder Chain
```rust
// TtsBuilder needs to provide context when creating streams
pub async fn stream_with_timestamps(self) -> Result<AudioStreamWithTimestamps> {
    let context = SynthesisContext::from_tts_builder(&self);
    let audio_with_timestamps = self.generate_with_timestamps().await?;
    Ok(AudioStreamWithTimestamps::from_audio_with_timestamps(
        audio_with_timestamps,
        context,
    ))
}
```

### 4. Update AudioStreamWithTimestamps
```rust
impl AudioStreamWithTimestamps {
    fn from_audio_with_timestamps(
        audio_with_timestamps: AudioWithTimestamps, 
        context: SynthesisContext
    ) -> Self {
        // Store context for later use in save_with_timestamps
    }
}
```

## Files to Update
- `packages/elevenlabs/src/engine.rs` - Add SynthesisContext type and builder integration
- `packages/elevenlabs/src/timestamp_metadata.rs` - Use context in metadata population
- Update all timestamp generation methods to accept and use context

## Architectural Considerations
- **Backward Compatibility**: Ensure existing APIs continue to work
- **Memory Usage**: Context should be lightweight and cloneable
- **Thread Safety**: Context must be Send + Sync for async usage
- **API Consistency**: Follow existing fluent-voice patterns

## Success Criteria
- [ ] Context propagates from TTS builder to timestamp generation
- [ ] All synthesis metadata fields populated with real values
- [ ] Existing APIs remain backward compatible
- [ ] Context is available everywhere timestamp metadata is created
- [ ] Memory overhead is minimal
- [ ] Thread safety requirements are met

## Dependencies
- Must be completed before Tasks 1 and 2 can be fully implemented
- Foundation for all metadata accuracy improvements

## Architecture Impact
**HIGH** - Fundamental change to how context flows through timestamp system