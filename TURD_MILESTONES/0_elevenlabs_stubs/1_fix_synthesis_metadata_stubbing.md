# Fix Synthesis Metadata Stubbing

## Executive Summary
Replace stubbed synthesis metadata with actual TTS context data in the ElevenLabs timestamp implementation by implementing a comprehensive context propagation architecture.

## Problem Analysis

### Current Stubbing Issues
In [`packages/elevenlabs/src/engine.rs:752-759`](../packages/elevenlabs/src/engine.rs), synthesis metadata is completely stubbed:

```rust
timestamp_metadata.synthesis_metadata = SynthesisMetadata {
    voice_id: "unknown".to_string(), // Context not available in this scope
    model_id: "unknown".to_string(), // Context not available in this scope
    text: "unknown".to_string(),     // Context not available in this scope
    voice_settings: None,            // Context not available in this scope
    output_format: "unknown".to_string(), // Context not available in this scope
    language: None,                  // Context not available in this scope
};
```

### Root Cause Analysis
The context flows through the system but gets lost at the critical points:

1. **TtsBuilder** ([`engine.rs:232`](../packages/elevenlabs/src/engine.rs)) contains all needed context:
   - `text: Option<String>` 
   - `voice_id: String`
   - `model: Model`
   - `voice_settings: Option<InternalVoiceSettings>`
   - `output_format: OutputFormat`
   - `language_code: Option<String>`

2. **Context Flow Breaks Here**: 
   ```
   TtsBuilder::stream_with_timestamps() 
   → generate_with_timestamps() 
   → AudioStreamWithTimestamps::from_audio_with_timestamps()
   → save_with_timestamps() ❌ Context Lost
   ```

3. **AudioStreamWithTimestamps** ([`engine.rs:653`](../packages/elevenlabs/src/engine.rs)) has no context fields:
   ```rust
   pub struct AudioStreamWithTimestamps {
       inner: Pin<Box<dyn Stream<Item = Result<TimestampedAudioChunk>> + Send>>,
       // ❌ No synthesis context stored here
   }
   ```

## Required Solution Architecture

### 1. Create SynthesisContext Struct
Add a new context structure to capture TTS parameters:

```rust
/// Complete synthesis context for timestamp metadata generation
#[derive(Debug, Clone)]
pub struct SynthesisContext {
    pub voice_id: String,
    pub model_id: String,
    pub text: String,
    pub voice_settings: Option<VoiceSettingsSnapshot>,
    pub output_format: String,
    pub language: Option<String>,
}

impl From<&TtsBuilder> for SynthesisContext {
    fn from(builder: &TtsBuilder) -> Self {
        Self {
            voice_id: builder.voice_id.clone(),
            model_id: builder.model.to_string(),
            text: builder.text.clone().unwrap_or_default(),
            voice_settings: builder.voice_settings.as_ref().map(|vs| VoiceSettingsSnapshot {
                stability: vs.stability,
                similarity_boost: vs.similarity_boost,
                style: vs.style,
                use_speaker_boost: vs.use_speaker_boost,
                speed: vs.speed,
            }),
            output_format: builder.output_format.to_string(),
            language: builder.language_code.clone(),
        }
    }
}
```

### 2. Modify AudioStreamWithTimestamps
Add context storage to the stream wrapper:

```rust
pub struct AudioStreamWithTimestamps {
    inner: Pin<Box<dyn Stream<Item = Result<TimestampedAudioChunk>> + Send>>,
    synthesis_context: SynthesisContext, // ✅ Store context here
}
```

### 3. Update Context Propagation Methods

#### A. Modify `from_audio_with_timestamps()` signature:
```rust
fn from_audio_with_timestamps(
    audio_with_timestamps: AudioWithTimestamps,
    synthesis_context: SynthesisContext, // ✅ Accept context
) -> Self
```

#### B. Update `stream_with_timestamps()` to create and pass context:
```rust
pub async fn stream_with_timestamps(self) -> Result<AudioStreamWithTimestamps> {
    let synthesis_context = SynthesisContext::from(&self); // ✅ Create context
    let audio_with_timestamps = self.generate_with_timestamps().await?;
    Ok(AudioStreamWithTimestamps::from_audio_with_timestamps(
        audio_with_timestamps,
        synthesis_context, // ✅ Pass context
    ))
}
```

#### C. Update `save_with_timestamps()` to use stored context:
```rust
pub async fn save_with_timestamps(
    mut self,
    audio_path: impl AsRef<std::path::Path>,
    timestamps_path: impl AsRef<std::path::Path>,
) -> Result<()> {
    // ... existing audio processing ...

    // ✅ Use actual context instead of stubs
    timestamp_metadata.synthesis_metadata = SynthesisMetadata {
        voice_id: self.synthesis_context.voice_id,
        model_id: self.synthesis_context.model_id, 
        text: self.synthesis_context.text,
        voice_settings: self.synthesis_context.voice_settings,
        output_format: self.synthesis_context.output_format,
        language: self.synthesis_context.language,
    };
    
    // ... rest of implementation ...
}
```

## Implementation Steps

### Phase 1: Create Context Infrastructure
1. **Add SynthesisContext struct** to [`timestamp_metadata.rs`](../packages/elevenlabs/src/timestamp_metadata.rs)
   - Define all fields matching TtsBuilder context
   - Implement `From<&TtsBuilder>` trait
   - Add proper documentation and examples

2. **Import necessary types**:
   - Import `Model` enum from appropriate module
   - Import `OutputFormat` from shared module
   - Import `InternalVoiceSettings` type

### Phase 2: Modify AudioStreamWithTimestamps
1. **Add context field** to struct in [`engine.rs:653`](../packages/elevenlabs/src/engine.rs)
2. **Update constructor methods**:
   - `new()` method to accept context parameter
   - `from_audio_with_timestamps()` to accept and store context
3. **Preserve existing stream functionality**

### Phase 3: Update Context Propagation Chain
1. **Modify `stream_with_timestamps()`** ([`engine.rs:460`](../packages/elevenlabs/src/engine.rs))
   - Create `SynthesisContext::from(&self)` before consuming self
   - Pass context to `from_audio_with_timestamps()`

2. **Update `save_with_timestamps()`** ([`engine.rs:723`](../packages/elevenlabs/src/engine.rs))
   - Replace all "unknown" placeholders with `self.synthesis_context` fields
   - Ensure proper type conversions (.to_string() where needed)

### Phase 4: Handle Edge Cases
1. **Context validation**: Ensure required fields (text, voice_id) are present
2. **Ownership handling**: Clone context appropriately for async operations
3. **Error propagation**: Handle missing context gracefully

## Success Criteria
- [ ] No "unknown" values in saved timestamp metadata
- [ ] `voice_id` matches TTS request voice ID
- [ ] `model_id` matches TTS request model (e.g., "eleven_multilingual_v2")  
- [ ] `text` matches original synthesis text
- [ ] `voice_settings` match TTS configuration (stability, similarity, etc.)
- [ ] `output_format` matches TTS format setting (e.g., "mp3_44100_128")
- [ ] `language` matches specified language code (when provided)
- [ ] All existing functionality preserved (streaming, saving, export)
- [ ] Compilation succeeds with no warnings
- [ ] Integration tests pass with real ElevenLabs API calls

## Dependencies & Constraints

### Prerequisites
- Must be completed before [audio chunk timing calculations](./2_implement_audio_chunk_timing_calculations.md)
- Requires understanding of [TtsBuilder state management](../packages/elevenlabs/src/engine.rs)

### Architecture Impact
**MEDIUM** - Changes method signatures but preserves existing API contracts through careful parameter addition

### Potential Complications
1. **Async ownership**: SynthesisContext must be cloneable for async stream operations
2. **Backward compatibility**: Ensure fluent API surface remains unchanged
3. **Memory efficiency**: Context should be lightweight since it's stored in streams

## Testing Strategy

### Unit Tests
```rust
#[tokio::test]
async fn test_synthesis_context_propagation() {
    let builder = TtsBuilder::new(client)
        .voice("voice_123")
        .text("Hello world")
        .model(Model::ElevenMultilingualV2);
    
    let stream = builder.stream_with_timestamps().await.unwrap();
    
    // Verify context is properly stored
    assert_eq!(stream.synthesis_context.voice_id, "voice_123");
    assert_eq!(stream.synthesis_context.text, "Hello world");
    assert_eq!(stream.synthesis_context.model_id, "eleven_multilingual_v2");
}

#[tokio::test] 
async fn test_metadata_population() {
    // Test full TTS → timestamp → save workflow
    // Verify no "unknown" values in saved JSON
}
```

### Integration Tests
1. **Full workflow test**: TTS generation → timestamp saving → metadata verification
2. **Context preservation**: Verify context survives async stream operations
3. **Export format validation**: Ensure SRT/WebVTT exports work with real context

## File References
- **Primary Implementation**: [`packages/elevenlabs/src/engine.rs`](../packages/elevenlabs/src/engine.rs)
- **Context Structure**: [`packages/elevenlabs/src/timestamp_metadata.rs`](../packages/elevenlabs/src/timestamp_metadata.rs)
- **TTS Body Types**: [`packages/elevenlabs/src/endpoints/genai/tts.rs`](../packages/elevenlabs/src/endpoints/genai/tts.rs)
- **Domain Types**: [`packages/domain/src/timestamps.rs`](../packages/domain/src/timestamps.rs)

## Code Quality Requirements
- ✅ All code MUST pass `cargo check --message-format short --quiet` without warnings
- ✅ No `unwrap()` usage - use proper error handling with Result types
- ✅ Comprehensive documentation with examples
- ✅ Follow existing patterns in [`timestamp_metadata.rs`](../packages/elevenlabs/src/timestamp_metadata.rs)
- ✅ Use `VoiceSettingsSnapshot` from existing structures
- ✅ Implement proper `Clone` traits for async stream compatibility