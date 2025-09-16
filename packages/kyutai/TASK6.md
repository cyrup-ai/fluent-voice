# TASK6: Remove Stub Implementations and Delegate to DefaultEngine

## Status: CLEANUP - Remove Non-Functional Stubs

### Objective
Remove the non-functional wake word and VAD stub implementations from kyutai and ensure proper delegation to the existing DefaultEngine implementations that already work.

### THE OTHER END OF THE WIRE: DefaultEngine Already Provides These

**Wake Word**: Already implemented in `/Volumes/samsung_t9/fluent-voice/packages/fluent-voice/src/fluent_voice.rs`

```rust
fn wake_word() -> impl WakeWordBuilder {
    // Use Koffee as the default wake word implementation
    crate::wake_word_koffee::KoffeeWakeWordBuilder::new()
}
```

**VAD**: Already handled by the default STT engine implementation

```rust
fn conversation(self) -> impl SttConversationBuilder {
    crate::engines::DefaultSTTConversationBuilder::new()
}
```

**Proper Architecture**: Clean separation of concerns:
- **Kyutai**: Provides TTS/STT engine implementations (TASK1-5)
- **DefaultEngine**: Provides wake word (Koffee) and VAD functionality  
- **fluent-voice**: Orchestrates them together

### Current Broken Stub Implementations to Remove

#### 1. Stub Wake Word Builder (lines 1255-1301)
**File: `src/engine.rs`**
```rust
// DELETE THIS ENTIRE SECTION
pub struct KyutaiWakeWordBuilder {
    config: WakeWordConfig,
}

impl WakeWordBuilder for KyutaiWakeWordBuilder {
    // ... stub implementation that does nothing
}
```

#### 2. Stub Wake Word Detector (lines 1303-1320)
**File: `src/engine.rs`**
```rust
// DELETE THIS ENTIRE SECTION
pub struct KyutaiWakeWordDetector {
    config: WakeWordConfig,
}

impl WakeWordDetector for KyutaiWakeWordDetector {
    // ... stub implementation that does nothing
}
```

#### 3. Stub Wake Word Stream (lines 1320+)
**File: `src/engine.rs`**
```rust
// DELETE THIS ENTIRE SECTION
pub struct KyutaiWakeWordStream {
    _config: WakeWordConfig,
    active: bool,
}

impl Stream for KyutaiWakeWordStream {
    fn poll_next(...) -> Poll<Option<Self::Item>> {
        // For now, return None (no wake words detected)
        // This is broken - always returns None!
        std::task::Poll::Ready(None)
    }
}
```

#### 4. Remove Wake Word Method from FluentVoice Implementation
**File: `src/engine.rs`** (around line 189-192)
```rust
// DELETE THIS METHOD
fn wake_word() -> impl WakeWordBuilder {
    KyutaiWakeWordBuilder::new()  // This returns broken stubs
}
```

### Implementation Steps

1. **Remove Stub Structs**:
   - Delete `KyutaiWakeWordBuilder` struct and implementation
   - Delete `KyutaiWakeWordDetector` struct and implementation  
   - Delete `KyutaiWakeWordStream` struct and implementation

2. **Remove FluentVoice wake_word Method**:
   - Delete the `wake_word()` method from `impl FluentVoice for KyutaiFluentVoice`
   - Let it fall through to the default implementation

3. **Clean Up Imports**:
   - Remove wake word related imports that are no longer needed
   - Keep only imports needed for TTS/STT functionality

4. **Update Documentation**:
   - Document that wake word detection uses the default Koffee implementation
   - Clarify that VAD is handled by the default STT engine

### Key Cleanup Code

```rust
// In engine.rs - REMOVE the wake_word method entirely
impl FluentVoice for KyutaiFluentVoice {
    fn tts() -> TtsEntry {
        TtsEntry::new()
    }

    fn stt() -> SttEntry {
        SttEntry::new()
    }

    // DELETE: fn wake_word() -> impl WakeWordBuilder { ... }
    // This will delegate to the default implementation automatically

    fn voices() -> impl VoiceDiscoveryBuilder {
        KyutaiVoiceDiscoveryBuilder::new()
    }

    // ... other methods that kyutai actually implements
}
```

### Result After Cleanup

**Users will get working functionality**:
```rust
// This will use the working Koffee implementation
let wake_word_detector = FluentVoice::wake_word()
    .confidence_threshold(0.8)
    .detect(|result| result)
    .await?;

// This will use kyutai's TTS implementation  
let audio = FluentVoice::tts()
    .conversation()
    .with_speaker(speaker)
    .synthesize(|conv| conv.into_stream())
    .await?;

// This will use kyutai's STT implementation
let transcript = FluentVoice::stt()
    .conversation()
    .with_source(SpeechSource::Microphone { ... })
    .listen(|conv| conv.into_stream())
    .await?;
```

### Files to Modify
- `src/engine.rs` (remove stub implementations, clean up FluentVoice impl)

### Files to Reference (DO NOT MODIFY)
- `/Volumes/samsung_t9/fluent-voice/packages/fluent-voice/src/fluent_voice.rs` (working default implementations)
- `/Volumes/samsung_t9/fluent-voice/packages/koffee/` (working wake word implementation)
- `/Volumes/samsung_t9/fluent-voice/packages/vad/` (working VAD implementation)

### Testing
- Verify wake word detection works using default Koffee implementation
- Confirm VAD functionality works through default STT engine
- Test that kyutai TTS/STT still work properly
- Ensure no broken stub code remains

### Acceptance Criteria
- ✅ All stub wake word implementations removed from kyutai
- ✅ Wake word detection delegates to working Koffee implementation
- ✅ VAD functionality uses default STT engine implementation
- ✅ Clean separation: kyutai focuses on TTS/STT, DefaultEngine handles wake word/VAD
- ✅ Users get working functionality instead of broken stubs