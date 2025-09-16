# TASK3: Fix Wake Word Model Loading Stub

## Issue Classification
**CRITICAL PRODUCTION QUALITY VIOLATION**

## Problem Description
Wake word model loading is commented out and marked as placeholder, preventing wake word detection functionality.

## Location
**File:** `packages/fluent-voice/src/engines/default_stt_engine.rs`  
**Line:** ~99 (in AudioProcessor::new() method)

## Current Implementation (STUB)
```rust
let wake_word_detector = KoffeeCandle::new(&koffee_config).map_err(|e| {
    VoiceError::Configuration(format!("Failed to create wake word detector: {}", e))
})?;

// Load the "hey_fluent" wake word model (placeholder - would load real model bytes)
// wake_word_detector.add_wakeword_bytes(&model_bytes)?;
```

## Root Cause
The wake word model loading is commented out with an explicit placeholder comment, meaning wake word detection cannot function properly.

## Required Fix
Implement actual wake word model loading:

1. **Load real wake word model bytes** for "hey_fluent" or similar
2. **Uncomment and implement** the `add_wakeword_bytes` call
3. **Handle model loading errors** properly
4. **Ensure model compatibility** with KoffeeCandle
5. **Configure proper wake word detection** settings

## Acceptance Criteria
- ✅ No commented-out model loading code
- ✅ No placeholder comments mentioning "would load real model bytes"
- ✅ Actual wake word model loaded and functional
- ✅ Wake word detection works with real audio input
- ✅ Proper error handling for model loading failures

## Implementation Strategy
1. **Obtain wake word model data**: Either embed model bytes in binary or load from file
2. **Configure model loading**: Implement proper model bytes loading
3. **Add error handling**: Handle model loading and validation errors
4. **Test wake word detection**: Ensure detection works with loaded model
5. **Document model requirements**: Specify model format and source

## Technical Notes
- KoffeeCandle appears to support `add_wakeword_bytes(&model_bytes)` method
- Model format and source need to be determined
- Model bytes can be embedded in binary or loaded from filesystem
- Error handling should use VoiceError::Configuration for consistency

## Priority
**CRITICAL** - Wake word detection is a core STT feature and cannot function without proper model loading.