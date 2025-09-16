# TASK1: Fix TTS Audio Generation Stub

## Issue Classification
**CRITICAL PRODUCTION QUALITY VIOLATION**

## Problem Description
The TTS conversation builder creates AudioChunks with empty audio data and placeholder comments, violating the "Never Stub" principle.

## Location
**File:** `packages/fluent-voice/src/fluent_voice.rs`  
**Line:** 649  

## Current Implementation (STUB)
```rust
let chunk = AudioChunk::with_metadata(
    Vec::new(),              // Empty data for now - real engine would populate this
    0,                       // duration_ms
    0,                       // start_ms
    self.speaker_id.clone(), // speaker_id
    Some("Synthesis placeholder".to_string()), // text
    Some(AudioFormat::Pcm16Khz), // format
);
```

## Root Cause
The TTS builder returns an empty AudioChunk with a comment explicitly stating "real engine would populate this", which is a clear stub implementation.

## Required Fix
Replace the stub with actual TTS synthesis using the available TTS engines:

1. **Integrate with ElevenLabs engine** for real audio synthesis
2. **Use proper audio data** instead of `Vec::new()`
3. **Calculate actual duration** based on synthesized audio length
4. **Remove placeholder text** and use actual synthesis input
5. **Ensure proper timing metadata** for audio chunks

## Acceptance Criteria
- ✅ No empty Vec::new() for audio data
- ✅ No placeholder comments mentioning "real engine would populate"
- ✅ Actual TTS synthesis produces real audio bytes
- ✅ Proper duration calculation based on audio length
- ✅ Real text input used for synthesis (not "placeholder")

## Implementation Strategy
Use the fluent-voice TTS engine integration pattern:
1. Configure TTS engine (ElevenLabs, OpenAI, etc.) 
2. Perform actual text-to-speech synthesis
3. Return AudioChunk with real synthesized audio data
4. Calculate proper timing metadata

## Priority
**CRITICAL** - This violates the core "Never Stub" requirement and prevents actual TTS functionality.