# TASK2: Fix TTS Synthesis Stream Stub

## Issue Classification
**CRITICAL PRODUCTION QUALITY VIOLATION**

## Problem Description
The TTS conversation's `into_stream()` method returns empty synthesis data with explicit placeholder comments, violating the "Never Stub" principle.

## Location
**File:** `packages/fluent-voice/src/fluent_voice.rs`  
**Line:** 748  

## Current Implementation (STUB)
```rust
// Return properly formatted AudioChunk with real synthesis results
fluent_voice_domain::AudioChunk::with_metadata(
    Vec::new(),                    // Real implementation would contain synthesized audio
    0,                             // duration_ms
    0,                             // start_ms
    Some("dia_voice".to_string()), // speaker_id
    Some("Synthesized via dia-voice".to_string()), // text
    Some(fluent_voice_domain::AudioFormat::Pcm16Khz), // format
)
```

## Root Cause
The TTS conversation stream returns empty audio data with a comment explicitly stating "Real implementation would contain synthesized audio", which is a clear stub.

## Required Fix
Replace the stub with actual DiaVoiceBuilder integration for real TTS synthesis:

1. **Integrate with DiaVoiceBuilder** for actual synthesis
2. **Use real synthesis API calls** instead of empty Vec::new()
3. **Stream actual audio chunks** from dia-voice synthesis
4. **Calculate proper timing** based on synthesis progress
5. **Handle synthesis errors** appropriately

## Acceptance Criteria
- ✅ No empty Vec::new() for synthesis data
- ✅ No placeholder comments mentioning "would contain synthesized audio"
- ✅ Actual integration with DiaVoiceBuilder synthesis
- ✅ Real audio stream from synthesis engine
- ✅ Proper error handling for synthesis failures

## Implementation Strategy
Use the DiaVoiceBuilder properly:
1. Unwrap the `dia_builder` and configure it for synthesis
2. Call actual synthesis methods (likely async)
3. Stream real AudioChunks as synthesis progresses
4. Handle synthesis completion and errors properly
5. Return a proper Stream of real audio data

## Context
The code already has access to `DiaVoiceBuilder` but doesn't use it for actual synthesis. The builder should be utilized for real TTS functionality.

## Priority
**CRITICAL** - This prevents the TTS conversation system from producing actual synthesized speech output.