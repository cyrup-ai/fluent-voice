# Implement Audio Chunk Timing Calculations

## Description
Replace stubbed audio chunk timing with actual calculations based on ElevenLabs alignment data.

## Current Problem
In `packages/elevenlabs/src/engine.rs:766-775`, all timing calculations are stubbed:

```rust
let chunk = AudioChunkTimestamp {
    chunk_id: chunk_idx,
    start_ms: 0, // Would need to calculate from previous chunks
    end_ms: 0,   // Would need to calculate from alignment data
    text_segment: "unknown".to_string(), // Context not available
    speaker_id: None, // Multi-speaker support
    format: "unknown".to_string(),
    size_bytes: 0, // Would need to calculate from audio data
};
```

## Required Solution
1. **Calculate start_ms/end_ms** from ElevenLabs character timing data
2. **Extract text_segment** corresponding to each audio chunk
3. **Calculate size_bytes** from actual audio data
4. **Determine format** from synthesis configuration

## Implementation Steps
1. Convert ElevenLabs character timings (seconds) to milliseconds
2. Aggregate character timings to determine chunk start/end times
3. Map alignment data to corresponding text segments
4. Calculate audio data size for each chunk
5. Extract format from TTS output configuration
6. Handle multi-speaker scenarios with speaker_id extraction

## Technical Details
### Timing Calculation Algorithm
```rust
// Convert character alignment to chunk timing
let chunk_start_seconds = alignment.character_start_times_seconds.first().unwrap_or(&0.0);
let chunk_end_seconds = alignment.character_end_times_seconds.last().unwrap_or(&0.0);
let start_ms = (*chunk_start_seconds * 1000.0) as u64;
let end_ms = (*chunk_end_seconds * 1000.0) as u64;
```

### Text Segment Extraction
```rust
let text_segment = alignment.characters.join("");
```

### Audio Size Calculation
```rust
let size_bytes = audio_bytes.len();
```

## Success Criteria
- [ ] start_ms and end_ms contain actual timing values (not 0)
- [ ] text_segment contains actual text from alignment data
- [ ] size_bytes reflects actual audio chunk size
- [ ] format matches synthesis output format
- [ ] Timing values are logical (start_ms < end_ms)
- [ ] Cumulative timing matches total audio duration

## Dependencies
- Requires Task 1 (synthesis metadata) for format information
- Needs access to raw audio chunk data for size calculation

## Architecture Impact
**HIGH** - Affects core timestamp accuracy and audio synchronization functionality