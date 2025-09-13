# Integrate with Domain Timestamp Types

## Description
Integrate ElevenLabs timestamp implementation with existing fluent_voice_domain timestamp types.

## Current Problem
Task specification mentioned integration with existing domain types like `TimestampsGranularity`, `WordTimestamps`, `Diarization` from `packages/domain/src/timestamps.rs`, but implementation uses completely separate types.

## Required Solution
1. **Review existing domain timestamp types** and their intended usage
2. **Adapt ElevenLabs implementation** to use or extend domain types where appropriate
3. **Create conversion functions** between ElevenLabs and domain types
4. **Maintain API compatibility** with existing fluent-voice patterns

## Implementation Steps
1. **Analyze domain timestamp types** in `packages/domain/src/timestamps.rs`
2. **Identify integration opportunities** between domain and ElevenLabs types
3. **Create conversion traits** (From/Into) between type systems
4. **Update ElevenLabs types** to extend or use domain types where appropriate
5. **Ensure compatibility** with fluent-voice builder patterns
6. **Add tests** for type conversions

## Technical Investigation Needed
### Domain Types Analysis
```rust
// Need to examine:
// - packages/domain/src/timestamps.rs
// - TimestampsGranularity enum values and usage
// - WordTimestamps structure and fields  
// - Diarization capabilities and speaker handling
```

### Integration Strategy
```rust
// Example conversion implementation:
impl From<TimestampMetadata> for fluent_voice_domain::WordTimestamps {
    fn from(metadata: TimestampMetadata) -> Self {
        // Convert ElevenLabs word alignments to domain WordTimestamps
    }
}

impl From<&CharacterTimestamp> for fluent_voice_domain::TimestampsGranularity {
    fn from(char_ts: &CharacterTimestamp) -> Self {
        // Map character-level data to domain granularity enum
    }
}
```

## Files to Investigate
- `packages/domain/src/timestamps.rs` - Existing domain types
- `packages/fluent-voice/src/*.rs` - Fluent-voice API patterns
- `packages/elevenlabs/src/timestamp_metadata.rs` - Current implementation

## Files to Update
- `packages/elevenlabs/src/timestamp_metadata.rs` - Add domain type conversions
- `packages/elevenlabs/src/lib.rs` - Export domain-compatible types
- `packages/elevenlabs/src/engine.rs` - Use domain types in API where appropriate

## Success Criteria
- [ ] ElevenLabs types integrate seamlessly with domain types
- [ ] Conversion functions preserve all timestamp data
- [ ] API maintains compatibility with fluent-voice patterns
- [ ] No duplication of timestamp functionality
- [ ] Domain types are used where they provide value
- [ ] Documentation explains integration patterns

## Dependencies
- Requires understanding of existing domain type architecture
- Should be done after core stubbing issues are resolved

## Architecture Impact
**HIGH** - Affects type system design and API consistency across fluent-voice ecosystem