# Add Alignment Data Validation

## Description
Add comprehensive validation for ElevenLabs alignment data to ensure production robustness.

## Current Problem
No validation that:
- ElevenLabs alignment arrays have matching lengths
- Character and timing data is well-formed  
- Timing values are logical (start < end, etc.)

## Required Solution
Implement comprehensive validation for all alignment data before processing.

## Implementation Steps
1. **Validate array length consistency** across alignment fields
2. **Validate timing value logic** (start < end, non-negative, etc.)
3. **Validate character data integrity** (non-empty characters, valid UTF-8)
4. **Add validation to From<&Alignment> trait**
5. **Create validation error variants** in FluentVoiceError
6. **Add validation before word aggregation**

## Technical Implementation
### Array Length Validation
```rust
fn validate_alignment(alignment: &Alignment) -> Result<(), FluentVoiceError> {
    let char_count = alignment.characters.len();
    let start_count = alignment.character_start_times_seconds.len();
    let end_count = alignment.character_end_times_seconds.len();
    
    if char_count != start_count || char_count != end_count {
        return Err(FluentVoiceError::ConfigError(format!(
            "Alignment array mismatch: {} chars, {} start times, {} end times",
            char_count, start_count, end_count
        )));
    }
    
    Ok(())
}
```

### Timing Logic Validation
```rust
fn validate_timing_logic(alignment: &Alignment) -> Result<(), FluentVoiceError> {
    for (i, (&start, &end)) in alignment.character_start_times_seconds
        .iter()
        .zip(&alignment.character_end_times_seconds)
        .enumerate() 
    {
        if start < 0.0 || end < 0.0 {
            return Err(FluentVoiceError::ConfigError(format!(
                "Negative timing at character {}: start={}, end={}", i, start, end
            )));
        }
        
        if start >= end {
            return Err(FluentVoiceError::ConfigError(format!(
                "Invalid timing at character {}: start={} >= end={}", i, start, end
            )));
        }
    }
    
    Ok(())
}
```

### Character Data Validation
```rust
fn validate_character_data(alignment: &Alignment) -> Result<(), FluentVoiceError> {
    for (i, character) in alignment.characters.iter().enumerate() {
        if character.is_empty() {
            return Err(FluentVoiceError::ConfigError(format!(
                "Empty character at position {}", i
            )));
        }
        
        if !character.is_ascii() && !character.chars().all(|c| c.is_alphabetic() || c.is_whitespace()) {
            // Log warning for non-standard characters but don't fail
        }
    }
    
    Ok(())
}
```

## Files to Update
- `packages/elevenlabs/src/timestamp_metadata.rs` - Add validation to From trait
- `packages/elevenlabs/src/timestamp_metadata.rs` - Add validation before word generation
- `packages/elevenlabs/src/engine.rs` - Add validation error variants

## Success Criteria
- [ ] All alignment data is validated before processing
- [ ] Array length mismatches are caught and reported
- [ ] Invalid timing values are rejected with clear errors
- [ ] Character data integrity is verified
- [ ] Validation errors provide actionable debugging information
- [ ] Performance impact is minimal for valid data

## Dependencies
- Should be implemented after Task 3 (error handling fixes)

## Architecture Impact
**MEDIUM** - Adds robustness without changing core functionality, may affect performance