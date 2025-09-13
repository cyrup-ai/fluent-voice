# Fix Error Type Inconsistencies

## Description
Standardize error type paths throughout the ElevenLabs timestamp implementation.

## Current Problem
In `packages/elevenlabs/src/timestamp_export.rs`:
- Line 10: `crate::engine::FluentVoiceError`
- Line 33: `crate::FluentVoiceError`

**Issue**: Inconsistent error type paths could cause compilation issues.

## Required Solution
Standardize on a single, correct error type path throughout all timestamp modules.

## Implementation Steps
1. **Determine correct error type path** by checking FluentVoiceError definition
2. **Update all timestamp modules** to use consistent path
3. **Verify compilation** after changes
4. **Add use statement** if needed to simplify repeated usage

## Technical Details
### Current Inconsistency
```rust
// timestamp_export.rs line 10
pub fn to_srt(&self) -> Result<String, crate::engine::FluentVoiceError> {

// timestamp_export.rs line 33  
Err(crate::FluentVoiceError::ConfigError(
```

### Standardized Approach
Option 1 - Use import:
```rust
use crate::engine::FluentVoiceError;

pub fn to_srt(&self) -> Result<String, FluentVoiceError> {
    // ...
    Err(FluentVoiceError::ConfigError(
```

Option 2 - Consistent full path:
```rust
pub fn to_srt(&self) -> Result<String, crate::engine::FluentVoiceError> {
    // ...
    Err(crate::engine::FluentVoiceError::ConfigError(
```

## Files to Update
- `packages/elevenlabs/src/timestamp_export.rs`
- `packages/elevenlabs/src/timestamp_metadata.rs`
- Any other timestamp files using FluentVoiceError

## Success Criteria
- [ ] All FluentVoiceError references use consistent path
- [ ] Code compiles without error type issues
- [ ] Import statements are clean and consistent
- [ ] Error handling follows same pattern across all files

## Dependencies
- None - can be implemented independently

## Architecture Impact
**MINIMAL** - Fixes compilation issues without changing functionality