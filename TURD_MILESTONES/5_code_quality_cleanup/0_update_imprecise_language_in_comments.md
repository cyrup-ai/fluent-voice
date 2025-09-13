# Update Imprecise Language in Comments

## Description
Update comments that use imprecise terminology like "dummy", "stubs", etc. to use more accurate technical language.

## Current Violations
- `packages/dia/src/state.rs:271` - "dummy 'all real' mask" 
- `packages/dia/src/ui.rs:351` - "Provide stubs when UI is not enabled"
- `packages/kyutai/tests/audio_logits.rs:14-16` - Uses "dummy" in test code

## Technical Resolution
Update comments to use precise technical language:

```rust
// ❌ OLD: "dummy 'all real' mask"
// ✅ NEW: "initial 'all real' mask" or "default 'all real' mask"

// ❌ OLD: "Provide stubs when UI is not enabled"  
// ✅ NEW: "Provide no-op implementations when UI is not enabled"

// ❌ OLD: "Create a dummy VarBuilder for testing"
// ✅ NEW: "Create a test VarBuilder for testing"
```

## Success Criteria
- [ ] Replace "dummy" with "initial", "default", or "test" as appropriate
- [ ] Replace "stubs" with "no-op implementations"
- [ ] Ensure all comments accurately describe functionality
- [ ] Maintain clarity while using precise terminology
- [ ] Review for any other imprecise language

## Dependencies
None - independent language cleanup task

## Architecture Impact
LOW - Improves code documentation quality and precision