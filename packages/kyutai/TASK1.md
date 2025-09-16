# TASK1: Fix on_prediction Handler Bug

## Status: CRITICAL BUG - 2 Line Fix Required

### Problem
The `on_prediction` handler in `engine.rs:746-752` ignores the callback parameter, breaking STT event handling.

### THE OTHER END OF THE WIRE: Required by SttConversationBuilder Trait

**Trait Method**: From `/Volumes/samsung_t9/fluent-voice/packages/fluent-voice/src/stt_conversation.rs:112-119`

```rust
/// Set a callback to be invoked when a prediction is available.
///
/// The callback receives the final transcript segment and the
/// predicted text that follows it.
fn on_prediction<F>(self, f: F) -> Self
where
    F: FnMut(String, String) + Send + 'static;
```

**Purpose**: This trait method is **required** by the `SttConversationBuilder` trait. The callback should be invoked during STT processing to provide real-time prediction updates to the user.

### Current Broken Code
```rust
fn on_prediction<F>(self, _f: F) -> Self
where F: FnMut(String, String) + Send + 'static,
{
    self  // BUG: Ignores the callback completely
}
```

### Required Fix
```rust
fn on_prediction<F>(mut self, f: F) -> Self
where F: FnMut(String, String) + Send + 'static,
{
    self.prediction_callback = Some(Box::new(f));
    self
}
```

### Implementation Steps
1. Add `mut` to the `self` parameter
2. Store the callback in `self.prediction_callback` 
3. Ensure `KyutaiSttConversationBuilder` has `prediction_callback` field
4. Use the stored callback when predictions are generated

### Files to Modify
- `src/engine.rs` (lines 746-752)

### Testing
- Verify callback is called during STT processing
- Test with sample audio input
- Confirm prediction strings are passed correctly

### Acceptance Criteria
- ✅ Callback parameter is stored, not ignored
- ✅ Predictions trigger the callback during STT
- ✅ No compiler warnings about unused parameters