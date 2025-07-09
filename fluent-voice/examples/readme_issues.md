# README.md Syntax Issues Found

## 1. Invalid Matcher Syntax

The README shows this pattern in multiple places:
```rust
.synthesize(|conversation| {
    Ok  => conversation.into_stream(),
    Err(e) => Err(e),
})
```

This is **invalid Rust syntax**. The correct syntax should be:
```rust
.synthesize(|conversation| match conversation {
    Ok(conv) => conv.into_stream(),
    Err(e) => Err(e),
})
```

## 2. Missing `collect()` Method on SttConversation

The first STT example (line 53-68) shows:
```rust
let _transcript = FluentVoice::stt()
    // ... configuration ...
    .listen(|segment| {
        Ok  => segment.text(),
        Err(e) => Err(e),
    })
    .collect();  // ❌ This won't work!
```

Problems:
1. `listen()` returns a Future that resolves to whatever the matcher returns
2. There's no `collect()` method on the result
3. The matcher syntax is also invalid

The correct approach would be:
```rust
// Option 1: Use transcribe() which has collect()
let transcript = FluentVoice::stt()
    .transcribe("audio.wav")
    .collect()
    .await?;

// Option 2: Use listen() and process the stream
let mut stream = FluentVoice::stt()
    .with_source(...)
    .listen(|conversation| match conversation {
        Ok(conv) => Ok(conv.into_stream()),
        Err(e) => Err(e),
    })
    .await?;
```

## 3. Wrong Type Names in Examples

README shows:
- `Speaker::speaker()` - but implementation has `SpeakerLine::speaker()`
- `MyTtsEngine::conversation()` - but traits show this should be an instance method
- `MySttEngine::conversation()` - same issue

## 4. VoiceId Constructor Inconsistency

README shows:
```rust
.voice_id(VoiceId::new("voice-uuid"))
```

But the struct uses a tuple constructor:
```rust
.voice_id(VoiceId("voice-uuid".to_string()))
```

## 5. Missing Configuration Method

The first STT example doesn't include `timestamps_granularity()` which is part of the trait.

## Summary of Required Fixes

### README needs to be updated to show:

1. **Correct matcher syntax** using `match` expressions
2. **Correct usage patterns** for STT:
   - Use `transcribe()` for file processing with `collect()`
   - Use `listen()` for streaming with proper stream handling
3. **Correct type names** that match the implementation
4. **Consistent constructors** for VoiceId and other types
5. **Complete configuration examples** including all available methods