# Fluent-Voice Architecture Fix TODO

## CRITICAL ARCHITECTURE UNDERSTANDING

**The fluent-voice architecture is FULLY UNWRAPPED:**

1. **User writes matcher closures with both arms:**
   ```rust
   .on_chunk(|chunk| {
       Ok => chunk.into(),     // User defines success handling
       Err(e) => handle_error(e), // User defines error handling
   })
   ```

2. **Behind the scenes, the system:**
   - Captures these closures
   - Calls the appropriate arm based on actual results
   - `Ok` arm gets called on success → data flows to stream
   - `Err` arm gets called on error → error handling

3. **Final streams contain ONLY unwrapped data:**
   - `AsyncStream<ConcreteTranscriptSegment>` (never `Result<AsyncStream<...>, Error>`)
   - `AsyncStream<AudioChunk>` (never `Result<AsyncStream<...>, Error>`)
   - All data in streams is unwrapped concrete types

4. **NotResult constraint enforces this:**
   - `AsyncStream<T>` requires `T: NotResult`
   - Prevents any Result types from entering streams
   - Forces architecture to be fully unwrapped

## PRIORITY FIXES

### 1. Fix NotResult Constraint Violations (Critical - Blocks Examples)

**Problem:** Using `Box<dyn TranscriptSegment + Send>` which doesn't implement `NotResult`
**Solution:** Use concrete types that implement `NotResult`

- [ ] **Fix default_stt_engine.rs transcript types**
  - Change `Box<dyn crate::transcript::TranscriptSegment + Send>` to `fluent_voice_domain::ConcreteTranscriptSegment`
  - Update all method signatures in default_stt_engine.rs lines 572 and 1000
  - Ensure return types are `AsyncStream<ConcreteTranscriptSegment>` (unwrapped)
  - DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

- [ ] **QA Check:** Act as an Objective QA Rust developer - Verify that all AsyncStream types use concrete types that implement NotResult, not boxed trait objects. Confirm no Result types exist in any stream.

### 2. Fix Method Signatures to Match Domain Traits

**Problem:** Implementation methods have wrong number of type parameters
**Solution:** Add missing type parameters to match domain trait signatures

- [ ] **Fix STT listen method signatures**
  - In stt_builder.rs line 264: Change `fn listen<F>` to `fn listen<F, R>`
  - In stt_builder.rs line 414: Change `fn listen<F>` to `fn listen<F, S>`
  - In default_stt_engine.rs line 572: Change `fn listen<F>` to `fn listen<F, R>`
  - In default_stt_engine.rs line 1000: Change `fn listen<F>` to `fn listen<F, S>`
  - Update where clauses to match domain trait requirements exactly
  - DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

- [ ] **QA Check:** Act as an Objective QA Rust developer - Verify that all listen method signatures match the domain trait signatures exactly with correct type parameters.

### 3. Fix Missing Trait Implementations

**Problem:** Missing required trait methods
**Solution:** Implement missing methods or rename existing ones

- [ ] **Fix TtsConversationChunkBuilder implementation**
  - In tts_builder.rs line 562: Add missing `synthesize` method to TtsConversationChunkBuilder impl
  - Either rename `synthesize_stream` to `synthesize` or add new `synthesize` method
  - Ensure method returns unwrapped types (no Result wrapping)
  - DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

- [ ] **QA Check:** Act as an Objective QA Rust developer - Verify that all trait implementations are complete and methods return unwrapped types.

### 4. Fix FluentVoice Return Types

**Problem:** FluentVoice methods return types that don't implement required traits
**Solution:** Make return types implement the required traits or return different types

- [ ] **Fix FluentVoice::tts() and FluentVoice::stt() return types**
  - In fluent_voice.rs line 167: Make TtsEntry implement TtsConversationBuilder trait
  - In fluent_voice.rs line 171: Make SttEntry implement SttConversationBuilder trait
  - Or change return types to types that already implement these traits
  - Ensure all returned builders work with unwrapped data flows
  - DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

- [ ] **QA Check:** Act as an Objective QA Rust developer - Verify that FluentVoice trait implementation returns types that implement the required builder traits.

### 5. Fix ChunkBuilder Type Issues

**Problem:** ChunkBuilder type doesn't implement required trait
**Solution:** Use correct type for ChunkBuilder

- [ ] **Fix DefaultTtsBuilder ChunkBuilder type**
  - In fluent_voice.rs line 276: Change `type ChunkBuilder = Self;` to use type that implements TtsConversationChunkBuilder
  - Or implement TtsConversationChunkBuilder for DefaultTtsBuilder
  - Ensure ChunkBuilder works with unwrapped data flows
  - DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

- [ ] **QA Check:** Act as an Objective QA Rust developer - Verify that ChunkBuilder type implements TtsConversationChunkBuilder trait correctly.

### 6. Fix Import Errors

**Problem:** Trying to import builders that don't exist
**Solution:** Import the correct builder implementations

- [ ] **Fix builder imports in lib.rs**
  - In lib.rs lines 151-152: Change imports to use existing `*BuilderImpl` versions
  - AudioIsolationBuilder → AudioIsolationBuilderImpl
  - SoundEffectsBuilder → SoundEffectsBuilderImpl
  - SpeechToSpeechBuilder → SpeechToSpeechBuilderImpl
  - VoiceCloneBuilder → VoiceCloneBuilderImpl
  - VoiceDiscoveryBuilder → VoiceDiscoveryBuilderImpl
  - DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

- [ ] **QA Check:** Act as an Objective QA Rust developer - Verify that all builder imports reference existing types and compile without errors.

## VERIFICATION CRITERIA

### Success Criteria:
1. ✅ `cargo run --package fluent_voice --example tts` compiles and runs
2. ✅ `cargo run --package fluent_voice --example stt` compiles and runs  
3. ✅ All AsyncStream types use concrete types that implement NotResult
4. ✅ No Result types exist in any stream data flows
5. ✅ JSON syntax `{"key" => "value"}` works in examples (already working)
6. ✅ User-defined matcher closures handle both Ok and Err arms
7. ✅ All trait implementations are complete and match domain definitions

### Architecture Validation:
- User writes: `.engine_config({"provider" => "dia"})`
- Gets transformed: `.engine_config(hash_map_fn!{"provider" => "dia"})`
- Builder receives closure and calls it to get HashMap
- User defines error handling in matcher closures
- System calls appropriate arm based on actual results
- Streams contain only unwrapped concrete data

**CRITICAL:** Everything flows as unwrapped concrete types. The user's matcher closures handle both success and error cases, but the streams only contain the unwrapped success data.