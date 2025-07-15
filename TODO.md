# TODO: Add cyrup_sugars JSON Object Syntax

## 🎯 USER OBJECTIVE
Add cyrup_sugars hashbrown-json feature to enable JSON object syntax `{"key" => "value"}` in fluent-voice builders. Everything else already exists and works.

## 📋 MINIMAL TASKS

### 1. Enable hashbrown-json feature in cyrup_sugars dependency
**File:** `fluent-voice/packages/fluent-voice/Cargo.toml`
**Line:** 31 (cyrup_sugars dependency)
**Changes:** 
- Change `cyrup_sugars = { git = "https://github.com/cyrup-ai/cyrup-sugars", default-features = true }`
- To `cyrup_sugars = { git = "https://github.com/cyrup-ai/cyrup-sugars", features = ["hashbrown-json"] }`

DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### 2. QA: Verify hashbrown-json feature enablement
Act as an Objective QA Rust developer. Rate the work performed previously on enabling hashbrown-json feature. Verify that:
- Feature is correctly specified in Cargo.toml
- Dependency still points to correct repository
- No other dependencies were modified unnecessarily
- Feature enables JSON object syntax capabilities

### 3. Add JSON config method to TtsConversationBuilder
**File:** `fluent-voice/packages/fluent-voice/src/builders/tts_builder.rs`
**Lines:** Add new method to TtsConversationBuilderImpl impl block
**Changes:**
- Add `engine_config<F>(self, f: F) -> Self where F: FnOnce() -> hashbrown::HashMap<&'static str, &'static str>`
- Store config in internal HashMap field
- Route to engine during synthesis

DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### 4. QA: Verify TTS JSON configuration method
Act as an Objective QA Rust developer. Rate the work performed previously on TTS JSON configuration. Verify that:
- Method signature accepts hashbrown::HashMap closure
- Implementation stores configuration properly
- Method maintains fluent builder pattern
- No unwrap() or expect() calls in implementation

### 5. Add JSON config method to SttConversationBuilder
**File:** `fluent-voice/packages/fluent-voice/src/builders/stt_builder.rs`
**Lines:** Add new method to SttConversationBuilderImpl impl block
**Changes:**
- Add `engine_config<F>(self, f: F) -> Self where F: FnOnce() -> hashbrown::HashMap<&'static str, &'static str>`
- Store config in internal HashMap field
- Route to engine during listening

DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### 6. QA: Verify STT JSON configuration method
Act as an Objective QA Rust developer. Rate the work performed previously on STT JSON configuration. Verify that:
- JSON config method is properly implemented
- Configuration routing works correctly
- Method integrates with existing builder pattern
- No blocking operations introduced

### 7. Add JSON config methods to SpeakerBuilder
**File:** `fluent-voice/packages/fluent-voice/src/builders/tts_builder.rs`
**Lines:** Add methods to SpeakerLineBuilder impl block
**Changes:**
- Add `metadata<F>(self, f: F) -> Self where F: FnOnce() -> hashbrown::HashMap<&'static str, &'static str>`
- Add `vocal_settings<F>(self, f: F) -> Self where F: FnOnce() -> hashbrown::HashMap<&'static str, &'static str>`
- Store in HashMap fields

DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### 8. QA: Verify SpeakerBuilder JSON methods
Act as an Objective QA Rust developer. Rate the work performed previously on SpeakerBuilder JSON methods. Verify that:
- Methods follow cyrup_sugars patterns correctly
- Configuration is stored properly
- Integration with speaker system works
- Type safety is maintained

### 9. Update TTS example with JSON syntax
**File:** `fluent-voice/packages/fluent-voice/examples/tts.rs`
**Lines:** Enhance audio_stream creation (lines 25-63)
**Changes:**
- Add `.engine_config(|| hashbrown::hashmap!{"provider" => "dia", "quality" => "high"})`
- Add `.metadata()` and `.vocal_settings()` to speaker configurations
- Demonstrate JSON object syntax in action

DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### 10. QA: Verify TTS example enhancement
Act as an Objective QA Rust developer. Rate the work performed previously on TTS example enhancement. Verify that:
- JSON syntax demonstrates cyrup_sugars integration
- Example runs successfully with enhanced syntax
- Configuration demonstrates real usage patterns
- Code is clean and demonstrates API properly

### 11. Update STT example with JSON syntax
**File:** `fluent-voice/packages/fluent-voice/examples/stt.rs`
**Lines:** Enhance transcript_stream creation (lines 25-48)
**Changes:**
- Add `.engine_config(|| hashbrown::hashmap!{"provider" => "whisper", "model" => "large-v3"})`
- Demonstrate JSON object syntax for STT configuration
- Show integration with existing wake word and VAD

DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### 12. QA: Verify STT example enhancement
Act as an Objective QA Rust developer. Rate the work performed previously on STT example enhancement. Verify that:
- JSON configuration demonstrates API capabilities
- Example works with existing engine integrations
- Syntax is clean and intuitive
- Integration with wake word/VAD is maintained

### 13. Update README.md examples with JSON syntax
**File:** `fluent-voice/packages/fluent-voice/README.md`
**Lines:** Replace examples (lines 61-143) with JSON syntax
**Changes:**
- Show `.engine_config()` in TTS and STT examples
- Demonstrate `.metadata()` and `.vocal_settings()` for speakers
- Update advanced usage examples with JSON syntax
- Maintain all existing functionality while showing enhanced API

DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### 14. QA: Verify README.md syntax update
Act as an Objective QA Rust developer. Rate the work performed previously on README.md update. Verify that:
- All examples demonstrate JSON object syntax
- Documentation is accurate and complete
- Examples are runnable and correct
- API enhancements are properly showcased

## 🎯 SUCCESS CRITERIA

When all tasks are complete:

1. **JSON Object Syntax Works**: `{"key" => "value"}` syntax functions in all builders
2. **Examples Enhanced**: Both tts.rs and stt.rs demonstrate JSON configuration
3. **Documentation Updated**: README.md shows enhanced API usage
4. **Existing Functionality Preserved**: All current features continue working
5. **Zero Allocation Maintained**: Performance characteristics unchanged

## 🚫 ABSOLUTE CONSTRAINTS

- Never use unwrap() in src/* files or examples/*
- Never use expect() in src/* files or examples/*
- DO USE expect() in tests/* files for assertions
- No unsafe code anywhere
- Complete implementation - no stubs or "TODO" comments
- Maintain backward compatibility with existing API
- Do not modify working engine implementations
- Focus only on adding JSON syntax capabilities