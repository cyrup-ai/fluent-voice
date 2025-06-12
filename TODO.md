# Fluent Voice Project TODO

## 🎯 Current Objective
**Integrate Kokoros (TTS) and Whisper (STT) engines from the workspace with the fluent-voice trait system.**

Target: Create adapter crates that bridge `kokoros/kokoros` (TTS) and `candle/whisper` (STT) implementations to work seamlessly with the `TtsEngine`/`SttEngine` traits. Additionally, integrate `candle/koffee` for wake-word detection capabilities.

## 🚨 CRITICAL: Macro Implementation Audit & Fixes

**BLOCKING ALL ENGINE IMPLEMENTATIONS** - The `stt_engine!` and `tts_engine!` macros are incomplete and missing critical trait methods.

### Macro Audit Results:

#### `stt_engine!` macro missing 7 methods + 1 type from `SttConversationBuilder`:
- ❌ `with_microphone(device: impl Into<String>) -> Self` - polymorphic branching  
- ❌ `transcribe(path: impl Into<String>) -> Self` - polymorphic branching
- ❌ `with_progress<S: Into<String>>(template: S) -> Self` - progress templates
- ❌ `emit<F, R>(matcher: F) -> impl Future<Output = R> + Send` - terminal method with matcher
- ❌ `collect() -> impl Future<Output = Result<Self::Transcript, VoiceError>> + Send` - convenience method
- ❌ `collect_with<F, R>(handler: F) -> impl Future<Output = R> + Send` - convenience method  
- ❌ `as_text() -> impl Stream<Item = String> + Send` - convenience method
- ❌ `type Transcript: Send` - missing associated type

#### `tts_engine!` macro missing `FluentVoice` trait implementation:
- ❌ `impl FluentVoice for $engine` with `tts()` and `stt()` methods

#### Both macros missing:
- ❌ `FluentVoice` trait implementation for unified entry points

### Required Fixes - 16 Critical TODO Items:

#### STT Macro Missing Methods (8 items):
1. **Add `with_microphone()` to `stt_engine!` macro** - Implement `fn with_microphone(self, device: impl Into<String>) -> Self` that sets SpeechSource::Microphone with device parameter
2. **Add `transcribe()` to `stt_engine!` macro** - Implement `fn transcribe(self, path: impl Into<String>) -> Self` that sets SpeechSource::File with path parameter  
3. **Add `with_progress()` to `stt_engine!` macro** - Implement `fn with_progress<S: Into<String>>(self, template: S) -> Self` for progress template storage
4. **Add `emit()` to `stt_engine!` macro** - Implement `fn emit<F, R>(self, matcher: F) -> impl Future<Output = R> + Send` terminal method with matcher closure pattern
5. **Add `collect()` to `stt_engine!` macro** - Implement `fn collect(self) -> impl Future<Output = Result<Self::Transcript, VoiceError>> + Send` convenience method
6. **Add `collect_with()` to `stt_engine!` macro** - Implement `fn collect_with<F, R>(self, handler: F) -> impl Future<Output = R> + Send` convenience method with handler
7. **Add `as_text()` to `stt_engine!` macro** - Implement `fn as_text(self) -> impl Stream<Item = String> + Send` text-only stream convenience method
8. **Add `type Transcript` to `stt_engine!` macro** - Add missing associated type `type Transcript: Send` to complete trait implementation

#### FluentVoice Trait Implementation (4 items):
9. **Add `FluentVoice` impl to `stt_engine!` macro** - Generate `impl FluentVoice for $engine` with `fn stt() -> impl SttConversationBuilder` method
10. **Add `FluentVoice` impl to `tts_engine!` macro** - Generate `impl FluentVoice for $engine` with `fn tts() -> impl TtsConversationBuilder` method  
11. **Add cross-trait support to `stt_engine!` macro** - Generate `fn tts() -> impl TtsConversationBuilder` method that panics or delegates to separate TTS engine
12. **Add cross-trait support to `tts_engine!` macro** - Generate `fn stt() -> impl SttConversationBuilder` method that panics or delegates to separate STT engine

#### Validation & Testing (4 items):
13. **Test `stt_engine!` macro completeness** - Verify generated SttConversationBuilder impl has all 16 methods + 2 associated types from trait definition
14. **Test `tts_engine!` macro completeness** - Verify generated TtsConversationBuilder impl has all 3 methods + 1 associated type from trait definition
15. **Test `FluentVoice` trait integration** - Verify `MyEngine::stt()` and `MyEngine::tts()` entry points work correctly with generated builders
16. **Update macro documentation** - Add examples showing all new methods and FluentVoice trait usage patterns in macro docstrings

**Priority: IMMEDIATE** - This blocks all engine implementations including Whisper and Kokoros.

## 📋 Task List

### Phase 1: Analysis & Architecture (Research Heavy 🔬) ✅ COMPLETED
- [x] **Deep dive into existing implementations**
  - [x] Analyze `Kokoros` TTS engine (`speakrs_kokoros` crate) - ONNX inference and audio generation
  - [x] Analyze `Whisper` transcription builder - current fluent API patterns
  - [x] Analyze `KoffeeCandle` (wake-word detection) - integration opportunities for voice activation
  - [x] Map audio format compatibility between Kokoros, Whisper, and fluent-voice traits
  - [x] Document current async patterns and streaming approaches
- [x] **Design bridge architecture**
  - [x] Design `KokorosEngine` struct implementing `TtsEngine` trait
  - [ ] Design `WhisperEngine` struct implementing `SttEngine` trait  
  - [ ] Design `KoffeeWakeWord` integration for voice activation triggers
  - [x] Plan audio pipeline: PCM format standardization across engines
  - [x] Design error mapping from engine-specific errors to `VoiceError`

### Phase 2: Kokoros TTS Integration 🎙️ 🚧 IN PROGRESS
- [x] **Create `kokoros-fluent-voice` adapter crate**
  - [x] New crate in workspace: `kokoros-fluent-voice/`
  - [x] Implement `KokorosEngine` struct with `TtsEngine` trait
  - [x] Implement `KokorosConversationBuilder` with fluent builder pattern
  - [x] Bridge `TTSKoko` ONNX inference to fluent-voice streaming API
- [x] **Audio pipeline implementation**
  - [x] Convert Kokoros WAV generation to real-time audio streaming
  - [x] Implement `TtsConversation::into_stream()` → `Stream<Item = i16>`
  - [x] Add proper audio buffering and chunking for real-time playback
  - [x] Handle audio format conversion (Kokoros WAV → standard PCM 16kHz)
- [x] **Multi-speaker support** 
  - [x] Map fluent-voice `Speaker` trait to Kokoros voice styles
  - [x] Implement voice blending (e.g., "af_sarah.4+af_nicole.6" style syntax)
  - [x] Support speed, language, and style modifiers per speaker
- [ ] **Fix compilation issues**
  - [ ] Resolve clap std dependency conflicts in fluent-voice core
  - [ ] Test basic compilation and integration
  - [ ] Add missing VoiceId and Language constructors (partially done)

### Phase 3: Whisper STT Integration 🎧
- [ ] **Create `fluent-voice-whisper` adapter crate**
  - [ ] New crate in workspace: `fluent-voice-whisper/`
  - [ ] Implement `WhisperEngine` struct with `SttEngine` trait
  - [ ] Implement `WhisperConversationBuilder` with all STT configuration options
  - [ ] Bridge existing `TranscribeBuilder` to fluent-voice pattern
- [ ] **Transcript streaming implementation**
  - [ ] Convert `WhisperStream<TtsChunk>` to `TranscriptStream`
  - [ ] Implement `SttConversation::into_stream()` with proper transcript segments
  - [ ] Add real-time transcription support (live microphone input)
  - [ ] Map Whisper confidence scores to transcript segment metadata
- [ ] **Advanced STT features**
  - [ ] Implement speaker diarization using Whisper capabilities
  - [ ] Add word-level timestamps and punctuation insertion
  - [ ] Integrate VAD mode support with Whisper processing
  - [ ] Support multiple audio input formats (file, microphone, streams)

### Phase 4: Integration Testing & Examples 🧪
- [ ] **End-to-end testing**
  - [ ] Create test suite for Kokoros TTS integration
  - [ ] Create test suite for Whisper STT integration  
  - [ ] Test round-trip: text → Kokoros TTS → audio file → Whisper STT → text
  - [ ] Performance benchmarks comparing direct usage vs. fluent-voice wrapper
- [ ] **Comprehensive examples**
  - [ ] Basic Kokoros TTS example with multiple speakers and voice styles
  - [ ] Basic Whisper STT example with file and microphone input
  - [ ] Real-time conversation: Kokoros TTS + Whisper STT pipeline
  - [ ] Wake-word activation: Koffee detection → Kokoros TTS response
  - [ ] Integration with existing ElevenLabs/OpenAI engines for fallback scenarios
- [ ] **Error handling validation**
  - [ ] Test graceful degradation when models fail to load
  - [ ] Test audio format mismatch handling
  - [ ] Test timeout and resource exhaustion scenarios

### Phase 5: Optimization & Production Polish 🚀
- [ ] **Performance optimization**
  - [ ] Memory usage optimization for long conversations
  - [ ] Async streaming optimization for low-latency scenarios
  - [ ] GPU acceleration integration (WGPU/Candle optimization)
  - [ ] Audio buffer size tuning for optimal latency vs. quality
- [ ] **Documentation & Developer Experience**
  - [ ] API documentation with practical examples
  - [ ] Migration guide from direct Koffee/Whisper usage
  - [ ] Troubleshooting guide for common integration issues
  - [ ] Performance tuning guide for different use cases

## 🎯 Next Steps Priority Order

1. **[HIGH]** Fix compilation issues in `kokoros-fluent-voice` adapter
   - Resolve clap std dependency conflicts
   - Clean up fluent-voice core dependencies
   - Test basic compilation without std conflicts
2. **[HIGH]** Complete Kokoros TTS integration
   - Add working examples demonstrating the fluent API
   - Test end-to-end audio generation
   - Add proper error handling and logging
3. **[MEDIUM]** Design and create `whisper-fluent-voice` adapter crate structure  
4. **[MEDIUM]** Implement basic Whisper STT integration with fluent-voice traits
5. **[MEDIUM]** Create comprehensive examples and testing infrastructure
6. **[LOW]** Add Koffee wake-word detection integration

## 📝 Notes & Research Links

### Key Dependencies to Research
- **ONNX Runtime (ORT)**: Kokoros model loading and inference patterns ✅
- **Kokoros Audio Pipeline**: WAV generation, voice style blending, and ONNX integration ✅
- **Whisper Transcription**: Async streaming patterns and transcript chunk handling
- **Koffee Wake-word**: Voice activation detection for triggering TTS responses
- **Rubato/RustFFT**: Audio format conversion and signal processing integration

### Architecture Decisions Made ✅
- [x] Choose audio format standardization approach: i16 PCM samples via Stream<Item = i16>
- [x] Plan model loading strategy: eager initialization in KokorosEngine::new()
- [x] Design speaker management: KokorosSpeaker with voice style mapping to fluent-voice Speaker traits
- [x] Error mapping strategy: normalize to VoiceError with engine-specific context

### Architecture Decisions Still Needed  
- [ ] Decide on streaming strategy for Whisper: pull-based vs. push-based transcript delivery
- [ ] Wake-word integration: event-driven vs. polling-based activation patterns
- [ ] Dependency management: how to handle conflicting std vs no-std requirements across workspace

### Current Issues
- **Compilation blockers**: clap dependency requires std feature, conflicts with no-std fluent-voice goals
- **Integration gaps**: Need to complete Whisper STT adapter following same pattern as Kokoros
- **Testing needs**: Require model files for end-to-end testing

---

**Status**: 🟡 In Progress - Phase 2 Implementation | **Last Updated**: Kokoros Adapter Created | **Next Review**: After compilation fixes

### Recent Progress ✅
- Created `kokoros-fluent-voice` adapter crate with complete trait implementations
- Implemented `KokorosEngine`, `KokorosSpeaker`, and `KokorosConversationBuilder`
- Added audio streaming pipeline from Kokoros ONNX output to fluent-voice Stream<i16>
- Designed multi-speaker conversation support with voice style blending
- Added VoiceId and Language constructors to fluent-voice core

### Immediate Blockers 🚨
- clap dependency std requirement conflicts with no-std fluent-voice design
- Need to either: (1) Fix fluent-voice core dependencies, or (2) Work around in adapter crate