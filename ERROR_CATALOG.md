# COMPREHENSIVE ERROR AND WARNING CATALOG

Generated at start of work to track ALL compilation issues across the workspace.

## CRITICAL FINDINGS 

### Feature Configuration Issues
- **fluent-voice**: Requires either candle acceleration features (cuda, metal, accelerate, mkl) AND audio features (microphone, encodec, mimi, snac)
- **animator/livekit**: Both require audio features to be enabled
- **CUDA Build Failure**: cudarc dependency fails to build due to missing nvcc (CUDA toolkit not installed)

### Import Resolution Issues 
- **fluent-voice**: Missing `fluent_voice_domain` imports - should be `domain` package
- **cpal imports**: Missing throughout fluent-voice when microphone feature disabled
- **candle imports**: Missing Tensor, Device, Config types when features disabled

### Core API Issues
- **FluentVoice**: Main entry point not implemented properly 
- **Prelude**: Not exporting required types
- **Builder patterns**: Missing method implementations

### Thread Safety Issues in video package
- **CVBuffer pointers**: `*mut __CVBuffer` not Send/Sync safe across threads
- **48 compilation errors** in video package alone

### External Dependency Compatibility 
- **GPUI**: Version conflicts in livekit
- **CoreAudio**: API changes in livekit  
- **core-video**: Method name changes in video package

## PACKAGE-BY-PACKAGE BREAKDOWN

### fluent-voice (142 errors, 83 warnings) ❌
**Critical for examples to work**
- Feature gating prevents compilation without acceleration/audio features
- Import path issues: `fluent_voice_domain` → `domain`
- Missing cpal imports when microphone feature disabled  
- Missing candle imports when acceleration features disabled
- Unused variable warnings (83 total)

### domain (0 errors checked) ✅
- Contains the actual domain types needed by examples
- Should be aliased as `fluent_voice_domain` for compatibility

### video (48 errors, 12 warnings) ❌
- Thread safety issues with CVBuffer pointers
- API compatibility issues with core-video crate
- Missing Debug/Default implementations

### livekit (21 errors, 20 warnings) ❌  
- GPUI version compatibility issues
- CoreAudio API changes
- Feature gating issues (requires audio features)
- ratagpu renderer API changes

### animator (2 errors, multiple warnings) ❌
- cpal Host API changes (no input_devices/default_input_device methods)
- Feature gating issues (requires audio features)

### whisper (0 errors, 8 warnings) ⚠️
- Unused code warnings but compiles successfully

### dia (needs verification) ❓
- Required for TTS example but not checked yet

### kyutai (needs verification) ❓  
- Known to have build issues from conversation history

## EXAMPLES STATUS

### stt.rs ❌ BLOCKED
**Imports needed:**
- `fluent_voice::prelude::*` 
- `fluent_voice_domain::{AudioFormat, Diarization, Language, MicBackend, Punctuation, SpeechSource, TimestampsGranularity, VadMode, WordTimestamps}`

**API calls needed:**
- `FluentVoice::stt().conversation()`
- Builder pattern with engine_config, with_source, vad_mode, etc.
- `on_chunk()` and `listen()` methods

### tts.rs ❌ BLOCKED  
**Imports needed:**
- `fluent_voice::prelude::*`
- `fluent_voice_domain::{Language, Speaker, VocalSpeedMod, VoiceId}`

**API calls needed:**
- `FluentVoice::tts().conversation()`
- Builder pattern with engine_config, with_speaker, etc.
- `on_chunk()` and `synthesize()` methods

## ESTIMATED WORK REQUIRED

1. **Feature Configuration**: Enable default features to resolve gating issues
2. **Import Path Fixes**: Update fluent_voice_domain → domain throughout
3. **Core API Implementation**: Implement FluentVoice entry points and builders
4. **Engine Integration**: Fix whisper/dia compilation and integration
5. **Thread Safety**: Wrap CVBuffer in Send/Sync wrappers for video package
6. **Dependency Updates**: Update external dependencies with API changes

## SUCCESS CRITERIA 
- 0 errors, 0 warnings across workspace
- Both examples compile successfully
- Both examples can execute without immediate crashes