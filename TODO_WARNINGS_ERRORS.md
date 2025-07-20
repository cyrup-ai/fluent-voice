# COMPREHENSIVE ERROR AND WARNING FIXES TODO

## CURRENT STATUS: 🚨 MULTIPLE ERRORS AND WARNINGS 

### ERRORS (BLOCKING COMPILATION)

1. **ElevenLabs Build Script API Failure** 
   - Error: `reqwest::Error { kind: Request, url: "https://api.elevenlabs.io/v1/voices", source: hyper_util::client::legacy::Error(Connect, Custom { kind: Other, error: Custom { kind: InvalidData, error: InvalidCertificate(UnknownIssuer) } }) }`
   - Location: `packages/elevenlabs/build.rs`
   - Fix Required: Make build script work offline with fallback voice data

2. **Commented Out Modules in Kyutai**
   - Location: `packages/kyutai/src/lib.rs`
   - Disabled: `seanet`, `stream_both` 
   - Fix Required: Uncomment and implement missing modules

### WARNINGS (MUST BE FIXED)

3. **Unused Import: VoiceError in domain/audio_isolation.rs**
   - Warning: `unused import: crate::voice_error::VoiceError`
   - Location: `packages/domain/src/audio_isolation.rs:3:5`

4. **Unused Import: VoiceError in domain/audio_stream.rs**
   - Warning: `unused import: VoiceError`
   - Location: `packages/domain/src/audio_stream.rs:1:25`

5. **Unused Import: VoiceError in domain/sound_effects.rs**
   - Warning: `unused import: crate::voice_error::VoiceError`
   - Location: `packages/domain/src/sound_effects.rs:3:5`

6. **Unused Import: VoiceError in domain/speech_to_speech.rs**
   - Warning: `unused import: crate::voice_error::VoiceError`
   - Location: `packages/domain/src/speech_to_speech.rs:3:5`

7. **Unused Import: VoiceError in domain/tts_conversation.rs**
   - Warning: `unused import: crate::voice_error::VoiceError`
   - Location: `packages/domain/src/tts_conversation.rs:3:5`

8. **Unused Import: VoiceError in domain/voice_clone.rs**
   - Warning: `unused import: crate::voice_error::VoiceError`
   - Location: `packages/domain/src/voice_clone.rs:3:5`

9. **Unused Import: VoiceError in domain/voice_discovery.rs**
   - Warning: `unused import: crate::voice_error::VoiceError`
   - Location: `packages/domain/src/voice_discovery.rs:3:5`

10. **Unused Import: VoiceError in domain/wake_word.rs**
    - Warning: `unused import: crate::voice_error::VoiceError`
    - Location: `packages/domain/src/wake_word.rs:3:5`

11. **Unnecessary Mutable Variable in koffee**
    - Warning: `variable does not need to be mutable`
    - Location: `packages/koffee/src/audio/encoder.rs:184:13`

### ADDITIONAL INVESTIGATION NEEDED

12. **Check for other commented modules across workspace**
13. **Verify all library versions are latest via cargo search**
14. **Ensure code actually works end-to-end**

## SUCCESS CRITERIA
- ✅ 0 (Zero) Errors
- ✅ 0 (Zero) Warnings  
- ✅ All modules uncommented and working
- ✅ Code tested and functional