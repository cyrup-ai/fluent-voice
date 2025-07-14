# ZERO TOLERANCE FOR WARNINGS - Complete Error and Warning Elimination

## OBJECTIVE: 0 Errors, 0 Warnings

**Current Status:** 6 Errors, 13 Warnings
**Target:** 0 Errors, 0 Warnings

---

## ERRORS (6 total)

### Naga WriteColor Trait Errors (3)
1. **ERROR**: `/Users/davidmaple/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/naga-26.0.0/src/error.rs:50:17` - WriteColor trait not implemented for std::string::String
2. **QA**: Rate naga WriteColor error fix quality (1-10)
3. **ERROR**: `/Users/davidmaple/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/naga-26.0.0/src/front/wgsl/error.rs:112:20` - WriteColor trait not implemented for std::string::String  
4. **QA**: Rate naga WriteColor error fix quality (1-10)
5. **ERROR**: `/Users/davidmaple/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/naga-26.0.0/src/span.rs:330:20` - WriteColor trait not implemented for std::string::String
6. **QA**: Rate naga WriteColor error fix quality (1-10)

### Whisper Module Resolution Errors (3)
7. **ERROR**: `packages/whisper/src/multilingual.rs:1:13` - unresolved imports `crate::Model`, `crate::token_id`: no `Model` in the root, no `token_id` in the root
8. **QA**: Rate whisper multilingual import fix quality (1-10)
9. **ERROR**: `packages/whisper/src/whisper.rs:22:12` - unresolved import `crate::microphone`: could not find `microphone` in the crate root
10. **QA**: Rate whisper microphone import fix quality (1-10)  
11. **ERROR**: `packages/whisper/src/whisper.rs:520:42` - failed to resolve: could not find `pcm_decode` in the crate root
12. **QA**: Rate whisper pcm_decode import fix quality (1-10)

---

## WARNINGS (13 total)

### Unused Import Warnings (7)
13. **WARNING**: `packages/koffee/src/audio/encoder.rs:124:13` - variable does not need to be mutable
14. **QA**: Rate koffee mutable variable fix quality (1-10)
15. **WARNING**: `packages/dia/src/audio/pcm.rs:3:14` - unused import: `Context`
16. **QA**: Rate dia pcm unused import fix quality (1-10)
17. **WARNING**: `packages/dia/src/codec.rs:6:31` - unused import: `anyhow`
18. **QA**: Rate dia codec anyhow import fix quality (1-10)
19. **WARNING**: `packages/dia/src/codec.rs:24:20` - unused imports: `SAMPLE_RATE`, `normalize_loudness`, and `to_24k_mono`
20. **QA**: Rate dia codec audio imports fix quality (1-10)
21. **WARNING**: `packages/dia/src/generation.rs:38:5` - unused imports: `Timer`, `log_gpu_memory`, and `self`
22. **QA**: Rate dia generation imports fix quality (1-10)
23. **WARNING**: `packages/dia/src/setup.rs:1:14` - unused import: `Context`
24. **QA**: Rate dia setup Context import fix quality (1-10)
25. **WARNING**: `packages/whisper/src/stream.rs:9:27` - unused import: `TranscriptStream`
26. **QA**: Rate whisper stream import fix quality (1-10)

### Code Structure Warnings (4)  
27. **WARNING**: `packages/whisper/src/types.rs:113:7` - unexpected `cfg` condition value: `internal`
28. **QA**: Rate whisper cfg condition fix quality (1-10)
29. **WARNING**: `packages/whisper/src/builder.rs:68:13` - unreachable pattern: no value can reach this
30. **QA**: Rate whisper unreachable pattern fix quality (1-10)
31. **WARNING**: `packages/whisper/src/builder.rs:47:34` - unused variable: `path`
32. **QA**: Rate whisper unused path variable fix quality (1-10)
33. **WARNING**: `packages/dia/src/setup.rs:22:5` - unused variable: `weights_path`
34. **QA**: Rate dia setup weights_path variable fix quality (1-10)
35. **WARNING**: `packages/dia/src/setup.rs:23:5` - unused variable: `tokenizer_path`  
36. **QA**: Rate dia setup tokenizer_path variable fix quality (1-10)

### Dead Code Warnings (2)
37. **WARNING**: `packages/dia/src/codec.rs:29:8` - static `ENCODEC_MODEL` is never used
38. **QA**: Rate dia codec ENCODEC_MODEL dead code fix quality (1-10)
39. **WARNING**: `packages/dia/src/codec.rs:31:4` - function `load_encodec` is never used
40. **QA**: Rate dia codec load_encodec dead code fix quality (1-10)
41. **WARNING**: `packages/dia/src/generation.rs:146:5` - fields `gpu_config`, `memory_pool`, and `compute_dtype` are never read
42. **QA**: Rate dia generation struct fields fix quality (1-10)

---

## CONSTRAINTS & STANDARDS

- ❌ NO EXCUSES: Fix every single error and warning
- ❌ NO SHORTCUTS: Production quality code only
- ❌ NO BLOCKING CODE: Unless explicitly approved by David Maple with timestamp
- ✅ RESEARCH THOROUGHLY: Understand each issue and all call sites
- ✅ ASK QUESTIONS: When uncertain, ask David for clarification
- ✅ QA EVERYTHING: Score 9+ required, redo if less
- ✅ TEST LIKE USER: Verify actual functionality works

---

**SUCCESS CRITERIA: `cargo check --features metal --message-format short --quiet` shows ZERO errors and ZERO warnings**