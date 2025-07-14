# ZERO TOLERANCE FOR ERRORS - Complete Build Restoration

## OBJECTIVE: 0 Errors, 0 Warnings

**Current Status:** 200+ Errors, 50+ Warnings  
**Target:** 0 Errors, 0 Warnings

---

## CRITICAL ERROR CATEGORIES

### 1. Missing Dependencies (HIGH PRIORITY)
**Packages affected:** fluent-voice, livekit, cyterm, whisper

1. **fluent-voice missing deps**: serde_json, cpal, byteorder, tracing
   - DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.
2. **QA**: Act as an Objective QA Rust developer - rate the fluent-voice dependency additions (1-10) for correctness and minimal impact
3. **livekit missing deps**: log, tracing, gpui, core_video, coreaudio, sugarloaf
   - DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.
4. **QA**: Act as an Objective QA Rust developer - rate the livekit dependency additions (1-10) for correctness and minimal impact

### 2. Feature Configuration Errors (HIGH PRIORITY)
**Error:** "At least one audio feature must be enabled: microphone, encodec, mimi, or snac"

5. **Enable required audio features in livekit and cyterm**
   - DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.
6. **QA**: Act as an Objective QA Rust developer - rate the audio feature configuration (1-10) for correctness

### 3. Module Redefinition Errors (HIGH PRIORITY)
**Package:** cyterm - multiple definitions of `error`, `llm`, `VoiceActivityDetector`

7. **Fix cyterm module redefinitions and import conflicts**
   - DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.
8. **QA**: Act as an Objective QA Rust developer - rate the cyterm module conflict resolution (1-10)

### 4. Missing Files (HIGH PRIORITY)
**Files:** README.md, silero_vad.onnx in cyterm

9. **Create or locate missing README.md and onnx files for cyterm**
   - DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.
10. **QA**: Act as an Objective QA Rust developer - rate the missing file resolution (1-10)

### 5. Trait Implementation Issues (MEDIUM PRIORITY)
**Package:** fluent-voice - missing trait methods, type parameter mismatches

11. **Fix fluent-voice trait implementations and type parameters**
    - DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.
12. **QA**: Act as an Objective QA Rust developer - rate the trait implementation fixes (1-10)

### 6. Import Resolution Errors (MEDIUM PRIORITY)
**Packages:** Multiple - unresolved imports and missing modules

13. **Resolve all unresolved import errors across packages**
    - DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.
14. **QA**: Act as an Objective QA Rust developer - rate the import resolution fixes (1-10)

### 7. Workspace Dependencies (LOW PRIORITY)
**Action:** Update workspace-hack after dependency changes

15. **Run cargo hakari generate to update workspace-hack**
    - DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.
16. **QA**: Act as an Objective QA Rust developer - rate the hakari update (1-10)

### 8. Final Verification (LOW PRIORITY)

17. **Verify zero errors and warnings with full build**
    - DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.
18. **QA**: Act as an Objective QA Rust developer - rate the final build verification (1-10)

---

## CONSTRAINTS & STANDARDS

- ❌ NO EXCUSES: Fix every single error and warning
- ❌ NO SHORTCUTS: Production quality code only  
- ❌ NO BLOCKING CODE: Unless explicitly approved by David Maple with timestamp
- ❌ NO unwrap() anywhere in src/* or examples/*
- ❌ NO expect() in src/* or examples/* 
- ✅ DO USE expect() in ./tests/*
- ✅ RESEARCH THOROUGHLY: Understand each issue and all call sites
- ✅ ASK QUESTIONS: When uncertain, ask David for clarification
- ✅ QA EVERYTHING: Score 9+ required, redo if less
- ✅ TEST LIKE USER: Verify actual functionality works

---

**SUCCESS CRITERIA: `cargo check --features metal --message-format short --quiet` shows ZERO errors and ZERO warnings**