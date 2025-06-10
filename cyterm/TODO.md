# CyTerm Development TODO

**State:** Initial
**Last Updated:** 2025-06-08

## Primary Objective
> **AWAITING OBJECTIVE** - Please provide the primary goal for this session.

## Critical Issues (from README)
- [ ] Missing ONNX Model: Silero VAD model file not included
- [ ] Incomplete ASR Implementation: Whisper integration needs proper API calls
- [ ] Missing Audio Pipeline: No microphone capture or real-time processing loop
- [ ] No Main Application: Missing entry point that connects all components

## Current State Analysis
Based on project exploration:
- Main.rs exists but only demonstrates wake-word detection
- ASR module structure exists but likely incomplete
- Software rendering backend implemented (soft_backend.rs)
- Terminal UI components present
- Wake word detection implemented
- VAD integration present

## Dependencies (Cargo.toml)
- ratatui: 0.30.0-alpha.4
- cosmic-text: 0.14
- cpal: 0.15
- candle-*: 0.9.1 (git/main)
- ort: 2.0.0-rc.10
- Various supporting libraries

## Questions for Clarity
1. What is the primary objective for this session?
2. Should we focus on:
   - Getting basic compilation working?
   - Implementing the missing audio pipeline?
   - Completing the ASR integration?
   - Creating the main application loop?
   - Something else?
3. Do you have the Silero VAD ONNX model available?
4. What specific functionality would constitute success?

## Action Items
*Waiting for objective before defining tasks*

---
## Research Notes
*To be populated based on objective*

## Cited Examples
*To be populated during research phase*
