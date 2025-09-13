# TURD Milestones - Dependency-Aware Task Organization

## Overview

This directory contains a structured, dependency-aware breakdown of all technical debt violations identified in the fluent-voice codebase. The organization prioritizes foundational work and enables maximum parallelization while respecting architectural dependencies.

## Project Context

**Core Architecture**: Pure-trait fluent builder API with "One fluent chain → One matcher closure → One `.await?`" design philosophy
**Critical Rule**: ALL voice operations must use fluent-voice builders - NO direct engine calls allowed
**Zero-Tolerance Policy**: No blocking operations anywhere in async contexts

## Milestone Structure

```
TURD_MILESTONES/
├── 0_async_architecture_compliance/     # CRITICAL - Foundational
├── 1_configuration_management/          # HIGH - Enables proper implementations  
├── 2_audio_processing_enhancement/      # MEDIUM - Core functionality
├── 3_ml_model_integration/             # MEDIUM - Advanced features
├── 4_example_quality_improvement/       # LOW - Documentation quality
└── 5_code_quality_cleanup/             # LOW - Independent cleanup
```

## Dependency Matrix

| Milestone | Dependencies | Can Start After | Parallel With |
|-----------|--------------|------------------|---------------|
| 0_async_architecture_compliance | None | Immediately | 5_code_quality_cleanup |
| 1_configuration_management | Milestone 0 | Async fixes complete | 5_code_quality_cleanup |
| 2_audio_processing_enhancement | Milestones 0, 1 | Config + Async complete | 3_ml_model_integration, 5_code_quality_cleanup |
| 3_ml_model_integration | Milestones 0, 1 | Config + Async complete | 2_audio_processing_enhancement, 5_code_quality_cleanup |
| 4_example_quality_improvement | Milestones 0, 1, 2 | Core functionality complete | 5_code_quality_cleanup |
| 5_code_quality_cleanup | None | Immediately | All milestones |

## Execution Strategy

### Phase 1: Foundation (CRITICAL)
**Start Immediately:**
- Milestone 0: Async Architecture Compliance
- Milestone 5: Code Quality Cleanup (parallel)

### Phase 2: Configuration (HIGH)  
**Start After Phase 1:**
- Milestone 1: Configuration Management

### Phase 3: Core Features (MEDIUM)
**Start After Phase 2:**
- Milestone 2: Audio Processing Enhancement
- Milestone 3: ML Model Integration (parallel with Milestone 2)

### Phase 4: Polish (LOW)
**Start After Phase 3:**
- Milestone 4: Example Quality Improvement

## Task Breakdown Summary

### Milestone 0: Async Architecture Compliance (3 tasks)
- Fix dia TTS builder blocking violation
- Fix video macOS spawn_blocking usage
- Fix audio stream spawn_blocking references

### Milestone 1: Configuration Management (4 tasks)
- Implement Kyutai audio vocab size configuration
- Implement Kyutai conditioning logic
- Implement fluent voice parameter storage
- Implement Kyutai speaker PCM handling

### Milestone 2: Audio Processing Enhancement (3 tasks)
- Implement 24-bit audio format support
- Implement ElevenLabs timestamp handling
- Implement Dia optimization algorithms

### Milestone 3: ML Model Integration (4 tasks)
- Implement phonemization system
- Implement wake word training pipeline
- Implement Whisper TODO resolution
- Implement Kyutai seanet module

### Milestone 4: Example Quality Improvement (1 task)
- Replace animator mock implementations

### Milestone 5: Code Quality Cleanup (2 tasks)
- Update imprecise language in comments
- Improve search patterns for unwrap detection

## Success Metrics

**Milestone 0 Success**: All blocking operations eliminated, async-first architecture restored
**Milestone 1 Success**: Configuration system operational, no hardcoded values
**Milestone 2 Success**: Full audio format support, comprehensive timestamp tracking
**Milestone 3 Success**: Complete ML model integration, functional wake word detection
**Milestone 4 Success**: Production-quality examples with real integrations
**Milestone 5 Success**: Clean, precise code documentation and tooling

## Validation Checklist

- [ ] All TURD.md violations captured in task structure
- [ ] Dependencies correctly mapped and validated
- [ ] No circular dependencies exist
- [ ] Parallel execution paths are truly independent
- [ ] Each task has clear success criteria
- [ ] Task breakdown supports project's highest-level goals

## Implementation Notes

- Each task file contains detailed technical resolution patterns
- Success criteria are specific and measurable
- Dependencies are explicitly documented
- Architecture impact is clearly stated
- Code examples follow existing codebase patterns

---

**Total Tasks**: 17 tasks across 6 milestones
**Critical Path**: Milestone 0 → Milestone 1 → Milestones 2&3 → Milestone 4
**Parallel Opportunities**: Milestone 5 can run parallel with all others