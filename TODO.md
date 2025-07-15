# Domain-Builder Decoupling TODO

## Phase 1: Extract Builder Traits from Domain
- [ ] Move TtsConversationBuilder trait from domain to fluent-voice package. DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.
- [ ] Act as an Objective QA Rust developer - rate the work performed previously on moving TtsConversationBuilder trait against requirements for clean decoupling and minimal surgical changes
- [ ] Move TtsConversationChunkBuilder trait from domain to fluent-voice package. DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.
- [ ] Act as an Objective QA Rust developer - rate the work performed previously on moving TtsConversationChunkBuilder trait against requirements for eliminating coupling and scope adherence
- [ ] Move SttConversationBuilder trait from domain to fluent-voice package. DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.
- [ ] Act as an Objective QA Rust developer - rate the work performed previously on moving SttConversationBuilder trait against requirements for clean architectural separation and minimal changes
- [ ] Move all *Builder traits (MicrophoneBuilder, TranscriptionBuilder, etc.) from domain to fluent-voice. DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.
- [ ] Act as an Objective QA Rust developer - rate the work performed previously on moving all builder traits against requirements for complete decoupling and surgical precision

## Phase 2: Remove Concrete Implementations from Domain
- [ ] Move DefaultWakeWordDetector from domain to fluent-voice package. DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.
- [ ] Act as an Objective QA Rust developer - rate the work performed previously on moving DefaultWakeWordDetector against requirements for pure domain abstractions and minimal scope changes
- [ ] Move DefaultWakeWordBuilder from domain to fluent-voice package. DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.
- [ ] Act as an Objective QA Rust developer - rate the work performed previously on moving DefaultWakeWordBuilder against requirements for eliminating concrete implementations from domain
- [ ] Remove FluentVoiceImpl from domain package (move to fluent-voice if needed). DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.
- [ ] Act as an Objective QA Rust developer - rate the work performed previously on removing FluentVoiceImpl against requirements for pure domain abstractions and minimal changes

## Phase 3: Purify Domain FluentVoice Trait
- [ ] Remove all `impl XxxBuilder` return types from domain FluentVoice trait. DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.
- [ ] Act as an Objective QA Rust developer - rate the work performed previously on removing builder return types against requirements for eliminating circular dependencies and scope adherence
- [ ] Convert domain FluentVoice trait to pure abstraction without builder references. DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.
- [ ] Act as an Objective QA Rust developer - rate the work performed previously on purifying FluentVoice trait against requirements for clean domain abstractions and minimal surgical changes
- [ ] Remove all TtsConversationExt, SttConversationExt references from domain. DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.
- [ ] Act as an Objective QA Rust developer - rate the work performed previously on removing conversation extensions against requirements for domain purity and scope boundary compliance

## Phase 4: Update Fluent-Voice Package Structure
- [ ] Create comprehensive builder module in fluent-voice with all moved builder traits. DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.
- [ ] Act as an Objective QA Rust developer - rate the work performed previously on creating builder module against requirements for complete functionality and minimal changes
- [ ] Implement FluentVoice trait in fluent-voice with concrete builder return types. DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.
- [ ] Act as an Objective QA Rust developer - rate the work performed previously on implementing FluentVoice trait against requirements for proper concrete implementations and scope adherence
- [ ] Update fluent-voice lib.rs exports to include all builder traits and implementations. DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.
- [ ] Act as an Objective QA Rust developer - rate the work performed previously on updating exports against requirements for complete API exposure and minimal changes

## Phase 5: Clean Domain Package
- [ ] Remove builders module from domain package completely. DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.
- [ ] Act as an Objective QA Rust developer - rate the work performed previously on removing builders module against requirements for pure domain package and surgical precision
- [ ] Update domain lib.rs to export only pure domain types (no builder references). DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.
- [ ] Act as an Objective QA Rust developer - rate the work performed previously on cleaning domain exports against requirements for pure abstractions and scope compliance
- [ ] Verify domain package has zero dependencies on builder concepts. DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.
- [ ] Act as an Objective QA Rust developer - rate the work performed previously on verifying zero builder dependencies against requirements for complete decoupling and minimal scope

## Phase 6: Update Cross-Package Dependencies
- [ ] Fix all import statements in fluent-voice to use local builder traits instead of domain. DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.
- [ ] Act as an Objective QA Rust developer - rate the work performed previously on fixing imports against requirements for proper dependency flow and minimal changes
- [ ] Update examples to use fluent-voice builder API without referencing domain builders. DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.
- [ ] Act as an Objective QA Rust developer - rate the work performed previously on updating examples against requirements for clean API usage and scope adherence
- [ ] Verify workspace compiles with clean one-way dependency flow (fluent-voice → domain only). DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.
- [ ] Act as an Objective QA Rust developer - rate the work performed previously on verifying dependency flow against requirements for complete decoupling and architectural cleanliness