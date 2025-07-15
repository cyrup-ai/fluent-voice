# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Structure

This is a Rust workspace with 12 member crates implementing a comprehensive voice processing ecosystem:

- **`fluent-voice/`** - Core pure-trait fluent builder API for TTS/STT engines
- **`kyutai/`** - Moshi language model implementation using Kyutai research (currently has build errors)
- **`koffee/`** - Cross-platform wake-word detection using Candle ML framework
- **`whisper/`** - Speech-to-text implementation
- **`elevenlabs/`** - ElevenLabs TTS API integration
- **`dia/`** - Voice conversation system with justfile commands
- **`cyterm/`** - Terminal-based voice interface
- **`livekit/`** - Real-time audio/video streaming
- **`vad/`** - Voice Activity Detection
- **`video/`** - Video processing components
- **`animator/`** - Animation and visual components
- **`domain/`** - Shared domain types and error handling

## Development Commands

**Standard workflow (from CONVENTIONS.md):**
```bash
# Format and check - run before any development
cargo fmt && cargo check --message-format short --quiet

# Testing (requires nextest)
cargo nextest run --message-format short --quiet

# Build release
cargo build --release --message-format short --quiet

# Run (workspace level)
cargo run --message-format short --quiet
```

**Using workspace-level justfile:**
```bash
just check    # Format, check, and clippy across entire workspace
just test     # Format and run tests across entire workspace
just build    # Format and build release across entire workspace
just hakari   # Regenerate workspace-hack after dependency changes
just lock     # Make all Cargo.toml files read-only
just unlock   # Make all Cargo.toml files writable
just status   # Show Cargo.toml file permissions
```

**Using dia-specific justfile (packages/dia/):**
```bash
just check    # Format, check, and clippy with auto-fix
just test     # Format and run tests
just build    # Format and build release
just run      # Format and run
just dia      # Run dia binary with metal features and custom args
```

**Dependency Management with cargo-hakari:**
This workspace uses cargo-hakari for optimized dependency compilation. See [CARGO_HAKARI.md](./CARGO_HAKARI.md) for comprehensive guidance.

**Critical hakari workflow:**
```bash
# ALWAYS regenerate after ANY dependency change:
cargo hakari generate

# Verify before committing:
cargo hakari verify

# If adding/removing crates (not needed for dependency updates):
cargo hakari manage-deps
```

**Important:** 
- All member crates depend on `workspace-hack` 
- Dependencies already in workspace-hack are commented out in member Cargo.tomls
- Never manually edit workspace-hack/Cargo.toml - always use `cargo hakari generate`

## Architecture & Design Philosophy

**CRITICAL: Fluent-Voice API Only**
- ❌ **NO OTHER APIs ALLOWED** - All libraries must use ONLY fluent-voice builders
- ❌ **NO DIRECT ENGINE CALLS** - No direct calls to ElevenLabs, OpenAI, Whisper, etc.
- ✅ **ONLY FLUENT-VOICE BUILDERS** - All voice operations must go through fluent-voice API
- ✅ **UNIFIED INTERFACE** - Single consistent API across all voice functionality

**Core Design Pattern:**
- **"One fluent chain → One matcher closure → One `.await?`"**
- Pure-trait architecture with engine-agnostic implementations
- Fluent builder pattern for all voice operations
- Polymorphic builders that branch based on input type

**Key Architectural Traits:**
- `TtsEngine` / `SttEngine` - Engine registration and initialization
- `TtsConversationBuilder` - TTS fluent configuration API
- `SttBuilder` - Base STT with polymorphic branching to:
  - `MicrophoneBuilder` - Live microphone transcription
  - `TranscriptionBuilder` - File-based transcription
- `Speaker` / `SpeakerBuilder` - Voice and speaker configuration

**Critical Async Rules:**
- ❌ NEVER use `async_trait` or `async fn` in public APIs
- ❌ NEVER return `Box<dyn Future>` or `Pin<Box<dyn Future>>`
- ✅ Return synchronous interfaces that provide awaitable Streams or Futures
- ✅ Use `cyrup-ai/async_task` crate for async operations

## Code Quality Standards

**Strict Requirements:**
- ✅ All code MUST pass `cargo check --message-format short --quiet` without warnings
- ✅ Use `nextest` for all testing
- ✅ Maximum 300 lines per file - decompose when exceeded
- ✅ Tests in `tests/` directory, not co-located with source
- ✅ Use `Result<T, E>` with custom errors
- ✅ No `unwrap()` except in tests with explicit error handling

**Forbidden Practices:**
- ❌ **NO DIRECT ENGINE APIS** - Never call ElevenLabs, OpenAI, Whisper, etc. directly
- ❌ **NO BYPASSING FLUENT-VOICE** - All voice operations must use fluent-voice builders
- ❌ No suppression of compiler/clippy warnings
- ❌ No underscore variable naming to hide warnings
- ❌ No `#[allow(dead_code)]` or other suppressing annotations
- ❌ No commenting out code or disabling modules
- ❌ No "TODO: in production..." comments
- ❌ No blocking code anywhere (including tests)

## Current Build Status

**Critical Issues:**
- `kyutai/` has multiple build errors (StreamTensor method issues, VarMap API mismatches, type errors)
- 10+ warnings across workspace (mostly unused imports)
- Active development on Moshi language model implementation

**Before Development:**
1. Run `cargo fmt && cargo check --message-format short --quiet`
2. Fix all errors and warnings before proceeding
3. Ensure you're starting from a clean state

## Key Dependencies

**ML/Audio Processing:**
- `candle-*` (from GitHub main branch) for neural networks and ML
- `tokio` for async runtime
- `hound` for audio file handling
- `cpal` for audio I/O (optional, enabled with microphone feature)
- `rubato` for audio resampling (optional)
- `symphonia` for audio format support (optional)

**Development Tools:**
- `nextest` for fast parallel testing
- `tracing` for structured logging
- `clap` for CLI interfaces
- `serde` for serialization
- `cyrup-sugars` for enhanced async syntax patterns

**Workspace Management:**
- `fluent-voice-workspace-hack` for dependency optimization
- `cargo-hakari` for workspace dependency management

## Testing Strategy

- Use `nextest` exclusively for test execution
- Focus on end-user testing scenarios
- Tests must prove functionality works for users
- No blocking code in tests
- Comprehensive integration tests in `tests/` directories

## Documentation Standards

- Extensive inline documentation with examples
- Clear trait definitions and usage patterns
- Practical examples showing real-world usage
- API documentation focuses on "how to use" not "what it does"

## Workspace Management

- Uses Cargo workspace with resolver = "3"
- Shared dependency versions across all crates
- Edition = "2024" for cutting-edge Rust features
- All crates follow the same quality standards
- Use `cargo workspace` commands for cross-crate operations

## Feature Flags

**fluent-voice core features:**
- `default = ["metal", "microphone"]` - Metal acceleration + microphone support
- `cuda`, `cudnn`, `metal`, `accelerate`, `mkl` - ML acceleration backends
- `microphone`, `encodec`, `mimi`, `snac` - Audio processing features
- `tokio-runtime`, `wasm-runtime`, `std_async` - Runtime variants

**kyutai features:**
- `default = []` - No default features
- `cuda`, `metal` - GPU acceleration options

## Single Test Execution

**Run specific tests:**
```bash
# Run a single test file
cargo nextest run --package package-name test-name

# Run tests matching a pattern
cargo nextest run --package fluent-voice test_pattern

# Run tests in specific crate
cd packages/kyutai && cargo nextest run
```