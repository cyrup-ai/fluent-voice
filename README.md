# Fluent Voice

[![Crates.io](https://img.shields.io/crates/v/fluent_voice.svg)](https://crates.io/crates/fluent_voice)
[![Documentation](https://docs.rs/fluent_voice/badge.svg)](https://docs.rs/fluent_voice)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

A pure-trait fluent builder API for Text-to-Speech (TTS) and Speech-to-Text (STT) engines in Rust.

##  Design Philosophy

Fluent Voice follows a simple, elegant pattern for all voice operations:

**One fluent chain → One matcher closure → One `.await?`**

This design eliminates the complexity of multiple awaits, nested async calls, and scattered error handling that plague many voice APIs.

## ✨ Features

- **🔗 Unified API**: Single interface for both TTS and STT operations
- **⚡ Single Await**: All operations complete with exactly one `.await?`
- **🎭 Multi-Speaker**: Built-in support for conversations with multiple speakers
- **🔧 Engine Agnostic**: Works with any TTS/STT engine through trait implementations
- **🎛️ Rich Configuration**: Comprehensive settings for voice control, audio processing, and recognition
- **📊 Streaming**: Real-time audio streams and transcript processing
- **🛡️ Type Safe**: Leverages Rust's type system for compile-time correctness
- **📝 Well Documented**: Extensive documentation with practical examples

## 📦 Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
fluent_voice = "1.0"
```

For async runtime support, also add:

```toml
tokio = { version = "1", features = ["full"] }
futures-util = "0.3"
```

## 🚀 Quick Start

### Text-to-Speech Example

```rust
use fluent_voice::prelude::*;
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> Result<(), VoiceError> {
    // Note: Requires an engine implementation (see Engine Integration below)
    let mut audio_stream = MyTtsEngine::conversation()
        .with_speaker(
            Speaker::speaker("Alice")
                .voice_id(VoiceId::new("voice-uuid"))
                .with_speed_modifier(VocalSpeedMod(0.9))
                .speak("Hello, world!")
                .build()
        )
        .with_speaker(
            Speaker::speaker("Bob")
                .with_speed_modifier(VocalSpeedMod(1.1))
                .speak("Hi Alice! How are you today?")
                .build()
        )
        .synthesize(|conversation| {
            Ok  => conversation.into_stream(),  // Returns audio stream
            Err(e) => Err(e),
        })
        .await?;  // Single await point

    // Process audio samples
    while let Some(sample) = audio_stream.next().await {
        // Play sample or save to file
        println!("Audio sample: {}", sample);
    }

    Ok(())
}
```

### Speech-to-Text Example

```rust
use fluent_voice::prelude::*;
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> Result<(), VoiceError> {
    let mut transcript_stream = MySttEngine::conversation()
        .with_source(SpeechSource::Microphone {
            backend: MicBackend::Default,
            format: AudioFormat::Pcm16Khz,
            sample_rate: 16_000,
        })
        .vad_mode(VadMode::Accurate)
        .language_hint(Language("en-US"))
        .diarization(Diarization::On)  // Speaker identification
        .word_timestamps(WordTimestamps::On)
        .punctuation(Punctuation::On)
        .listen(|conversation| {
            Ok  => conversation.into_stream(),  // Returns transcript stream
            Err(e) => Err(e),
        })
        .await?;  // Single await point

    // Process transcript segments
    while let Some(result) = transcript_stream.next().await {
        match result {
            Ok(segment) => {
                println!("[{:.2}s] {}: {}",
                    segment.start_ms() as f32 / 1000.0,
                    segment.speaker_id().unwrap_or("Unknown"),
                    segment.text()
                );
            },
            Err(e) => eprintln!("Recognition error: {}", e),
        }
    }

    Ok(())
}
```

## 🏗️ Architecture

Fluent Voice is built around a pure-trait architecture:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Code     │    │  Fluent Voice    │    │ Engine Impls    │
│                 │    │    (Traits)      │    │   (Concrete)    │
├─────────────────┤    ├──────────────────┤    ├─────────────────┤
│ .conversation() │───▶│ TtsConversation  │◀───│ ElevenLabsImpl  │
│ .with_speaker() │    │ Builder          │    │ OpenAIImpl      │
│ .synthesize()   │    │                  │    │ AzureImpl       │
│ .await?         │    │ SttConversation  │    │ GoogleImpl      │
└─────────────────┘    │ Builder          │    │ WhisperImpl     │
                       └──────────────────┘    └─────────────────┘
```

### Core Traits

- **`TtsEngine`** / **`SttEngine`**: Engine registration and initialization
- **`TtsConversationBuilder`** / **`SttConversationBuilder`**: Fluent configuration API
- **`TtsConversation`** / **`SttConversation`**: Runtime session objects
- **`Speaker`** / **`SpeakerBuilder`**: Voice and speaker configuration
- **`TranscriptSegment`** / **`TranscriptStream`**: STT result handling

## 🔧 Configuration Options

### TTS Configuration

```rust
let conversation = engine.conversation()
    .with_speaker(
        Speaker::speaker("Narrator")
            .voice_id(VoiceId::new("narrator-voice"))
            .language(Language("en-US"))
            .with_speed_modifier(VocalSpeedMod(0.8))        // Slower speech
            .with_pitch_range(PitchRange::new(80.0, 200.0)) // Pitch control
            .speak("Your text here")
            .build()
    )
    .language(Language("en-US"))  // Global language setting
    .synthesize(/* matcher */)
    .await?;
```

### STT Configuration

```rust
let conversation = engine.conversation()
    .with_source(SpeechSource::File {
        path: "audio.wav".to_string(),
        format: AudioFormat::Pcm16Khz,
    })
    .vad_mode(VadMode::Accurate)                           // Voice activity detection
    .noise_reduction(NoiseReduction::High)                 // Background noise filtering
    .language_hint(Language("en-US"))                      // Language optimization
    .diarization(Diarization::On)                          // Speaker identification
    .timestamps_granularity(TimestampsGranularity::Word)   // Timing precision
    .punctuation(Punctuation::On)                          // Auto-punctuation
    .listen(/* matcher */)
    .await?;
```

## 🔌 Engine Integration

Fluent Voice is designed to work with any TTS/STT service. Engine implementations provide concrete types that implement the core traits.

### Available Engines

- **ElevenLabs**: `elevenlabs-fluent-voice` (planned)
- **OpenAI**: `openai-fluent-voice` (planned)
- **Azure Cognitive Services**: `azure-fluent-voice` (planned)
- **Google Cloud**: `google-fluent-voice` (planned)
- **Local Whisper**: `whisper-fluent-voice` (planned)

### Implementing Your Own Engine

```rust
use fluent_voice::prelude::*;

// 1. Define your engine struct
pub struct MyEngine {
    api_key: String,
}

// 2. Implement the engine trait
impl TtsEngine for MyEngine {
    type Conv = MyConversationBuilder;

    fn conversation(&self) -> Self::Conv {
        MyConversationBuilder::new(self.api_key.clone())
    }
}

// 3. Implement the conversation builder
pub struct MyConversationBuilder { /* ... */ }

impl TtsConversationBuilder for MyConversationBuilder {
    type Conversation = MyConversation;

    fn with_speaker<S: Speaker>(self, speaker: S) -> Self { /* ... */ }
    fn language(self, lang: Language) -> Self { /* ... */ }

    fn synthesize<F, R>(self, matcher: F) -> impl Future<Output = R> + Send
    where F: FnOnce(Result<Self::Conversation, VoiceError>) -> R + Send + 'static
    {
        async move {
            // Perform synthesis and call matcher with result
            let result = self.do_synthesis().await;
            matcher(result)
        }
    }
}

// 4. Implement the conversation object
impl TtsConversation for MyConversation {
    type AudioStream = impl Stream<Item = i16> + Send + Unpin;

    fn into_stream(self) -> Self::AudioStream {
        // Convert to audio stream
    }
}
```

## 📚 Advanced Usage

### Error Handling with Graceful Fallbacks

```rust
let audio = primary_engine.conversation()
    .with_speaker(speaker)
    .synthesize(|conversation| {
        match conversation {
            Ok(conv) => Ok(conv.into_stream()),
            Err(primary_error) => {
                eprintln!("Primary engine failed: {}", primary_error);
                // Could try fallback engine here
                Err(primary_error)
            }
        }
    })
    .await
    .or_else(|_| {
        // Fallback to different engine or settings
        fallback_engine.conversation()
            .with_speaker(speaker)
            .synthesize(|conv| {
                Ok => conv.into_stream(),
                Err(e) => Err(e),
            })
    })?;
```

### Real-time Audio Processing

```rust
let mut audio_stream = engine.conversation()
    .with_speaker(speaker)
    .synthesize(|conv| Ok => conv.into_stream(), Err(e) => Err(e))
    .await?;

// Apply real-time effects
while let Some(sample) = audio_stream.next().await {
    let processed_sample = apply_effects(sample);
    audio_output.play(processed_sample)?;
}
```

### Batch Transcript Processing

```rust
let mut transcript_stream = engine.conversation()
    .with_source(SpeechSource::from_file("meeting.wav", AudioFormat::Pcm16Khz))
    .diarization(Diarization::On)
    .listen(|conv| Ok => conv.into_stream(), Err(e) => Err(e))
    .await?;

// Collect and format transcript
let mut segments = Vec::new();
while let Some(result) = transcript_stream.next().await {
    if let Ok(segment) = result {
        segments.push(segment);
    }
}

// Generate formatted output
generate_transcript_document(segments)?;
```

## 🧪 Testing

Run the test suite:

```bash
cargo test
```

Run examples:

```bash
cargo run --example api_usage
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

1. Clone the repository
2. Install Rust (latest stable)
3. Run tests: `cargo test`
4. Run examples: `cargo run --example api_usage`

### Code Style

- Follow Rust conventions and `cargo fmt`
- Add tests for new functionality
- Document public APIs with examples
- Use `cargo clippy` for linting

## 📄 License

This project is licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## 🙋 Support

- 📖 [Documentation](https://docs.rs/fluent_voice)
- 🐛 [Issue Tracker](https://github.com/your-org/fluent_voice/issues)
- 💬 [Discussions](https://github.com/your-org/fluent_voice/discussions)

---

Made with ❤️ for the Rust community
