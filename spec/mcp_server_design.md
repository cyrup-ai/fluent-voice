# Fluent Voice MCP Server Design Specification

## Overview

This document specifies the design for a Model Context Protocol (MCP) server that exposes the fluent-voice library's TTS and STT capabilities to AI assistants like Claude Desktop. The server follows the patterns established in the MCP specification and the Taskwarrior integration example.

## Architecture

### Core Components

1. **MCP Server** (`fluent_voice_mcp`)
   - Built using `mcp-sdk-rs` for Rust
   - Exposes TTS and STT functionality as MCP tools
   - Uses stdio transport for Claude Desktop integration
   - Implements proper error handling and JSON-RPC compliance

2. **Tool Categories**
   - **TTS Tools**: Text synthesis, voice configuration, audio streaming
   - **STT Tools**: Speech recognition, live transcription, audio file processing
   - **Management Tools**: Voice listing, configuration management, session control

### Design Principles

1. **Stateless Operations**: Each tool call is independent, avoiding session management complexity
2. **Type Safety**: Leverage Rust's type system with Serde for JSON schema validation
3. **Error Handling**: Graceful failures with meaningful error messages
4. **Security**: Sanitize inputs, prevent injection attacks, validate file paths
5. **Performance**: Efficient streaming for audio data, minimal overhead

## Tool Definitions

### TTS Tools

#### 1. `synthesize_text`
Converts text to speech with specified voice and settings.

```rust
#[derive(Deserialize)]
struct SynthesizeParams {
    text: String,
    voice_id: Option<String>,
    speaker_name: Option<String>,
    speed: Option<f32>,      // 0.5 to 2.0
    pitch: Option<f32>,      // -1.0 to 1.0
    language: Option<String>, // ISO 639-1 code
    output_format: Option<AudioFormat>,
}

#[derive(Serialize)]
struct SynthesizeResult {
    audio_base64: String,    // Base64 encoded audio
    format: AudioFormat,
    duration_ms: u32,
    voice_used: String,
}
```

#### 2. `list_voices`
Returns available voices/speakers for TTS.

```rust
#[derive(Deserialize)]
struct ListVoicesParams {
    language: Option<String>,
    gender: Option<String>,
    engine: Option<String>,
}

#[derive(Serialize)]
struct Voice {
    id: String,
    name: String,
    language: String,
    gender: Option<String>,
    preview_url: Option<String>,
    engine: String,
}

#[derive(Serialize)]
struct ListVoicesResult {
    voices: Vec<Voice>,
}
```

#### 3. `stream_synthesis`
Initiates a streaming TTS session (returns a session ID for subsequent operations).

```rust
#[derive(Deserialize)]
struct StreamSynthesisParams {
    voice_id: String,
    language: String,
    initial_settings: Option<TtsSettings>,
}

#[derive(Serialize)]
struct StreamSession {
    session_id: String,
    websocket_url: Option<String>, // For future WebSocket support
    ready: bool,
}
```

### STT Tools

#### 4. `transcribe_audio`
Transcribes an audio file to text.

```rust
#[derive(Deserialize)]
struct TranscribeParams {
    audio_path: String,      // Path to audio file
    language: Option<String>,
    timestamps: Option<bool>,
    diarization: Option<bool>,
    vad_mode: Option<VadMode>,
}

#[derive(Serialize)]
struct TranscribeResult {
    text: String,
    segments: Option<Vec<TranscriptSegment>>,
    language_detected: Option<String>,
    confidence: f32,
}
```

#### 5. `start_live_transcription`
Starts live microphone transcription.

```rust
#[derive(Deserialize)]
struct LiveTranscriptionParams {
    microphone: Option<String>, // "default" or specific device
    language: Option<String>,
    vad_mode: Option<VadMode>,
    noise_reduction: Option<bool>,
}

#[derive(Serialize)]
struct LiveTranscriptionSession {
    session_id: String,
    status: String,
    microphone_used: String,
}
```

#### 6. `get_transcription_results`
Retrieves results from an active transcription session.

```rust
#[derive(Deserialize)]
struct GetResultsParams {
    session_id: String,
    since_timestamp: Option<u64>, // Unix timestamp in ms
}

#[derive(Serialize)]
struct TranscriptionUpdate {
    segments: Vec<TranscriptSegment>,
    is_final: bool,
    session_active: bool,
}
```

### Management Tools

#### 7. `configure_engine`
Sets default engine configuration.

```rust
#[derive(Deserialize)]
struct ConfigureEngineParams {
    engine_type: String, // "tts" or "stt"
    engine_name: String, // e.g., "elevenlabs", "whisper"
    api_key: Option<String>,
    base_url: Option<String>,
    default_settings: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct ConfigureResult {
    success: bool,
    message: String,
}
```

## Data Types

### Common Types

```rust
#[derive(Serialize, Deserialize)]
enum AudioFormat {
    Mp3,
    Wav,
    Ogg,
    Flac,
    Pcm16,
}

#[derive(Serialize, Deserialize)]
enum VadMode {
    Disabled,
    Fast,
    Accurate,
}

#[derive(Serialize, Deserialize)]
struct TranscriptSegment {
    text: String,
    start_ms: Option<u32>,
    end_ms: Option<u32>,
    confidence: Option<f32>,
    speaker: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct TtsSettings {
    stability: Option<f32>,
    similarity_boost: Option<f32>,
    style_exaggeration: Option<f32>,
    speaker_boost: Option<bool>,
}
```

## Implementation Structure

### Server Main (`src/main.rs`)

```rust
use mcp_sdk_rs::server::{Server, ToolDescription};
use mcp_sdk_rs::transport::stdio::StdioServerTransport;
use fluent_voice::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut server = Server::new("fluent-voice", "0.1.0")?;
    server.set_description("MCP server for TTS and STT using fluent-voice API");

    // Register TTS tools
    server.add_tool(
        ToolDescription::new("synthesize_text")
            .with_description("Convert text to speech")
            .with_input_schema::<SynthesizeParams>(),
        synthesize_text_handler,
    );

    // Register STT tools
    server.add_tool(
        ToolDescription::new("transcribe_audio")
            .with_description("Transcribe audio file to text")
            .with_input_schema::<TranscribeParams>(),
        transcribe_audio_handler,
    );

    // ... register other tools

    // Start server
    let transport = StdioServerTransport::new();
    server.start(transport)?;
    Ok(())
}
```

### Tool Handlers (`src/handlers/`)

Each tool gets its own handler module:
- `src/handlers/tts.rs` - TTS tool implementations
- `src/handlers/stt.rs` - STT tool implementations  
- `src/handlers/management.rs` - Configuration tools

Example handler:

```rust
// src/handlers/tts.rs
pub async fn synthesize_text_handler(params: SynthesizeParams) -> Result<SynthesizeResult, MCPError> {
    // Validate parameters
    let speed = params.speed.unwrap_or(1.0).clamp(0.5, 2.0);
    
    // Build TTS request using fluent-voice
    let audio = FluentVoice::tts()
        .with_speaker(
            Speaker::named(params.speaker_name.as_deref().unwrap_or("default"))
                .with_speed_modifier(VocalSpeedMod(speed))
                .speak(&params.text)
                .build()
        )
        .synthesize(|conversation| {
            conversation.into_stream()
        })
        .await
        .map_err(|e| MCPError::new(&format!("TTS failed: {}", e)))?;

    // Convert audio to base64
    let audio_bytes = audio.collect_bytes().await?;
    let audio_base64 = base64::encode(&audio_bytes);

    Ok(SynthesizeResult {
        audio_base64,
        format: params.output_format.unwrap_or(AudioFormat::Mp3),
        duration_ms: calculate_duration(&audio_bytes),
        voice_used: params.voice_id.unwrap_or_else(|| "default".to_string()),
    })
}
```

## Claude Desktop Integration

### Configuration File

Users add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "fluent-voice": {
      "command": "/path/to/fluent_voice_mcp",
      "args": [],
      "env": {
        "ELEVENLABS_API_KEY": "your-api-key",
        "FLUENT_VOICE_CONFIG": "/path/to/config.toml"
      }
    }
  }
}
```

### Usage Examples

Once integrated, users can interact naturally:

1. **Text to Speech**:
   - "Read this text aloud: 'Hello, this is a test'"
   - "Use a British female voice to say 'Good morning'"
   - "Generate speech for this paragraph with slow speed"

2. **Speech to Text**:
   - "Transcribe the audio file at /path/to/recording.wav"
   - "Start listening to my microphone and transcribe what I say"
   - "What did the speaker say in this audio?"

3. **Voice Management**:
   - "List all available voices"
   - "Show me female voices that speak Spanish"
   - "Configure ElevenLabs as my TTS engine"

## Security Considerations

1. **File Access**: Validate all file paths, restrict to allowed directories
2. **API Keys**: Store securely, never log or expose in responses
3. **Input Validation**: Sanitize text inputs, validate audio formats
4. **Resource Limits**: Cap audio duration, implement timeouts
5. **Process Isolation**: Run synthesis/recognition in isolated processes

## Error Handling

All errors return structured MCP errors:

```rust
#[derive(Serialize)]
struct MCPError {
    code: i32,
    message: String,
    data: Option<serde_json::Value>,
}

// Common error codes
const ERR_INVALID_INPUT: i32 = -32602;
const ERR_INTERNAL: i32 = -32603;
const ERR_RESOURCE_NOT_FOUND: i32 = -32001;
const ERR_ENGINE_NOT_CONFIGURED: i32 = -32002;
```

## Testing Strategy

1. **Unit Tests**: Test each handler with mock fluent-voice responses
2. **Integration Tests**: Test full MCP message flow
3. **E2E Tests**: Test with actual Claude Desktop connection
4. **Performance Tests**: Measure latency and memory usage
5. **Security Tests**: Validate input sanitization and access controls

## Future Enhancements

1. **WebSocket Transport**: For real-time audio streaming
2. **Multi-Engine Support**: Switch between different TTS/STT providers
3. **Audio Effects**: Pitch shifting, speed adjustment post-synthesis
4. **Language Detection**: Auto-detect language for STT
5. **Voice Cloning**: Support for custom voice creation
6. **Batch Operations**: Process multiple texts/files efficiently

## References

- [Model Context Protocol Specification](https://modelcontextprotocol.io/)
- [MCP Rust SDK](https://github.com/anthropics/mcp-sdk-rs)
- [Fluent Voice API Documentation](../fluent-voice/README.md)
- [Claude Desktop Integration Guide](https://modelcontextprotocol.io/quickstart/user)