# CyTerm - Voice-Enabled Terminal Emulator

A software-rendered terminal emulator with integrated voice capabilities using Ratatui, Silero VAD, and Candle Whisper ASR.

## Features

- **Software-rendered terminal**: Custom Ratatui backend using cosmic-text for high-quality text rendering
- **Voice Activity Detection**: Real-time speech detection using Silero VAD model
- **Wake word detection**: Keyword-based activation system  
- **Speech recognition**: Whisper-based ASR using Candle framework
- **Pure Rust**: No heavy dependencies on external libraries

## Current Status

⚠️ **This project is in early development and has several critical issues that need to be resolved before it can compile and run.**

### Critical Issues

1. **Missing ONNX Model**: The Silero VAD model file is not included
   - Download from: https://github.com/snakers4/silero-vad
   - Place in: `onnx/silero_vad.onnx`

2. **Incomplete ASR Implementation**: The Whisper integration needs proper API calls
3. **Missing Audio Pipeline**: No microphone capture or real-time processing loop
4. **No Main Application**: Missing entry point that connects all components

## Setup Instructions

### Prerequisites

- Rust 1.70+
- Download Silero VAD model (see issue #1 above)

### Building

```bash
# This will currently fail due to missing ONNX model
cargo build
```

### Dependencies

- `ratatui`: Terminal UI framework
- `cosmic-text`: High-quality text rendering
- `ort`: ONNX Runtime for VAD
- `candle`: Machine learning framework for Whisper
- `cpal`: Audio capture
- `rubato`: Audio resampling

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Microphone    │───▶│   VAD + Wake    │───▶│   Whisper ASR   │
│    (CPAL)       │    │   Word Detection│    │    (Candle)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐
│   Terminal UI   │◀───│   Command       │
│   (Ratatui)     │    │   Processing    │
└─────────────────┘    └─────────────────┘
```

## Contributing

This project needs significant work to become functional. Key areas needing attention:

1. Fix ONNX model integration
2. Complete Whisper ASR implementation
3. Add microphone capture pipeline
4. Create main application loop
5. Add proper error handling
6. Performance optimization

## License

MIT OR Apache-2.0
