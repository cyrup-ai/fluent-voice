# Fluent Voice Animator

A powerful real-time audio visualization and LiveKit room management tool built in Rust. The `anima` binary provides two distinct modes: a lightweight terminal-based oscilloscope for live audio visualization and a full-featured GUI for LiveKit room visualization with participant management.

## Features

### 🎛️ Terminal Mode (Oscilloscope)

- **Real-time Audio Visualization**: Live waveform, spectrum, and vector scope displays
- **Multiple Display Modes**: Oscilloscope, spectrogram, and vector scope views
- **Cross-platform Audio**: CPAL-based audio input with device selection
- **Interactive Controls**: Real-time parameter adjustment and display switching
- **High Performance**: GPU-accelerated processing with Candle ML framework

### 🎥 Room Mode (LiveKit GUI)

- **LiveKit Integration**: Full-featured LiveKit room client with WebRTC support
- **Multi-participant Support**: Real-time audio/video for multiple participants
- **Audio Visualization**: Per-participant waveform displays with volume control
- **Video Rendering**: Hardware-accelerated video display with WGPU
- **Connection Management**: Automatic reconnection with exponential backoff
- **Individual Controls**: Per-participant volume, mute, and quality monitoring

## Installation

```bash
cargo install --path .
```

## Usage

### Terminal Oscilloscope Mode

```bash
# Basic audio visualization
anima terminal audio

# Specify audio device
anima terminal audio --device "Built-in Microphone"

# List available audio devices
anima terminal audio --list

# Customize display parameters
anima terminal audio --scale 0.8 --scatter --no-reference
```

**Terminal Controls:**

- `q` - Quit
- `m` - Switch display modes (oscilloscope/spectrogram/vector)
- `↑/↓` - Adjust scale
- `←/→` - Adjust parameters
- `space` - Pause/resume

### LiveKit Room Mode

The animator connects to LiveKit rooms for real-time audio/video visualization. **Rooms are automatically created by the LiveKit server when the first participant joins.**

```bash
# Connect to LiveKit room (creates room if it doesn't exist)
anima room \
  --url wss://your-livekit-server.com \
  --api-key your_api_key \
  --api-secret your_api_secret \
  --room-name demo-room \
  --participant-name "Audio Visualizer"

# Auto-connect using environment variables
export LIVEKIT_URL="wss://your-livekit-server.com"
export LIVEKIT_API_KEY="your_api_key"
export LIVEKIT_API_SECRET="your_api_secret"
anima room --auto-connect

# Join specific room (other participants use the same room-name)
anima room --room-name "meeting-123" --participant-name "John Doe"
```

**How to share the room with others:**

1. **In the GUI**: Once connected, the app displays a "📋 Copy Command" button with the exact command others can run
2. **Room Name**: Other participants join using the same `--room-name`
3. **LiveKit Server**: All participants must connect to the same `LIVEKIT_URL`
4. **Credentials**: All participants need valid API credentials for the same LiveKit server

**Sharing Example:**
When you connect to a room, the GUI shows:

```bash
anima room --url "wss://your-server.com" --room-name "meeting-123" --participant-name "Your Name"
```

Others can copy this command, change the participant name, and join instantly.

## Environment Variables

The room mode supports the following environment variables:

- `LIVEKIT_URL` - LiveKit server WebSocket URL
- `LIVEKIT_API_KEY` - LiveKit API key for authentication
- `LIVEKIT_API_SECRET` - LiveKit API secret for token generation

## Configuration

### Terminal Mode Options

| Option | Description | Default |
|--------|-------------|---------|
| `--scale` | Vertical scale factor (0.0-1.0) | 1.0 |
| `--scatter` | Use scatter plot mode | false |
| `--no-reference` | Hide reference lines | false |
| `--no-ui` | Hide UI elements | false |
| `--no-braille` | Disable braille line drawing | false |

### Room Mode Options

| Option | Description | Default |
|--------|-------------|---------|
| `--url` | LiveKit server URL | From env var |
| `--api-key` | LiveKit API key | From env var |
| `--api-secret` | LiveKit API secret | From env var |
| `--room-name` | Room to join | "fluent-voice-demo" |
| `--participant-name` | Display name | "Rust Visualizer" |
| `--auto-connect` | Connect immediately | false |

## Examples

### Development Setup

```bash
# Terminal mode for audio debugging
anima terminal audio --device "USB Audio Device" --scale 0.5

# LiveKit room for testing
export LIVEKIT_URL="wss://localhost:7880"
export LIVEKIT_API_KEY="devkey"
export LIVEKIT_API_SECRET="secret"
anima room --room-name "test-room" --auto-connect
```

### Production Deployment

```bash
# Set production LiveKit credentials
export LIVEKIT_URL="wss://livekit.yourcompany.com"
export LIVEKIT_API_KEY="prod_api_key_here"
export LIVEKIT_API_SECRET="prod_secret_here"

# Launch room visualizer
anima room \
  --room-name "conference-room-1" \
  --participant-name "Audio Monitor" \
  --auto-connect
```

## Architecture

### Terminal Mode

- **Audio Input**: CPAL cross-platform audio capture
- **Processing**: Real-time FFT and signal processing
- **Display**: Ratatui terminal UI with braille graphics
- **Performance**: GPU acceleration via Candle ML

### Room Mode  

- **WebRTC**: LiveKit Rust SDK for real-time communication
- **Audio**: Per-participant audio visualization and playback
- **Video**: WGPU-accelerated video rendering
- **UI**: egui immediate mode GUI
- **Networking**: Automatic reconnection and error recovery

## Requirements

- **Rust**: 1.70+ with 2024 edition
- **Audio**: System audio drivers (ALSA/PulseAudio on Linux, CoreAudio on macOS, WASAPI on Windows)
- **Graphics**: OpenGL 3.3+ or Vulkan for GUI mode
- **Network**: WebSocket and UDP connectivity for LiveKit

## Troubleshooting

### Audio Issues

```bash
# List available devices
anima terminal audio --list

# Test specific device
anima terminal audio --device "Your Device Name"
```

### LiveKit Connection Issues

```bash
# Verify environment variables
echo $LIVEKIT_URL
echo $LIVEKIT_API_KEY

# Test connection without auto-connect
anima room --room-name "test"
```

### Performance Issues

- Enable hardware acceleration features during build
- Use `--release` mode for production
- Check GPU drivers for GUI mode

## License

Apache-2.0

## Contributing

This is part of the fluent-voice ecosystem. See the main repository for contribution guidelines.
