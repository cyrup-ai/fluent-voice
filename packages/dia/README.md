# Dia Voice on Candle

A high-performance Rust implementation of the Dia Voice text-to-speech model with GPU acceleration and optimized audio processing.

## 󰓎 Features

- 󰍛 **GPU Acceleration**: Optimized for NVIDIA CUDA and Apple Metal
- 󰘏 **Multi-Channel Audio**: 9-channel EnCodec with temporal delay synchronization
- 󰕾 **Professional Audio**: BS.1770 loudness normalization and high-quality resampling
- 󰘚 **Efficient Memory**: Zero-copy tensor operations and memory pooling
- 󰓎 **Mixed Precision**: BF16 (CUDA) and F16 (Metal) support for faster inference
- 󰑋 **Voice Cloning**: Support for audio prompts and speaker embeddings

## 󰅸 Quick Start

### 󰅍 Prerequisites

- Rust 1.75 or later
- (Optional) CUDA Toolkit 11.8+ for NVIDIA GPU support
- (Optional) macOS 12.0+ for Metal support

### 󰢻 Building

```bash
# CPU-only build
cargo build --release

# CUDA GPU support
cargo build --release --features "cuda cudnn"

# Apple Metal support
cargo build --release --features metal

# With UI components
cargo build --release --features "ui metal"
```

### 󰆍 Running

```bash
# Basic usage
cargo run --features metal --prompt "Hello, world!"

# With GPU acceleration (auto-detected)
cargo run --features cuda -- --prompt "Hello from GPU!"

# With audio prompt for voice cloning
cargo run  --features metal -- --prompt "Hello!" --prompt-wav voice_sample.wav

# Save output to file
cargo run --features accelerate -- --prompt "Save this audio" --out output.wav
```

## 󰆧 Architecture

### 󰖩 Channel Delay System

Dia Voice uses a sophisticated channel delay mechanism for multi-channel audio synchronization:

```rust
// Delay pattern: [0, 8, 9, 10, 11, 12, 13, 14, 15]
// Channel 0: no delay
// Channel 1: 8-frame delay
// Channel 2: 9-frame delay, etc.
```

This is implemented using zero-copy tensor views for optimal performance:
- `delayed_view()`: Applied before model input
- `undelayed_view()`: Applied before audio decoding

### 󰍛 GPU Optimizations

When GPU features are enabled, the following optimizations are automatically applied:

1. **Mixed Precision Computation**
   - CUDA: BF16 for Ampere+ GPUs
   - Metal: F16 for Apple Silicon

2. **Optimized Kernels**
   - Flash Attention (when available)
   - TensorCore acceleration
   - Custom channel delay kernels

3. **Memory Management**
   - Tensor memory pooling
   - Contiguous memory layouts
   - Pinned memory transfers (CUDA)

### 󰘏 Audio Pipeline

```
Input Text → Tokenization → Encoder → Decoder → Channel Delays → EnCodec → PCM → Loudness Norm → WAV
```

## 󰢻 Configuration

The model uses a hierarchical configuration system:

```rust
DiaConfig {
    model: ModelConfig {
        encoder: EncoderConfig { ... },
        decoder: DecoderConfig { ... },
        // Rotary embeddings, vocab sizes, etc.
    },
    data: DataConfig {
        channels: 9,
        audio_length: 3072,
        delay_pattern: [0, 8, 9, 10, 11, 12, 13, 14, 15],
        // Token values and padding
    }
}
```

## 󰌣 Performance Benchmarks

| Device | Batch Size | Tokens/sec | Memory Usage |
|--------|------------|------------|--------------|
| CPU (M2) | 1 | ~50 | 2GB |
| Metal (M2 Pro) | 4 | ~200 | 3GB |
| CUDA (RTX 4090) | 8 | ~500 | 4GB |

*Benchmarks are approximate and depend on sequence length*

## 󰘨 Advanced Usage

### 󰑋 Custom Voice Generation

```rust
use dia_voice::{VoicePool, Conversation, Speaker};

// Create a voice pool
let pool = VoicePool::new(device)?;

// Create a conversation with a specific speaker
let mut conv = Conversation::new(
    "Hello, this is a custom voice!".to_string(),
    CustomSpeaker::new(),
    pool.clone()
)?;

// Generate audio
let audio = conv.generate().await?;
```

### 󰅭 Batch Processing

```rust
// Use optimal batch size for your GPU
let batch_size = match device {
    Device::Cuda(_) => 8,
    Device::Metal(_) => 4,
    _ => 1,
};
```

## 󰘨 Development

### 󰙅 Project Structure

```
src/
├── audio/          # Audio processing (loudness, resampling, delays)
├── voice/          # High-level voice synthesis APIs
├── layers.rs       # Neural network layers
├── model.rs        # Dia model architecture
├── generation.rs   # Autoregressive generation
├── optimizations.rs # GPU optimization utilities
└── main.rs         # CLI interface
```

### 󰙨 Testing

```bash
# Run all tests
cargo test

# Test specific module
cargo test audio::channel_delay::tests

# Benchmark performance
cargo bench
```

### 󰊤 Contributing

1. Check existing issues and discussions
2. Fork the repository
3. Create a feature branch
4. Ensure all tests pass
5. Submit a pull request

## 󰃤 Troubleshooting

### 󰚽 CUDA Issues
- Ensure CUDA toolkit is installed and in PATH
- Set `CUDA_COMPUTE_CAP` for your GPU (e.g., "89" for RTX 4090)
- Check cuDNN installation matches CUDA version

### 󰚽 Metal Issues
- Requires macOS 12.0 or later
- Install Xcode command line tools: `xcode-select --install`
- For best performance, use Apple Silicon Macs

### 󰘏 Audio Quality
- Input audio should be 24kHz mono for best results
- Use `--temperature` and `--top-p` to control generation diversity
- Higher `--cfg-scale` values produce more coherent speech

## 󰗦 License

This implementation is based on the Dia model architecture. Please refer to the original model's license for usage terms.

## 󰓢 Acknowledgments

- Original Dia model by [Authors]
- Candle framework by Hugging Face
- EnCodec implementation
- Community contributors

## 󰗠 Citation

If you use this implementation in your research, please cite:

```bibtex
@software{dia_voice_rust,
  title = {Dia Voice - High Performance Rust Implementation},
  year = {2024},
  url = {https://github.com/yourusername/dia-voice}
}
```
