# Building Dia-Voice with GPU Acceleration

This document explains how to build and run dia-voice with GPU acceleration using either Metal (macOS) or CUDA (Linux/Windows).

## Prerequisites

### For Metal (macOS)
- macOS 10.15 or newer
- Xcode Command Line Tools
- Metal-compatible GPU

### For CUDA (Linux/Windows)
- CUDA toolkit version 11.0 or newer
- CUDA-compatible GPU
- Appropriate GPU drivers

## Building with GPU Acceleration

### macOS (Metal)

```bash
# Build with Metal support (debug build)
cargo build --features metal

# Build with Metal support (release build, recommended)
cargo build --release --features metal

# Run with Metal support
cargo run --release --features metal -- --prompt "Hello, world!"
```

### Linux/Windows (CUDA)

```bash
# Build with CUDA support (debug build)
cargo build --features cuda

# Build with CUDA support (release build, recommended)
cargo build --release --features cuda

# Add cudnn for additional performance
cargo build --release --features cudnn

# Run with CUDA support
cargo run --release --features cuda -- --prompt "Hello, world!"
```

### Fallback to CPU

If you don't have a compatible GPU or want to force CPU execution:

```bash
# Build without GPU features but with native CPU optimizations
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Run with forced CPU execution
cargo run --release -- --prompt "Hello, world!" --cpu
```

## Troubleshooting

### Metal Issues

1. **"Metal function not found" errors**:
   - This typically means the Metal library wasn't compiled correctly. Try cleaning your build:
   ```bash
   cargo clean
   cargo build --release --features metal
   ```

2. **Metal shader compilation failures**:
   - Check that you're using a recent macOS version
   - Ensure XCode command line tools are installed: `xcode-select --install`

### CUDA Issues

1. **"CUDA library not found" errors**:
   - Ensure CUDA toolkit is installed and in your PATH
   - For Windows, verify that CUDA DLLs are in the system PATH

2. **cuBLAS/cuDNN errors**:
   - These libraries need to be installed separately on some systems
   - Make sure they match your CUDA version

3. **Compute capability errors**:
   - Adjust the CUDA_COMPUTE_CAP value in .cargo/config.toml to match your GPU
   - Common values: "52" (Maxwell), "61" (Pascal), "75" (Turing), "86" (Ampere)

## Performance Notes

- For optimal performance, always use `--release` mode
- On macOS, Metal performance is typically best on Apple Silicon (M1/M2/M3) chips
- For CUDA, using cuDNN can significantly improve performance
- If you experience out-of-memory errors, try reducing batch sizes in your configuration

## UI Components (Optional)

To build with UI components:

```bash
# Build with UI components and Metal
cargo build --release --features "ui metal"

# Build with UI components and CUDA
cargo build --release --features "ui cuda"
```