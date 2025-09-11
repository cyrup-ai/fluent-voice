# GPU Support in Dia Voice

## Current Status

The Dia Voice crate maintains full support for both CUDA and Metal GPU acceleration through feature flags. The GPU support has NOT been removed - it's still fully present in the codebase and can be enabled via Cargo features.

## Feature Flags

### CUDA Support
```toml
# In Cargo.toml
[features]
cuda = ["candle-core/cuda", "candle-nn/cuda", "candle-transformers/cuda"]
cudnn = ["cuda", "candle-core/cudnn"]  # Enhanced CUDA with cuDNN
```

To build with CUDA support:
```bash
cargo build --features cuda
# or with cuDNN for additional optimizations:
cargo build --features cudnn
```

### Metal Support (macOS)
```toml
# In Cargo.toml
[features]
metal = ["candle-core/metal", "candle-nn/metal", "candle-transformers/metal"]
```

To build with Metal support:
```bash
cargo build --features metal
```

## Implementation Details

### GPU Configuration (`src/optimizations.rs`)

The crate provides device-specific optimizations through conditional compilation:

#### 1. Optimal Configuration Selection
```rust
#[cfg(feature = "cuda")]
Device::Cuda(_) => GpuConfig {
    mixed_precision: true,      // Use BF16/F16 for better performance
    optimal_batch_size: 8,      // CUDA can handle larger batches
    memory_pooling: true,       // Enable memory reuse
    graph_capture: true,        // CUDA graph optimization
},

#[cfg(feature = "metal")]
Device::Metal(_) => GpuConfig {
    mixed_precision: true,      // Use F16 for Metal
    optimal_batch_size: 4,      // Metal optimal batch size
    memory_pooling: true,       // Enable memory reuse
    graph_capture: false,       // Not supported on Metal
},
```

#### 2. Data Type Selection
```rust
#[cfg(feature = "cuda")]
Device::Cuda(_) => DType::BF16,  // BFloat16 for modern NVIDIA GPUs

#[cfg(feature = "metal")]
Device::Metal(_) => DType::F16,  // Float16 for Metal Performance Shaders
```

#### 3. Optimized Operations

The crate includes placeholders for GPU-optimized operations:

- **Matrix Multiplication**: Device-specific kernels for `matmul_optimized()`
- **Attention Mechanisms**: GPU-optimized attention computation
- **Channel Delay**: GPU kernels for audio processing operations

### Current Implementation State

While the infrastructure for GPU support is fully present, the actual GPU-specific kernel implementations are marked with comments like:
```rust
// Would use TensorCore operations for half precision
// For now, fall back to standard implementation
```

This indicates that:
1. The GPU feature flags work correctly
2. The conditional compilation paths are in place
3. The actual low-level GPU kernels are using Candle's default GPU implementations
4. Custom GPU kernels can be added without changing the API

## Usage Examples

### Device Detection and Configuration
```rust
use dia_voice::optimizations::{get_optimal_config, get_compute_dtype};
use candle_core::Device;

// Automatically detect and configure for available GPU
let device = Device::cuda_if_available(0)?;
let config = get_optimal_config(&device);
let dtype = get_compute_dtype(&device, config.mixed_precision);

println!("Using device: {:?}", device);
println!("Batch size: {}", config.optimal_batch_size);
println!("Data type: {:?}", dtype);
```

### Building for Different Targets

```bash
# CPU only (default)
cargo build --release

# NVIDIA GPU with CUDA
cargo build --release --features cuda

# NVIDIA GPU with cuDNN (requires cuDNN installation)
cargo build --release --features cudnn

# Apple Silicon with Metal
cargo build --release --features metal

# Multiple features
cargo build --release --features "cuda ui"
```

## Performance Considerations

1. **CUDA (NVIDIA GPUs)**:
   - Best performance with RTX 30/40 series (Ampere/Ada architecture)
   - BF16 support provides 2x throughput on supported hardware
   - Graph capture can significantly reduce kernel launch overhead

2. **Metal (Apple Silicon)**:
   - Optimized for M1/M2/M3 chips
   - F16 provides optimal performance/precision trade-off
   - Unified memory architecture reduces data transfer overhead

3. **Memory Pooling**:
   - Both CUDA and Metal configurations enable memory pooling
   - Reduces allocation overhead in tight loops
   - Particularly beneficial for streaming inference

## Future Enhancements

The codebase is structured to easily add:
- Custom CUDA kernels using cuBLAS/cuDNN directly
- Metal Performance Shaders for specific operations
- Optimized attention kernels (Flash Attention, etc.)
- Quantized inference support (INT8/INT4)

The modular design in `optimizations.rs` ensures that GPU-specific optimizations can be added incrementally without breaking existing functionality.