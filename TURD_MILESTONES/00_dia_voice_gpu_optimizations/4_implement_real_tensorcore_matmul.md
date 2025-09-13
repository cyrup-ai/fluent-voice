# Implement Real TensorCore Matrix Multiplication

## Current Issue
The `matmul_optimized()` function claims device-specific kernels and TensorCore operations but just calls standard `a.matmul(&b)`.

## Current Code (Lines 82-94)
```rust
Device::Cuda(_) if a.dtype() == DType::BF16 || a.dtype() == DType::F16 => {
    // Would use TensorCore operations for half precision
    // For now, fall back to standard implementation
    let a = if a.is_contiguous() { a.clone() } else { a.contiguous()? };
    let b = if b.is_contiguous() { b.clone() } else { b.contiguous()? };
    a.matmul(&b)  // STANDARD MATMUL, NOT TENSORCORE
}
```

## Required Implementation
1. **TensorCore GEMM operations** for F16/BF16 mixed precision
2. **CUDA cublasLt integration** with TensorCore algorithm selection
3. **Optimal tensor layout** for TensorCore memory patterns
4. **Batch processing optimization** for multiple small matrices
5. **Memory alignment enforcement** for TensorCore requirements

## Expected Performance
- 10-20x speedup for mixed precision operations on Ampere+ GPUs
- Optimal memory throughput with proper data layouts
- Reduced precision overhead with maintained accuracy

## Dependencies
- cuBLAS LT library integration
- CUDA TensorCore API access
- Mixed precision tensor management
- Memory layout optimization utilities

## Acceptance Criteria
- [ ] Real TensorCore utilization (verified with nsight profiler)
- [ ] Significant speedup for F16/BF16 operations
- [ ] Proper memory alignment and layout optimization
- [ ] Fallback handling for non-TensorCore capable devices