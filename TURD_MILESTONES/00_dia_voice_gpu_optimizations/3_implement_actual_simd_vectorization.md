# Implement Actual SIMD Vectorization

## Current Issue
The `simd_layer_norm()` method claims vectorization but uses basic scalar operations (`data.iter().sum()`, `data.iter().map()`).

## Current Code (Lines 405-425)
```rust
fn simd_layer_norm(&self, data: &[f32]) -> Result<Vec<f32>, candle_core::Error> {
    let mut result = Vec::with_capacity(data.len());
    
    // Phase 1: Mean calculation (optimized for cache locality)
    let sum: f32 = data.iter().sum();  // SCALAR, NOT VECTORIZED
    let mean = sum / data.len() as f32;
    
    // Phase 2: Variance calculation
    let sum_sq_diff: f32 = data.iter().map(|&x| (x - mean).powi(2)).sum();  // SCALAR
    let variance = sum_sq_diff / data.len() as f32;
    
    // Phase 3: Normalization
    let inv_std = 1.0 / (variance + 1e-5).sqrt();
    
    for &value in data {  // SCALAR LOOP
        let normalized = (value - mean) * inv_std;
        result.push(normalized);
    }
    
    Ok(result)
}
```

## Required Implementation
1. **Real SIMD operations** using stable Rust SIMD or platform intrinsics
2. **Vectorized mean calculation** processing 8 f32 values simultaneously
3. **Vectorized variance computation** with SIMD parallel reductions
4. **Vectorized normalization** applying operations to SIMD lanes
5. **Remainder handling** for non-SIMD-aligned data

## Expected Performance
- 4-8x speedup on modern CPUs with AVX/AVX2 support
- Efficient SIMD lane utilization
- Cache-friendly memory access patterns

## Dependencies
- Platform-specific SIMD intrinsics (AVX2, NEON)
- Stable Rust portable SIMD (when available)
- Proper SIMD alignment considerations

## Acceptance Criteria
- [ ] Real SIMD instructions (verified with assembly inspection)
- [ ] Vectorized operations for all phases (mean, variance, normalization)
- [ ] Performance benchmarks showing actual speedup
- [ ] Proper handling of non-aligned data lengths