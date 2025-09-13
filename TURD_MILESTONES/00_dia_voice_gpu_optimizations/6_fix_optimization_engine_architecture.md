# Fix OptimizationEngine Architecture

## Current Issue
The `layer_norm_optimized()` function creates a new `OptimizationEngine` on every call, defeating memory pooling benefits.

## Current Code (Lines 510-511)
```rust
pub fn layer_norm_optimized(x: &Tensor, weight: &Tensor, bias: &Tensor, _eps: f64) -> CandleResult<Tensor> {
    // Create optimization engine
    let mut engine = OptimizationEngine::new(x.device().clone(), x.dtype());  // NEW ENGINE EVERY CALL
    
    // Configure optimization...
    let normalized = engine.optimize_tensor_operations(x, &config)?;  // POOL DISCARDED
    // ...
}
```

## Architectural Problems
1. **Memory pool waste**: Pool built up then immediately discarded
2. **Initialization overhead**: Engine creation cost on every operation  
3. **Device resource waste**: GPU contexts may be recreated unnecessarily
4. **Cache miss penalties**: No reuse of compiled kernels or resources

## Required Architecture Redesign

### Global Engine Management
```rust
// Thread-local or global engine pool
thread_local! {
    static OPTIMIZATION_ENGINES: RefCell<HashMap<(Device, DType), OptimizationEngine>> = Default::default();
}

pub fn layer_norm_optimized(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f64) -> CandleResult<Tensor> {
    OPTIMIZATION_ENGINES.with(|engines| {
        let mut engines = engines.borrow_mut();
        let key = (x.device().clone(), x.dtype());
        let engine = engines.entry(key).or_insert_with(|| OptimizationEngine::new(x.device().clone(), x.dtype()));
        
        // Use persistent engine with accumulated benefits
        engine.optimize_layer_norm(x, weight, bias, eps)
    })
}
```

### Alternative: Builder Pattern
```rust
pub struct OptimizationEngineBuilder {
    engines: HashMap<(Device, DType), OptimizationEngine>,
}

impl OptimizationEngineBuilder {
    pub fn layer_norm(&mut self, x: &Tensor, weight: &Tensor, bias: &Tensor) -> CandleResult<Tensor> {
        let engine = self.get_or_create_engine(x.device(), x.dtype());
        engine.optimize_layer_norm(x, weight, bias, 1e-5)
    }
}
```

## Expected Benefits
- **Persistent memory pools**: Accumulated tensor caching across calls
- **Kernel compilation reuse**: GPU kernels compiled once and cached
- **Resource efficiency**: Device contexts maintained across operations
- **Performance scaling**: Benefits increase with usage frequency

## Dependencies
- Thread-safe engine management
- Device context lifetime management  
- Memory pool persistence strategies

## Acceptance Criteria
- [ ] OptimizationEngine reused across multiple calls
- [ ] Memory pool accumulates tensors over time
- [ ] Device resources properly managed and persistent
- [ ] Thread safety for concurrent access
- [ ] Performance improves with repeated usage