# TODO8: Replace Conv Placeholder Logic with Learned Behavior

## Issue
**File**: `packages/kyutai/src/conv.rs`
**Lines**: 62, 119

## Problem
Placeholder logic instead of sophisticated learned behavior:

```rust
// This is a placeholder for more sophisticated learnt behavior
```

## Required Fix
1. Implement proper convolutional neural network behavior
2. Replace hardcoded logic with learned parameters
3. Add proper weight initialization and forward pass
4. Integrate with training pipeline

## Implementation Steps
1. Identify what "sophisticated learnt behavior" should be
2. Add proper CNN layer implementations
3. Replace placeholder logic with actual neural network computations
4. Add parameter loading from trained models
5. Implement proper forward pass with learned weights
6. Add gradient computation if training is needed

## Context
This appears to be in convolutional layers where actual neural network computation should happen instead of placeholder logic.

## Priority
Medium - Neural network computation accuracy