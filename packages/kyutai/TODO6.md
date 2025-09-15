# TODO6: Implement Top-K Sampling in Model

## Issue
**File**: `packages/kyutai/src/model.rs`
**Line**: 509

## Problem
Missing top-k sampling implementation:

```rust
// TODO: Implement proper top-k sampling when Candle API is available
```

## Required Fix
1. Implement top-k sampling algorithm
2. Handle probability distribution manipulation
3. Add temperature scaling and nucleus (top-p) sampling
4. Integrate with existing generation pipeline

## Implementation Steps
1. Add top-k logit filtering before softmax
2. Implement temperature scaling
3. Add nucleus (top-p) sampling option
4. Handle edge cases (k=1, k > vocab_size)
5. Add sampling configuration parameters
6. Integrate with token generation pipeline

## Context
Top-k sampling is crucial for text generation quality and diversity control.

## Priority
Medium-High - Generation quality feature missing