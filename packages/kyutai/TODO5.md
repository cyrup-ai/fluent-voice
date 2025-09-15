# TODO5: Implement Transformer Cross-Attention Mechanism

## Issue
**File**: `packages/kyutai/src/transformer.rs`
**Line**: 280

## Problem
Missing cross-attention implementation:

```rust
// TODO: Implement cross-attention mechanism when cross_attention config is enabled
```

## Required Fix
1. Implement cross-attention layers for transformer
2. Add key-value memory for cross-attention
3. Handle attention masking and position encoding
4. Integrate with existing transformer architecture

## Implementation Steps
1. Add cross-attention linear layers (query, key, value projections)
2. Implement cross-attention forward pass
3. Add memory management for encoder states
4. Handle attention masking for variable-length sequences
5. Integrate with multi-head attention mechanism
6. Add proper initialization and parameter management

## Context
Cross-attention is essential for encoder-decoder architectures and conditioning mechanisms in language models.

## Priority
High - Core transformer functionality missing