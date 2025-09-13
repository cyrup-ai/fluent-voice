# Fix Character-Based Text Tokenization Stub

## Problem
The TTS model uses naive character-to-u32 mapping for text tokenization instead of proper tokenization, making it unsuitable for production use.

## Location
**File**: `packages/kyutai/src/tts.rs`  
**Lines**: 227-232

## Current Code
```rust
/// Tokenize text input - simple character-based implementation
fn tokenize_text(&self, text: &str) -> Result<Vec<u32>> {
    // Simple character-based tokenization
    // In a full implementation, this would use a proper tokenizer like SentencePiece
    let text_tokens: Vec<u32> = text.chars().map(|c| c as u32).collect();
    Ok(text_tokens)
}
```

## Issue Details
- Maps each character directly to its Unicode codepoint as u32
- No vocabulary mapping or subword tokenization
- Comment explicitly states this is not production-ready
- Will produce incorrect tokens for the language model
- Cannot handle out-of-vocabulary characters properly
- No normalization or preprocessing

## Solution Required
1. **Implement proper tokenizer integration** using the existing `tokenizers` dependency
2. **Load appropriate tokenizer model** (SentencePiece, BPE, or WordPiece)
3. **Add vocabulary validation** to ensure tokens are within model's expected range
4. **Handle special tokens** (start, end, padding, unknown)

## Implementation Steps
1. Use the existing `tokenizers` crate dependency (already in Cargo.toml)
2. Load tokenizer model file (likely SentencePiece for Moshi)
3. Replace character mapping with proper tokenizer.encode()
4. Add vocabulary size validation against `config.lm.text_out_vocab_size`
5. Handle tokenizer errors and edge cases
6. Add proper text preprocessing (normalization, case handling)

## Example Implementation
```rust
use tokenizers::Tokenizer;

fn tokenize_text(&self, text: &str) -> Result<Vec<u32>> {
    let tokenizer = self.load_tokenizer()?; // Load from model files
    let encoding = tokenizer.encode(text, false)
        .map_err(|e| MoshiError::Tokenization(e))?;
    
    let tokens: Vec<u32> = encoding.get_ids()
        .iter()
        .map(|&id| id as u32)
        .filter(|&id| id < self.config.lm.text_out_vocab_size as u32)
        .collect();
    
    Ok(tokens)
}
```

## Priority
**MEDIUM** - Affects text processing quality but doesn't break core functionality

## Acceptance Criteria
- [ ] Uses proper tokenizer (SentencePiece/BPE) instead of character mapping
- [ ] Validates tokens are within vocabulary range
- [ ] Handles special tokens correctly
- [ ] Includes proper error handling for tokenization failures
- [ ] Removes stub comments and placeholder implementation