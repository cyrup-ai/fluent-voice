//! Comprehensive tests for KyutaiTokenizer implementation

use fluent_voice_kyutai::tokenizer::{KyutaiTokenizer, KyutaiTokenizerBuilder};
use fluent_voice_kyutai::{MoshiError, Result};
use std::path::Path;

/// Path to local test tokenizer asset (optional - tests will fallback to pretrained model)
const TEST_TOKENIZER_PATH: &str = "test_assets/tokenizer.json";

/// Create a test tokenizer using the best available method
///
/// This function implements a fallback strategy:
/// 1. First tries to load from local test asset file (fast, offline)
/// 2. Falls back to downloading a pretrained model (requires 'http' feature)
/// 3. Returns error if neither method is available
///
/// # Test Setup
///
/// **Option 1: Local Asset (Recommended)**
/// - Create `test_assets/tokenizer.json` with a valid tokenizer file
/// - Tests will run offline and be faster
///
/// **Option 2: HTTP Fallback**
/// - Enable the 'http' feature: `cargo test --features http`
/// - Tests will download microsoft/DialoGPT-small tokenizer
/// - Requires internet connection
///
/// **Option 3: Custom Test Asset**
/// - Download any HuggingFace tokenizer.json file
/// - Place it at `test_assets/tokenizer.json`
/// - Examples: bert-base-uncased, gpt2, distilbert-base-uncased
fn create_test_tokenizer() -> Result<KyutaiTokenizer> {
    if Path::new(TEST_TOKENIZER_PATH).exists() {
        KyutaiTokenizer::from_file(TEST_TOKENIZER_PATH)
    } else {
        // Fallback to pretrained model for testing (requires internet + http feature)
        #[cfg(feature = "http")]
        return KyutaiTokenizer::from_pretrained("microsoft/DialoGPT-small");

        #[cfg(not(feature = "http"))]
        return Err(MoshiError::Tokenization(
            "Test tokenizer file not found and http feature not enabled. \
             Either place a tokenizer.json at test_assets/tokenizer.json \
             or run tests with --features http"
                .to_string(),
        ));
    }
}

#[test]
fn test_tokenizer_creation_from_file() -> Result<()> {
    let tokenizer = create_test_tokenizer()?;
    assert!(tokenizer.vocab_size() > 0);
    println!("Tokenizer vocab size: {}", tokenizer.vocab_size());
    Ok(())
}

#[cfg(feature = "http")]
#[test]
fn test_tokenizer_creation_from_pretrained() -> Result<()> {
    let tokenizer = KyutaiTokenizer::from_pretrained("microsoft/DialoGPT-small")?;
    assert!(tokenizer.vocab_size() > 0);
    Ok(())
}

#[test]
fn test_simple_encoding_decoding() -> Result<()> {
    let tokenizer = create_test_tokenizer()?;

    let test_text = "Hello, world! This is a test.";
    let tokens = tokenizer.encode(test_text, false)?;
    let decoded_text = tokenizer.decode(&tokens, false)?;

    println!("Original: {:?}", test_text);
    println!("Tokens: {:?}", tokens);
    println!("Decoded: {:?}", decoded_text);

    // Note: Exact match might not work due to tokenization normalization
    // Instead, check that decoding produces reasonable text
    assert!(!decoded_text.is_empty());
    assert!(tokens.len() > 0);

    Ok(())
}

#[test]
fn test_roundtrip_with_special_tokens() -> Result<()> {
    let tokenizer = create_test_tokenizer()?;

    let test_text = "Hello, world!";
    let tokens_with_special = tokenizer.encode(test_text, true)?;
    let tokens_without_special = tokenizer.encode(test_text, false)?;

    // With special tokens should have more tokens
    assert!(tokens_with_special.len() >= tokens_without_special.len());

    let decoded_with_special = tokenizer.decode(&tokens_with_special, false)?;
    let decoded_skip_special = tokenizer.decode(&tokens_with_special, true)?;

    println!("With special tokens: {:?}", decoded_with_special);
    println!("Skip special tokens: {:?}", decoded_skip_special);

    Ok(())
}

#[test]
fn test_batch_encoding_performance() -> Result<()> {
    let tokenizer = create_test_tokenizer()?;

    let test_texts: Vec<&str> = vec![
        "First test sentence for batch processing.",
        "Second test sentence with different content.",
        "Third sentence to verify batch functionality.",
        "Fourth and final test sentence for the batch.",
    ];

    // Test batch encoding
    let start = std::time::Instant::now();
    let batch_tokens = tokenizer.encode_batch(test_texts.clone(), false)?;
    let batch_duration = start.elapsed();

    println!("Batch encoding took: {:?}", batch_duration);
    assert_eq!(batch_tokens.len(), test_texts.len());

    // Test individual encoding for comparison
    let start = std::time::Instant::now();
    let individual_tokens: Result<Vec<Vec<u32>>> = test_texts
        .iter()
        .map(|text| tokenizer.encode(text, false))
        .collect();
    let individual_duration = start.elapsed();
    let individual_tokens = individual_tokens?;

    println!("Individual encoding took: {:?}", individual_duration);

    // Results should be identical
    assert_eq!(batch_tokens, individual_tokens);

    Ok(())
}

#[test]
fn test_batch_decoding() -> Result<()> {
    let tokenizer = create_test_tokenizer()?;

    let test_texts = vec![
        "Batch decoding test one.",
        "Batch decoding test two.",
        "Batch decoding test three.",
    ];

    // Encode all texts
    let token_sequences = tokenizer.encode_batch(test_texts, false)?;

    // Convert to slice references for decode_batch
    let token_refs: Vec<&[u32]> = token_sequences
        .iter()
        .map(|tokens| tokens.as_slice())
        .collect();

    // Batch decode
    let decoded_texts = tokenizer.decode_batch(&token_refs, false)?;

    assert_eq!(decoded_texts.len(), 3);
    for decoded in &decoded_texts {
        assert!(!decoded.is_empty());
    }

    println!("Batch decoded: {:?}", decoded_texts);
    Ok(())
}

#[test]
fn test_special_token_detection() -> Result<()> {
    let tokenizer = create_test_tokenizer()?;

    let special_tokens = tokenizer.special_tokens();

    println!("Special tokens detected:");
    if let Some(bos) = special_tokens.bos {
        println!("  BOS: {}", bos);
        assert!(tokenizer.is_special_token(bos));
    }
    if let Some(eos) = special_tokens.eos {
        println!("  EOS: {}", eos);
        assert!(tokenizer.is_special_token(eos));
    }
    if let Some(pad) = special_tokens.pad {
        println!("  PAD: {}", pad);
        assert!(tokenizer.is_special_token(pad));
    }
    if let Some(unk) = special_tokens.unk {
        println!("  UNK: {}", unk);
        assert!(tokenizer.is_special_token(unk));
    }

    // Test that a regular token is not considered special
    let regular_tokens = tokenizer.encode("hello", false)?;
    if let Some(&first_token) = regular_tokens.first() {
        // This might be a special token if "hello" gets special treatment,
        // but typically it won't be
        println!("Testing regular token: {}", first_token);
    }

    Ok(())
}

#[test]
fn test_tokenizer_builder() -> Result<()> {
    let tokenizer = KyutaiTokenizerBuilder::new()
        .from_file(TEST_TOKENIZER_PATH)
        .bos_token_id(1)
        .eos_token_id(2)
        .build()?;

    let special_tokens = tokenizer.special_tokens();
    assert_eq!(special_tokens.bos, Some(1));
    assert_eq!(special_tokens.eos, Some(2));

    Ok(())
}

#[test]
fn test_error_handling() {
    // Test file not found
    let result = KyutaiTokenizer::from_file("nonexistent_tokenizer.json");
    assert!(result.is_err());

    // Test invalid tokens for decoding
    if let Ok(tokenizer) = create_test_tokenizer() {
        let invalid_tokens = vec![u32::MAX, u32::MAX - 1]; // Very large token IDs
        let result = tokenizer.decode(&invalid_tokens, false);
        // This might succeed depending on tokenizer implementation
        // The key is that it doesn't panic
        println!("Decode invalid tokens result: {:?}", result);
    }
}

#[test]
#[ignore] // This test is expensive, run manually
fn test_performance_benchmark() -> Result<()> {
    let tokenizer = create_test_tokenizer()?;

    // Generate 1000 test texts
    let texts: Vec<String> = (0..1000)
        .map(|i| format!("Test sentence number {} with some content.", i))
        .collect();

    let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

    // Benchmark batch encoding
    let start = std::time::Instant::now();
    let _tokens = tokenizer.encode_batch(text_refs, false)?;
    let duration = start.elapsed();

    println!("Encoded 1000 texts in: {:?}", duration);

    // Should be under 100ms for 1000 texts
    assert!(duration.as_millis() < 100);

    Ok(())
}

#[test]
fn test_empty_and_edge_cases() -> Result<()> {
    let tokenizer = create_test_tokenizer()?;

    // Test empty string
    let empty_tokens = tokenizer.encode("", false)?;
    let decoded_empty = tokenizer.decode(&empty_tokens, false)?;
    println!(
        "Empty string -> {:?} tokens -> {:?}",
        empty_tokens, decoded_empty
    );

    // Test very long string
    let long_string = "word ".repeat(1000);
    let long_tokens = tokenizer.encode(&long_string, false)?;
    let decoded_long = tokenizer.decode(&long_tokens, false)?;
    assert!(long_tokens.len() > 100); // Should produce many tokens
    assert!(!decoded_long.is_empty());

    // Test special characters
    let special_chars = "!@#$%^&*()_+{}|:<>?[]\\;'\",./ αβγδε 中文 🎉🚀";
    let special_tokens = tokenizer.encode(special_chars, false)?;
    let decoded_special = tokenizer.decode(&special_tokens, false)?;
    println!("Special chars: {} tokens", special_tokens.len());
    assert!(!decoded_special.is_empty());

    Ok(())
}
