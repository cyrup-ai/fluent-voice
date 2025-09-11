//! Production tokenizer implementation for Kyutai language model
//! 
//! Based on HuggingFace tokenizers library with proper error handling,
//! batch processing, and special token management.

use std::path::Path;
use tokenizers::Tokenizer;
use crate::error::{MoshiError, Result};

/// Production tokenizer for Kyutai language model
#[derive(Debug, Clone)]
pub struct KyutaiTokenizer {
    /// Underlying HuggingFace tokenizer
    tokenizer: Tokenizer,
    /// Vocabulary size
    vocab_size: usize,
    /// Beginning of sequence token ID
    bos_token_id: Option<u32>,
    /// End of sequence token ID
    eos_token_id: Option<u32>,
    /// Padding token ID
    pad_token_id: Option<u32>,
    /// Unknown token ID
    unk_token_id: Option<u32>,
}

impl KyutaiTokenizer {
    /// Create tokenizer from JSON file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path)
            .map_err(|e| MoshiError::Tokenization(format!("Failed to load tokenizer from file: {}", e)))?;
        
        Ok(Self::from_tokenizer(tokenizer))
    }
    
    /// Create tokenizer from pretrained model (requires 'http' feature)
    #[cfg(feature = "http")]
    pub fn from_pretrained(model_name: &str) -> Result<Self> {
        let tokenizer = Tokenizer::from_pretrained(model_name, None)
            .map_err(|e| MoshiError::Tokenization(format!("Failed to load pretrained tokenizer '{}': {}", model_name, e)))?;
        
        Ok(Self::from_tokenizer(tokenizer))
    }
    
    /// Internal constructor from HuggingFace tokenizer
    fn from_tokenizer(tokenizer: Tokenizer) -> Self {
        let vocab_size = tokenizer.get_vocab_size(true);
        let bos_token_id = Self::find_special_token_id(&tokenizer, &["<|startoftext|>", "<s>", "[BOS]", "<bos>"]);
        let eos_token_id = Self::find_special_token_id(&tokenizer, &["<|endoftext|>", "</s>", "[EOS]", "<eos>"]);
        let pad_token_id = Self::find_special_token_id(&tokenizer, &["<|pad|>", "[PAD]", "<pad>"]);
        let unk_token_id = Self::find_special_token_id(&tokenizer, &["<|unk|>", "[UNK]", "<unk>"]);
        
        Self {
            tokenizer,
            vocab_size,
            bos_token_id,
            eos_token_id,
            pad_token_id,
            unk_token_id,
        }
    }
    
    /// Find special token ID by trying multiple common formats
    fn find_special_token_id(tokenizer: &Tokenizer, candidates: &[&str]) -> Option<u32> {
        for candidate in candidates {
            if let Some(id) = tokenizer.token_to_id(candidate) {
                return Some(id);
            }
        }
        None
    }
    
    /// Encode single text to token IDs
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self.tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| MoshiError::Tokenization(format!("Failed to encode text: {}", e)))?;
        
        Ok(encoding.get_ids().to_vec())
    }
    
    /// Decode token IDs back to text
    pub fn decode(&self, tokens: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .decode(tokens, skip_special_tokens)
            .map_err(|e| MoshiError::Tokenization(format!("Failed to decode tokens: {}", e)))
    }
    
    /// Encode multiple texts in parallel (high performance)
    pub fn encode_batch(&self, texts: Vec<&str>, add_special_tokens: bool) -> Result<Vec<Vec<u32>>> {
        let encodings = self.tokenizer
            .encode_batch(texts, add_special_tokens)
            .map_err(|e| MoshiError::Tokenization(format!("Failed to encode batch: {}", e)))?;
        
        Ok(encodings.into_iter()
            .map(|encoding| encoding.get_ids().to_vec())
            .collect())
    }
    
    /// Decode multiple token sequences in parallel
    pub fn decode_batch(&self, token_sequences: &[&[u32]], skip_special_tokens: bool) -> Result<Vec<String>> {
        self.tokenizer
            .decode_batch(token_sequences, skip_special_tokens)
            .map_err(|e| MoshiError::Tokenization(format!("Failed to decode batch: {}", e)))
    }
    
    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    
    /// Get special token IDs
    pub fn special_tokens(&self) -> SpecialTokens {
        SpecialTokens {
            bos: self.bos_token_id,
            eos: self.eos_token_id,
            pad: self.pad_token_id,
            unk: self.unk_token_id,
        }
    }
    
    /// Check if token ID is a special token
    pub fn is_special_token(&self, token_id: u32) -> bool {
        let special = self.special_tokens();
        Some(token_id) == special.bos || 
        Some(token_id) == special.eos || 
        Some(token_id) == special.pad || 
        Some(token_id) == special.unk
    }
    
    /// Set custom BOS token ID (for builder pattern)
    pub(crate) fn set_bos_token_id(&mut self, id: u32) {
        self.bos_token_id = Some(id);
    }
    
    /// Set custom EOS token ID (for builder pattern) 
    pub(crate) fn set_eos_token_id(&mut self, id: u32) {
        self.eos_token_id = Some(id);
    }
    
    /// Set custom PAD token ID (for builder pattern)
    pub(crate) fn set_pad_token_id(&mut self, id: u32) {
        self.pad_token_id = Some(id);
    }
    
    /// Set custom UNK token ID (for builder pattern)
    pub(crate) fn set_unk_token_id(&mut self, id: u32) {
        self.unk_token_id = Some(id);
    }
}

/// Special token IDs for the tokenizer
#[derive(Debug, Clone, Copy)]
pub struct SpecialTokens {
    pub bos: Option<u32>,
    pub eos: Option<u32>, 
    pub pad: Option<u32>,
    pub unk: Option<u32>,
}

/// Builder for KyutaiTokenizer with custom special token configuration
#[derive(Debug, Default)]
pub struct KyutaiTokenizerBuilder {
    tokenizer_path: Option<String>,
    pretrained_model: Option<String>,
    custom_bos: Option<u32>,
    custom_eos: Option<u32>,
    custom_pad: Option<u32>,
    custom_unk: Option<u32>,
}

impl KyutaiTokenizerBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn from_file<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.tokenizer_path = Some(path.as_ref().to_string_lossy().into_owned());
        self
    }
    
    #[cfg(feature = "http")]
    pub fn from_pretrained(mut self, model_name: &str) -> Self {
        self.pretrained_model = Some(model_name.to_string());
        self
    }
    
    pub fn bos_token_id(mut self, id: u32) -> Self {
        self.custom_bos = Some(id);
        self
    }
    
    pub fn eos_token_id(mut self, id: u32) -> Self {
        self.custom_eos = Some(id);
        self
    }
    
    pub fn pad_token_id(mut self, id: u32) -> Self {
        self.custom_pad = Some(id);
        self
    }
    
    pub fn unk_token_id(mut self, id: u32) -> Self {
        self.custom_unk = Some(id);
        self
    }
    
    pub fn build(self) -> Result<KyutaiTokenizer> {
        let mut tokenizer = if let Some(path) = self.tokenizer_path {
            KyutaiTokenizer::from_file(path)?
        } else if let Some(_model) = self.pretrained_model {
            #[cfg(feature = "http")]
            return Ok(KyutaiTokenizer::from_pretrained(&_model)?);
            
            #[cfg(not(feature = "http"))]
            return Err(MoshiError::Tokenization("Pretrained models require 'http' feature".to_string()));
        } else {
            return Err(MoshiError::Tokenization("Must specify either file path or pretrained model".to_string()));
        };
        
        // Override special tokens if provided
        if let Some(id) = self.custom_bos { tokenizer.set_bos_token_id(id); }
        if let Some(id) = self.custom_eos { tokenizer.set_eos_token_id(id); }
        if let Some(id) = self.custom_pad { tokenizer.set_pad_token_id(id); }
        if let Some(id) = self.custom_unk { tokenizer.set_unk_token_id(id); }
        
        Ok(tokenizer)
    }
}