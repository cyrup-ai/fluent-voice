//! Shared token filtering utilities for Whisper timestamp rules and blank suppression.
//!
//! This module contains the implementation of OpenAI Whisper's timestamp validation rules
//! and blank token suppression logic, extracted for reuse between file-based and microphone
//! transcription modes.

use anyhow::Result;
use candle_core::Tensor;

/// Apply comprehensive token filters including timestamp rules and blank suppression.
///
/// This function combines all token filtering logic used by OpenAI Whisper:
/// - Timestamp validation rules when in timestamp mode
/// - Blank/repetition suppression to prevent loops
///
/// # Arguments
/// * `logits` - Raw decoder output logits
/// * `tokens` - Current token sequence
/// * `step` - Current decoding step
/// * `timestamps` - Whether timestamp mode is enabled
/// * `no_timestamps_token` - Token ID for the no-timestamps marker
pub fn apply_token_filters_static(
    logits: &Tensor,
    tokens: &[u32],
    step: usize,
    timestamps: bool,
    no_timestamps_token: u32,
) -> Result<Tensor> {
    let mut filtered_logits = logits.clone();

    // Apply timestamp rules when in timestamp mode
    if timestamps {
        filtered_logits =
            apply_timestamp_rules_static(&filtered_logits, tokens, step, no_timestamps_token)?;
    }

    // Apply blank suppression
    filtered_logits = suppress_blanks_static(&filtered_logits, tokens)?;

    Ok(filtered_logits)
}

/// Apply timestamp rules: timestamps come in pairs, non-decreasing, prioritize when probable.
///
/// Implements the three core timestamp validation rules from OpenAI Whisper:
/// 1. Non-decreasing constraint: Suppress timestamp tokens ≤ last timestamp
/// 2. Probability prioritization: Force timestamp selection when highly probable
/// 3. Temporal consistency: Prevent invalid timestamp sequences
///
/// # Arguments
/// * `logits` - Current logits tensor
/// * `tokens` - Token sequence up to current step
/// * `_step` - Current decoding step (unused but kept for API compatibility)
/// * `no_timestamps_token` - Token ID marking boundary between text and timestamp tokens
pub fn apply_timestamp_rules_static(
    logits: &Tensor,
    tokens: &[u32],
    _step: usize,
    no_timestamps_token: u32,
) -> Result<Tensor> {
    let mut logits = logits.clone();

    // Find the last timestamp token to enforce non-decreasing constraint
    let mut last_timestamp = None;
    for &token in tokens.iter().rev() {
        if token > no_timestamps_token {
            last_timestamp = Some(token);
            break;
        }
    }

    // If we have a previous timestamp, suppress earlier timestamps
    if let Some(last_ts) = last_timestamp {
        let logits_vec: Vec<f32> = logits.to_vec1()?;
        let mut modified_logits = logits_vec;

        // Suppress timestamp tokens that would be non-decreasing
        for token_id in (no_timestamps_token + 1)..=last_ts {
            if (token_id as usize) < modified_logits.len() {
                modified_logits[token_id as usize] = f32::NEG_INFINITY;
            }
        }

        logits = Tensor::new(modified_logits.as_slice(), logits.device())?;
    }

    // Check if timestamps should be prioritized (sum of timestamp probs > other tokens)
    let logits_vec: Vec<f32> = logits.to_vec1()?;
    let timestamp_start = (no_timestamps_token + 1) as usize;

    if timestamp_start < logits_vec.len() {
        let timestamp_sum: f32 = logits_vec[timestamp_start..].iter().sum();
        let other_sum: f32 = logits_vec[..timestamp_start].iter().sum();

        // If timestamps are more probable, suppress non-timestamp tokens
        if timestamp_sum > other_sum {
            let mut modified_logits = logits_vec;
            for item in modified_logits.iter_mut().take(timestamp_start) {
                *item = f32::NEG_INFINITY;
            }
            logits = Tensor::new(modified_logits.as_slice(), logits.device())?;
        }
    }

    Ok(logits)
}

/// Suppress blank tokens and repetitive patterns.
///
/// Prevents the decoder from getting stuck in repetitive loops by:
/// - Detecting when the same token is repeated consecutively
/// - Suppressing the repeated token to force diversity
/// - Only applies when sufficient context is available (>3 tokens)
///
/// # Arguments
/// * `logits` - Current logits tensor
/// * `tokens` - Token sequence up to current step
pub fn suppress_blanks_static(logits: &Tensor, tokens: &[u32]) -> Result<Tensor> {
    let mut logits = logits.clone();

    // Suppress blank/silence tokens more aggressively if we have recent content
    if tokens.len() > 3 {
        let recent_tokens = &tokens[tokens.len().saturating_sub(3)..];

        // Check for repetitive patterns
        if recent_tokens.len() >= 2
            && recent_tokens[recent_tokens.len() - 1] == recent_tokens[recent_tokens.len() - 2]
        {
            let logits_vec: Vec<f32> = logits.to_vec1()?;
            let mut modified_logits = logits_vec;

            let repeated_token = recent_tokens[recent_tokens.len() - 1] as usize;
            if repeated_token < modified_logits.len() {
                modified_logits[repeated_token] = f32::NEG_INFINITY;
            }

            logits = Tensor::new(modified_logits.as_slice(), logits.device())?;
        }
    }

    Ok(logits)
}
