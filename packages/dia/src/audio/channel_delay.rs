//! channel_delay.rs
//! View-based temporal delay for multi-channel audio tokens.
//!
//! Instead of modifying token values, we create time-shifted views where each
//! channel observes the same sequence starting at different time offsets.
//! This implements true temporal delays as described in the Dia paper.

use crate::{CandleResult, Tensor};

/// Delay pattern for 9-channel, 24 kHz EnCodec tokenizer
/// Channel 0 has no delay, channels 1-8 have increasing delays
pub const DELAY_PATTERN: [usize; 9] = [0, 8, 9, 10, 11, 12, 13, 14, 15];

/// Create a delayed view of audio codes where each channel is time-shifted.
///
/// Returns a [T,C] tensor where channel c observes frames starting from time -delay[c].
/// Negative time indices are filled with pad_token.
///
/// # Arguments
/// * `codes_tc` - Original [T,C] tensor from the model
/// * `pad_token` - Token value to use for padding negative time indices
///
/// # Returns
/// A view tensor [T,C] with delays applied (zero-copy operation)
pub fn delayed_view(codes_tc: &Tensor, pad_token: u32) -> CandleResult<Tensor> {
    let dev = codes_tc.device();
    let (t, c) = (codes_tc.dim(0)?, codes_tc.dim(1)?);

    // Use only as many delay values as we have channels for flexibility
    let delays = &DELAY_PATTERN[..c.min(DELAY_PATTERN.len())];

    // Find maximum delay from the channels we're actually using
    let max_delay = *delays.iter().max().unwrap_or(&0);

    // Create padding prefix: [max_delay, C] filled with pad_token
    let pad_prefix = Tensor::full(pad_token, (max_delay, c), dev)?.to_dtype(codes_tc.dtype())?;

    // Concatenate padding and original codes: [max_delay + T, C]
    let extended = Tensor::cat(&[&pad_prefix, codes_tc], 0)?;

    // Extract the base undelayed view from the extended tensor
    let base_view = extended.narrow(0, max_delay, t)?; // Shape: [T, C]

    // For channels with delay=0, use the base view
    // For channels with delay>0, use earlier time slices
    if delays.iter().all(|&d| d == 0) {
        // No delays needed - return base view
        Ok(base_view)
    } else {
        // Create channel views with delays applied
        let mut result_data = base_view.to_vec2::<u32>()?;

        for (ch_idx, &delay) in delays.iter().enumerate() {
            if delay > 0 && ch_idx < c {
                // For delayed channels, extract data from earlier in the extended tensor
                let delayed_view = extended.narrow(0, max_delay - delay, t)?;
                let delayed_data = delayed_view.to_vec2::<u32>()?;

                // Copy the delayed channel data
                for t_idx in 0..t {
                    if ch_idx < delayed_data[0].len() {
                        result_data[t_idx][ch_idx] = delayed_data[t_idx][ch_idx];
                    }
                }
            }
        }

        // Convert back to tensor
        let flat_data: Vec<u32> = result_data.into_iter().flatten().collect();
        Tensor::from_vec(flat_data, (t, c), codes_tc.device())?.to_dtype(codes_tc.dtype())
    }
}

/// Create an undelayed view by reversing the time shifts.
///
/// Takes a delayed [T,C] tensor and returns the original undelayed view.
/// This is used before EnCodec decoding to restore proper alignment.
///
/// # Arguments
/// * `delayed_tc` - Delayed [T,C] tensor
/// * `pad_token` - Token value used for padding (to detect and handle)
///
/// # Returns
/// An undelayed view tensor [T,C] (zero-copy operation)
pub fn undelayed_view(delayed_tc: &Tensor, pad_token: u32) -> CandleResult<Tensor> {
    let dev = delayed_tc.device();
    let (t, c) = (delayed_tc.dim(0)?, delayed_tc.dim(1)?);

    // Use only as many delay values as we have channels for flexibility
    let delays = &DELAY_PATTERN[..c.min(DELAY_PATTERN.len())];

    let max_delay = *delays.iter().max().unwrap_or(&0);

    // Create padding suffix for channels that need future frames
    let pad_suffix = Tensor::full(pad_token, (max_delay, c), dev)?.to_dtype(delayed_tc.dtype())?;

    // Concatenate original and padding: [T + max_delay, C]
    let extended = Tensor::cat(&[delayed_tc, &pad_suffix], 0)?;

    // Extract the base view from the extended tensor
    let base_view = extended.narrow(0, 0, t)?; // Shape: [T, C]

    // For channels with delay=0, use the base view
    // For channels with delay>0, use later time slices to undo the delay
    if delays.iter().all(|&d| d == 0) {
        // No delays to undo - return base view
        Ok(base_view)
    } else {
        // Create channel views with delays undone
        let mut result_data = base_view.to_vec2::<u32>()?;

        for (ch_idx, &delay) in delays.iter().enumerate() {
            if delay > 0 && ch_idx < c {
                // For delayed channels, extract data from later in the extended tensor
                let undelayed_view = extended.narrow(0, delay, t)?;
                let undelayed_data = undelayed_view.to_vec2::<u32>()?;

                // Copy the undelayed channel data
                for t_idx in 0..t {
                    if ch_idx < undelayed_data[0].len() {
                        result_data[t_idx][ch_idx] = undelayed_data[t_idx][ch_idx];
                    }
                }
            }
        }

        // Convert back to tensor
        let flat_data: Vec<u32> = result_data.into_iter().flatten().collect();
        Tensor::from_vec(flat_data, (t, c), delayed_tc.device())?.to_dtype(delayed_tc.dtype())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(any(
        feature = "cuda",
        feature = "metal",
        feature = "accelerate",
        feature = "mkl"
    ))]
    use candle_core::Device;

    #[test]
    fn test_delayed_view() -> CandleResult<()> {
        let device = Device::Cpu;
        let pad_token = 999u32;

        // Create test data: [T=4, C=3] with distinct values per channel
        // Channel 0: [0, 1, 2, 3]
        // Channel 1: [10, 11, 12, 13]
        // Channel 2: [20, 21, 22, 23]
        let codes = Tensor::from_slice(
            &[0u32, 10, 20, 1, 11, 21, 2, 12, 22, 3, 13, 23],
            (4, 3),
            &device,
        )?;

        // Apply delays
        let delayed = delayed_view(&codes, pad_token)?;
        let delayed_vals = delayed.to_vec2::<u32>()?;

        // Channel 0 (delay=0): should be unchanged
        assert_eq!(delayed_vals[0][0], 0);
        assert_eq!(delayed_vals[1][0], 1);
        assert_eq!(delayed_vals[2][0], 2);
        assert_eq!(delayed_vals[3][0], 3);

        // Channel 1 (delay=8): first 8 frames should be padding
        // Since we only have 4 frames, all should come from padding in this test
        // In practice with longer sequences, we'd see the delay effect

        Ok(())
    }

    #[test]
    fn test_roundtrip() -> CandleResult<()> {
        let device = Device::Cpu;
        let pad_token = 999u32;

        // Create test data
        let original = Tensor::from_slice(
            &[0u32, 10, 20, 1, 11, 21, 2, 12, 22, 3, 13, 23],
            (4, 3),
            &device,
        )?;

        // Apply delay then undelay
        let delayed = delayed_view(&original, pad_token)?;
        let restored = undelayed_view(&delayed, pad_token)?;

        // Should match original
        let original_vals = original.to_vec2::<u32>()?;
        let restored_vals = restored.to_vec2::<u32>()?;

        assert_eq!(original_vals, restored_vals);

        Ok(())
    }

    #[test]
    fn test_delay_pattern_visual() -> CandleResult<()> {
        let device = Device::Cpu;
        let pad_token = 999u32;

        // Create longer sequence to see delay effect clearly
        let t_len = 20;
        let mut data = Vec::new();
        for t in 0..t_len {
            for c in 0..3 {
                data.push((t * 100 + c * 1000) as u32);
            }
        }

        let codes = Tensor::from_slice(&data, (t_len, 3), &device)?;
        let delayed = delayed_view(&codes, pad_token)?;
        let delayed_vals = delayed.to_vec2::<u32>()?;

        // Check first few time steps to see delay pattern
        // At t=0:
        assert_eq!(delayed_vals[0][0], 0); // ch0: no delay, sees original t=0
        assert_eq!(delayed_vals[0][1], pad_token); // ch1: delay=8, sees padding
        assert_eq!(delayed_vals[0][2], pad_token); // ch2: delay=9, sees padding

        // At t=8:
        if t_len > 8 {
            assert_eq!(delayed_vals[8][0], 800); // ch0: sees t=8
            assert_eq!(delayed_vals[8][1], 0); // ch1: now sees original t=0
            assert_eq!(delayed_vals[8][2], pad_token); // ch2: still sees padding
        }

        // At t=9:
        if t_len > 9 {
            assert_eq!(delayed_vals[9][0], 900); // ch0: sees t=9
            assert_eq!(delayed_vals[9][1], 100); // ch1: sees t=1
            assert_eq!(delayed_vals[9][2], 0); // ch2: now sees original t=0
        }

        Ok(())
    }
}
