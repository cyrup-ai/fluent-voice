// src/visualizer.rs

use crate::error::{MoshiError, Result};
use std::io::{self, Write};

/// Visualizes an audio waveform in the console using ASCII characters.
///
/// This function takes PCM samples and prints a simple waveform visualization to the console.
/// The visualization is scaled to fit the terminal width.
///
/// # Arguments
///
/// * `samples` - A slice of f32 PCM samples.
/// * `width` - The width of the visualization in characters.
///
/// # Returns
///
/// * `Result<()>` - Ok if the visualization was printed successfully, otherwise an error.
pub fn visualize_waveform(samples: &[f32], width: usize) -> Result<()> {
    if samples.is_empty() {
        return Err(MoshiError::Custom(
            "No samples provided for visualization".into(),
        ));
    }

    let max_amp = samples.iter().fold(0.0f32, |acc, &s| acc.max(s.abs()));
    let scale = if max_amp > 0.0 { 1.0 / max_amp } else { 1.0 };

    let bin_size = (samples.len() + width - 1) / width;
    let stdout = io::stdout();
    let mut handle = stdout.lock();

    for i in 0..width {
        let start = i * bin_size;
        let end = std::cmp::min(start + bin_size, samples.len());
        if start >= end {
            break;
        }

        let bin_samples = &samples[start..end];
        let bin_max = bin_samples.iter().fold(0.0f32, |acc, &s| acc.max(s.abs())) * scale;

        let height = (bin_max * 20.0) as usize; // 20 lines high
        for _ in 0..height {
            handle
                .write_all(b"|")
                .map_err(|e| MoshiError::Io(e.into()))?;
        }
        handle
            .write_all(b"\n")
            .map_err(|e| MoshiError::Io(e.into()))?;
    }

    Ok(())
}

/// Visualizes audio codes as a heatmap in the console.
///
/// This function takes a 2D vector of audio codes and prints a simple heatmap visualization.
///
/// # Arguments
///
/// * `codes` - A 2D vector of u32 audio codes.
///
/// # Returns
///
/// * `Result<()>` - Ok if the visualization was printed successfully, otherwise an error.
pub fn visualize_codes(codes: &[Vec<u32>]) -> Result<()> {
    if codes.is_empty() || codes[0].is_empty() {
        return Err(MoshiError::Custom(
            "No codes provided for visualization".into(),
        ));
    }

    let max_code = codes
        .iter()
        .flat_map(|row| row.iter())
        .max()
        .cloned()
        .unwrap_or(0);
    let scale = if max_code > 0 { 255 / max_code } else { 1 };

    let stdout = io::stdout();
    let mut handle = stdout.lock();

    for row in codes {
        for &code in row {
            let intensity = (code * scale) as u8;
            let char = match intensity {
                0..=63 => ' ',
                64..=127 => '.',
                128..=191 => '*',
                192..=255 => '#',
            };
            write!(&mut handle, "{}", char).map_err(|e| MoshiError::Io(e.into()))?;
        }
        writeln!(&mut handle).map_err(|e| MoshiError::Io(e.into()))?;
    }

    Ok(())
}
