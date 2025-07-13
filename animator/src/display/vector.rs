/// Vector display utilities
use crate::input::Matrix;

/// Vector visualization function for multi-channel data
pub fn vector(data: &Matrix<f64>) -> Vec<(f64, f64)> {
    if data.is_empty() {
        return vec![];
    }

    // For vector mode, we'll create a phase plot using the first two channels
    // If only one channel exists, plot against its index
    match data.len() {
        1 => {
            // Single channel: plot value vs index
            data[0]
                .iter()
                .enumerate()
                .map(|(i, &value)| (i as f64, value))
                .collect()
        }
        _ => {
            // Multi-channel: create XY plot using first two channels
            let min_len = data[0].len().min(data[1].len());
            data[0]
                .iter()
                .zip(data[1].iter())
                .take(min_len)
                .map(|(&x, &y)| (x, y))
                .collect()
        }
    }
}
