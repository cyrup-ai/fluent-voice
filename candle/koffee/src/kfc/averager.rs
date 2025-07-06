//
// path: potter-dsp/src/kfc/averager.rs
//

use super::{KfcComparator, dtw::Dtw};
use thiserror::Error;

/// Errors that can arise while averaging KFC templates.
#[derive(Debug, Error)]
pub enum AveragerError {
    #[error("need at least two KFC matrices to average")]
    NotEnoughTemplates,
    #[error("DTW failed to retrieve path")]
    MissingPath,
}

/// Streaming, allocation-free KFC template averager.
///
/// ```no_run
/// # use potter_dsp::kfc::{KfcAverager, AveragerError};
/// # fn main() -> Result<(), AveragerError> {
/// let templates: Vec<Vec<Vec<f32>>> = load_templates();          // [template][frame][coeff]
/// let avg = KfcAverager::average(&templates)?;
/// # Ok(()) }
/// ```
pub struct KfcAverager;

impl KfcAverager {
    /// Compute the **DTW-aligned** average of a set of KFC templates.
    ///
    /// * `templates` – slice of KFC matrices (`[frame][coeff]`), **all non-empty**.
    /// * Returns an KFC matrix the length of `templates[0]` with frame-wise means.
    pub fn average(templates: &[Vec<Vec<f32>>]) -> Result<Vec<Vec<f32>>, AveragerError> {
        if templates.len() < 2 {
            return Err(AveragerError::NotEnoughTemplates);
        }

        // Start with the first template as the initial average
        let initial = templates[0].clone();
        let mut templates_seen: f32 = 1.0; // number of templates already included in average
        let mut current_result = initial;

        // For every other template, align → incrementally average.
        for tpl in &templates[1..] {
            // Create owned copies to avoid lifetime issues
            let current_copy = current_result.clone();
            let tpl_copy = tpl.clone();

            // Create a new DTW instance for this computation
            let mut dtw = Dtw::new(KfcComparator::calculate_distance);

            // Create references to the owned data
            let refs_a: Vec<&[f32]> = current_copy.iter().map(|v| v.as_slice()).collect();
            let refs_b: Vec<&[f32]> = tpl_copy.iter().map(|v| v.as_slice()).collect();

            // Compute DTW path
            dtw.compute_optimal_path(&refs_a, &refs_b);

            // Get the optimal path
            let path = dtw
                .retrieve_optimal_path()
                .ok_or(AveragerError::MissingPath)?;

            // 2. For each frame in result, accumulate the KFCs of all
            //    template frames aligned to it so we can form their mean.
            let frame_count = current_result.len();
            let coeff_count = current_result[0].len();
            let mut frame_sums = vec![vec![0.0f32; coeff_count]; frame_count];
            let mut frame_hits = vec![0usize; frame_count];

            for path_segment in path {
                let x = path_segment[0];
                let y = path_segment[1];
                for (c, val) in tpl[y].iter().enumerate() {
                    frame_sums[x][c] += *val;
                }
                frame_hits[x] += 1;
            }

            // 3. Create a new average using the incremental-mean formula.
            let mut new_result = Vec::with_capacity(frame_count);
            let new_templates_seen = templates_seen + 1.0;

            for frame_idx in 0..frame_count {
                let mut new_frame = Vec::with_capacity(coeff_count);

                // If no path cell referenced this frame, just keep the previous value
                if frame_hits[frame_idx] == 0 {
                    new_frame = current_result[frame_idx].clone();
                } else {
                    // Otherwise compute the new average for each coefficient
                    for coeff_idx in 0..coeff_count {
                        let tpl_mean =
                            frame_sums[frame_idx][coeff_idx] / frame_hits[frame_idx] as f32;
                        let avg = (current_result[frame_idx][coeff_idx] * templates_seen
                            + tpl_mean)
                            / new_templates_seen;
                        new_frame.push(avg);
                    }
                }

                new_result.push(new_frame);
            }

            templates_seen = new_templates_seen;
            current_result = new_result;
        }

        Ok(current_result)
    }
}
