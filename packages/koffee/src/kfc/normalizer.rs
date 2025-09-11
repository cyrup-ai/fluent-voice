//! KFC frame normalisation utilities.
//!
//! The wake-word algorithms expect KFC coefficients to be **mean-centred**
//! across the analysis window.  `KfcNormalizer` provides a fast in-place
//! normaliser plus an allocation-returning variant when you need to preserve
//! the original data.
//!
//! Both methods subtract the per-coefficient mean (µ) so every column of the
//! `frames` matrix has zero mean.
//!
//! ```rust
//! use rustpotter::kfc::KfcNormalizer;
//!
//! let mut frames = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
//! KfcNormalizer::normalize(&mut frames);
//! assert!((frames[0][0] + frames[1][0]).abs() < 1e-6); // ≈ 0
//! ```

/// Stateless helper for KFC mean-centering.
pub struct KfcNormalizer;

impl KfcNormalizer {
    /// In-place mean-centre of `frames`.
    ///
    /// * `frames` – slice of *mutable* KFC frames; all frames **must** have
    ///   the same length *(number of coefficients)*.
    ///
    /// When `frames` is empty the function returns immediately.<br>
    /// Runs in **O(n × m)** where *n* is the number of frames and *m* the
    /// coefficient count, with two tight loops and no heap allocation.
    pub fn normalize(frames: &mut [Vec<f32>]) {
        if frames.is_empty() {
            return;
        }
        let coeffs = frames[0].len();
        let mut mean = vec![0f32; coeffs];

        // --- accumulate column-wise sum ----------------------------------
        for f in frames.iter() {
            debug_assert_eq!(
                f.len(),
                coeffs,
                "all KFC frames must have the same coefficient count"
            );
            for (j, &v) in f.iter().enumerate() {
                mean[j] += v;
            }
        }

        // --- convert sums → means ---------------------------------------
        let n_inv = 1.0 / frames.len() as f32;
        for m in &mut mean {
            *m *= n_inv;
        }

        // --- subtract in-place ------------------------------------------
        for f in frames.iter_mut() {
            for (j, v) in f.iter_mut().enumerate() {
                *v -= mean[j];
            }
        }
    }

    /// Allocate-new variant: returns a **new** vector with normalised frames.
    ///
    /// Preserves the input slice untouched.  Internally re-uses the same mean
    /// calculation as [`normalize`].
    #[allow(dead_code)]
    pub fn normalize_to_new(frames: &[Vec<f32>]) -> Vec<Vec<f32>> {
        if frames.is_empty() {
            return Vec::new();
        }
        let coeffs = frames[0].len();
        let mut mean = vec![0f32; coeffs];

        for f in frames {
            for (j, &v) in f.iter().enumerate() {
                mean[j] += v;
            }
        }
        let n_inv = 1.0 / frames.len() as f32;
        for m in &mut mean {
            *m *= n_inv;
        }

        frames
            .iter()
            .map(|f| f.iter().enumerate().map(|(j, v)| v - mean[j]).collect())
            .collect()
    }
}

/* --------------------------------------------------------------------- */
/*  Unit-tests                                                           */

#[cfg(test)]
mod tests {
    use super::KfcNormalizer;

    #[test]
    fn in_place_normalisation_zeroes_column_means() {
        let mut frames = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        KfcNormalizer::normalize(&mut frames);

        let coeffs = frames[0].len();
        let mut col_sum = vec![0f32; coeffs];
        for f in &frames {
            for (j, v) in f.iter().enumerate() {
                col_sum[j] += v;
            }
        }
        for s in col_sum {
            assert!(s.abs() < 1e-5);
        }
    }

    #[test]
    fn allocate_new_preserves_input() {
        let orig = vec![vec![1.0, 1.0], vec![3.0, 5.0]];
        let normalised = KfcNormalizer::normalize_to_new(&orig);
        assert_eq!(orig, vec![vec![1.0, 1.0], vec![3.0, 5.0]]);
        // Ensure column means ~0
        let mean0 = (normalised[0][0] + normalised[1][0]) / 2.0;
        let mean1 = (normalised[0][1] + normalised[1][1]) / 2.0;
        assert!(mean0.abs() < 1e-5);
        assert!(mean1.abs() < 1e-5);
    }
}
