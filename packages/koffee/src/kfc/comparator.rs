//---
// path: potter-dsp/src/kfc/comparator.rs
//---

#![allow(unsafe_code)] // Required for lifetime transmutation in DTW
//! KFC Comparator – DTW-based similarity scoring
//!
//! Compares two sequences using Dynamic Time Warping with cosine similarity.

use super::dtw::Dtw;
use core::cmp::min;

/// Dynamic-time-warping **KFC comparator** backed by cosine distance.
///
/// *   **No heap allocations per call** – the internal `Dtw` matrix is
///     created once in `new` and re-used.
/// *   Scores are mapped to **\[0, 1\]** via a logistic curve so the
///     rest of the pipeline can treat them as probabilities.
/// *   Fully panic-free.
///
/// ```no_run
/// # use potter_dsp::kfc::{KfcComparator, cosine_similarity};
/// let mut cmp = KfcComparator::new(0.22, 5); // score_ref, band_size
/// let score = cmp.compare(&a_frames, &b_frames); // &[&[f32]], &[&[f32]]
/// ```
/// Internal design – why the extra `tmp_*` buffers?  
///  • We want to minimize heap allocations per `compare` call.  
///  • DTW (from `super::dtw`) works with owned Vec<f32> data.
///  • We therefore keep two reusable `Vec<Vec<f32>>` buffers
///    inside the struct, fill them with cloned frame data and hand them to DTW.  
///  • After the call they are cleared and ready for reuse.
#[allow(dead_code)]
pub struct KfcComparator {
    score_ref: f32,
    band_size: u16,
    dtw: Dtw<&'static [f32]>,   // reusable cost matrix
    tmp_a: Vec<&'static [f32]>, // reused - no alloc after first call
    tmp_b: Vec<&'static [f32]>,
}

impl Clone for KfcComparator {
    fn clone(&self) -> Self {
        Self {
            score_ref: self.score_ref,
            band_size: self.band_size,
            dtw: Dtw::new(Self::distance),
            tmp_a: Vec::with_capacity(self.tmp_a.capacity()),
            tmp_b: Vec::with_capacity(self.tmp_b.capacity()),
        }
    }
}

#[allow(dead_code)]
impl KfcComparator {
    /// * `score_ref` – empirical reference cost that maps to 0.5 probability
    /// * `band_size` – Sakoe-Chiba band width (frames) for the DTW window
    pub fn new(score_ref: f32, band_size: u16) -> Self {
        Self {
            score_ref,
            band_size,
            dtw: Dtw::new(Self::distance),
            tmp_a: Vec::with_capacity(64), // allocate once – sized for typical wake-word
            tmp_b: Vec::with_capacity(64),
        }
    }

    /// Compare two KFC sequences (`[frame][coeff]`) and return a **0-1 score**.
    ///
    /// The call is O(n·band) where *n* = `max(a.len(), b.len())`.
    pub fn compare(&mut self, a: &[&[f32]], b: &[&[f32]]) -> f32 {
        /* ---------------------------------------------------------
         * Build &'static [f32] views into the *borrowed* input data
         * without cloning the coefficients.  We reuse the same
         * vectors every call to avoid repeated allocations.
         * ------------------------------------------------------ */

        self.tmp_a.clear();
        self.tmp_b.clear();
        self.tmp_a.reserve(a.len()); // reuse underlying buf
        self.tmp_b.reserve(b.len());

        // SAFETY: The transmuted slices live only until we clear the
        //         buffers (end of this function). DTW never stores
        //         them beyond its call, so extending the lifetime is
        //         sound.
        unsafe {
            for &frame in a {
                self.tmp_a
                    .push(core::mem::transmute::<&[f32], &'static [f32]>(frame));
            }
            for &frame in b {
                self.tmp_b
                    .push(core::mem::transmute::<&[f32], &'static [f32]>(frame));
            }
        }

        let cost =
            self.dtw
                .compute_optimal_path_with_window(&self.tmp_a, &self.tmp_b, self.band_size);

        /* ---------------------------------------------------------
         * Safety: the &'static slices in tmp_a/tmp_b out-live the
         * DTW call **only**. Clear now so the struct never stores
         * dangling pointers after we return to the caller.
         * ------------------------------------------------------ */
        self.tmp_a.clear(); // drop 'static slices before returning → no UB
        self.tmp_b.clear();

        // Normalize by path length and convert to probability
        let norm = cost / (a.len() + b.len()) as f32;
        self.to_probability(norm)
    }

    /// DTW local distance = 1 - cosine_similarity.
    #[inline]
    pub fn distance(ax: &[f32], bx: &[f32]) -> f32 {
        1.0 - cosine_similarity(ax, bx)
    }

    /// Static distance function for use in other parts of the code.
    #[inline]
    pub fn calculate_distance(ax: &[f32], bx: &[f32]) -> f32 {
        Self::distance(ax, bx)
    }

    #[inline]
    fn to_probability(&self, cost: f32) -> f32 {
        1.0 / (1.0 + ((cost - self.score_ref) / self.score_ref).exp())
    }
}

/// Cosine similarity ∈ \[-1, 1\] (truncated to min length when slices differ).
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let n = min(a.len(), b.len());
    if n == 0 {
        return 0.0;
    }
    let (mut dot, mut norm_a, mut norm_b) = (0.0, 0.0, 0.0);
    for i in 0..n {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = (norm_a * norm_b).sqrt();
    if denom > 0.0 { dot / denom } else { 0.0 }
}
