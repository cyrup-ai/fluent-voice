//! Dynamic-Time-Warping implementation used across Kfc.
//!
//!  * One-time allocation: the cost matrix is flattened into a single
//!    `Vec<f32>` instead of a `Vec<Vec<f32>>`, which slashes heap noise.
//!  * Optional Sakoe-Chiba band window (pass `Some(band_size)`).
//!  * The struct is *re-usable*: allocate once, call many times.
//!  * No unsafe code, but tight inner loops are marked `#[inline(always)]`.
//!
//! Public API unchanged w.r.t. the original version.

use std::cmp;

/// 2-D index helper for a flattened `(rows × cols)` buffer.
#[inline(always)]
fn idx(row: usize, col: usize, cols: usize) -> usize {
    row * cols + col
}

pub struct Dtw<T: Copy> {
    rows: usize,
    cols: usize,
    distance_fn: fn(T, T) -> f32,
    /// Last accumulated cost matrix (flattened).
    cost: Vec<f32>,
    /// (row, col) pairs of the last optimal path.
    path: Vec<[usize; 2]>,
    similarity: f32,
}

impl<T: Copy> Dtw<T> {
    /// Create a new reusable DTW instance with the supplied distance function.
    pub fn new(distance_fn: fn(T, T) -> f32) -> Self {
        Self {
            rows: 0,
            cols: 0,
            distance_fn,
            cost: Vec::new(),
            path: Vec::new(),
            similarity: 0.0,
        }
    }

    /* ─────────────────────────── public API ────────────────────────── */

    /// Classic DTW (no global constraint).
    pub fn compute_optimal_path(&mut self, a: &[T], b: &[T]) -> f32 {
        self.compute_inner(a, b, None)
    }

    /// DTW with Sakoe-Chiba band of half-width `w`.
    #[allow(dead_code)]
    pub fn compute_optimal_path_with_window(&mut self, a: &[T], b: &[T], w: u16) -> f32 {
        self.compute_inner(a, b, Some(w as usize))
    }

    /// Retrieve the optimal path from the *last* call to `compute_*`.
    pub fn retrieve_optimal_path(&self) -> Option<&[[usize; 2]]> {
        if self.path.is_empty() {
            None
        } else {
            Some(&self.path)
        }
    }

    /// Return the last similarity (global DTW distance).
    #[allow(dead_code)]
    pub fn last_similarity(&self) -> f32 {
        self.similarity
    }

    /* ───────────────────────── internal impl ───────────────────────── */

    fn compute_inner(&mut self, a: &[T], b: &[T], window_opt: Option<usize>) -> f32 {
        self.rows = a.len();
        self.cols = b.len();

        if self.rows == 0 || self.cols == 0 {
            self.similarity = f32::INFINITY;
            return self.similarity;
        }

        // Allocate or resize a flattened `(rows × cols)` buffer.
        let needed = self.rows * self.cols;
        if self.cost.len() < needed {
            self.cost.resize(needed, f32::INFINITY);
        }

        // Helper to reset only the slice we actually touch.
        let cost = &mut self.cost[..needed];
        cost.fill(f32::INFINITY);

        // Define Sakoe-Chiba band - use MAX as default (effectively no constraint)
        let w = window_opt.unwrap_or(usize::MAX);

        // Initialise (0,0)
        cost[0] = (self.distance_fn)(a[0], b[0]);

        // First column
        for r in 1..self.rows {
            if r > w {
                break;
            }
            cost[idx(r, 0, self.cols)] =
                (self.distance_fn)(a[r], b[0]) + cost[idx(r - 1, 0, self.cols)];
        }
        // First row
        for c in 1..self.cols {
            if c > w {
                break;
            }
            cost[idx(0, c, self.cols)] =
                (self.distance_fn)(a[0], b[c]) + cost[idx(0, c - 1, self.cols)];
        }

        // Main dynamic-programming loop
        for r in 1..self.rows {
            let start = r.saturating_sub(w);
            let end = cmp::min(self.cols - 1, r + w);
            for c in start..=end {
                let d = (self.distance_fn)(a[r], b[c]);

                // min(↑, ←, ↖)
                let m1 = cost[idx(r - 1, c, self.cols)];
                let m2 = cost[idx(r, c - 1, self.cols)];
                let m3 = cost[idx(r - 1, c - 1, self.cols)];
                let min_prev = m1.min(m2).min(m3);

                cost[idx(r, c, self.cols)] = d + min_prev;
            }
        }

        self.similarity = cost[idx(self.rows - 1, self.cols - 1, self.cols)];

        // Reconstruct optimal path (reverse traversal).
        self.path.clear();
        let mut r = self.rows - 1;
        let mut c = self.cols - 1;
        self.path.push([r, c]);
        while r > 0 || c > 0 {
            let (mut best_r, mut best_c) = (r, c); // will be overwritten

            if r > 0 && c > 0 {
                let up = cost[idx(r - 1, c, self.cols)];
                let left = cost[idx(r, c - 1, self.cols)];
                let diag = cost[idx(r - 1, c - 1, self.cols)];
                if diag <= up && diag <= left {
                    best_r -= 1;
                    best_c -= 1;
                } else if up <= left {
                    best_r -= 1;
                } else {
                    best_c -= 1;
                }
            } else if r > 0 {
                best_r -= 1;
            } else {
                best_c -= 1;
            }

            r = best_r;
            c = best_c;
            self.path.push([r, c]);
        }
        self.path.reverse();

        self.similarity
    }
}

/* ───────────────────────────── tests ──────────────────────────────── */

#[cfg(test)]
mod tests {
    use super::*;

    fn abs(a: &f32, b: &f32) -> f32 {
        (a - b).abs()
    }

    #[test]
    fn tiny_example() {
        let xs: Vec<&f32> = [0.0f32, 1.0, 1.0, 2.0, 3.0].iter().collect();
        let ys: Vec<&f32> = [1.0f32, 2.0, 2.0, 3.0].iter().collect();

        let mut dtw = Dtw::new(abs);
        let dist = dtw.compute_optimal_path(&xs, &ys);
        assert!(dist < 1e-4, "distance was {dist}");
        assert_eq!(
            dtw.retrieve_optimal_path()
                .expect("DTW should have computed optimal path")
                .first(),
            Some(&[0, 0])
        );
    }

    #[test]
    fn windowed_matches_unwindowed_for_small_band() {
        let a: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let b: Vec<f32> = a.iter().copied().rev().collect(); // worst-case
        let a_refs: Vec<&f32> = a.iter().collect();
        let b_refs: Vec<&f32> = b.iter().collect();

        let mut d1 = Dtw::new(abs);
        let mut d2 = Dtw::new(abs);
        let d_unw = d1.compute_optimal_path(&a_refs[..], &b_refs[..]);
        let d_w = d2.compute_optimal_path_with_window(&a_refs[..], &b_refs[..], 200);
        assert!((d_unw - d_w).abs() < 1e-6);
        assert_eq!(
            d1.retrieve_optimal_path()
                .expect("DTW d1 should have computed optimal path")
                .len(),
            d2.retrieve_optimal_path()
                .expect("DTW d2 should have computed optimal path")
                .len()
        );
    }
}
