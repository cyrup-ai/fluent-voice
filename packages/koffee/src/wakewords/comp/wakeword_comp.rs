use std::{collections::HashMap, sync::Arc};

use crate::{
    KoffeeCandleDetection, ScoreMode, WakewordRef,
    kfc::{KfcComparator, KfcNormalizer},
    wakewords::WakewordDetector,
};

/// Comparator for reference-style wake-word files (`.rpw`).
///
/// The heavy, immutable feature matrices are wrapped in `Arc` so that
/// cloning the detector costs O(1).
#[allow(dead_code)]
pub(crate) struct WakewordComparator {
    name: String,
    avg_features: Option<Arc<Vec<Vec<f32>>>>,
    samples: HashMap<String, Arc<Vec<Vec<f32>>>>,
    threshold: f32,
    avg_threshold: f32,
    rms_level: f32,

    // runtime state
    score_mode: ScoreMode,
    kfc_cmp: KfcComparator,
}

#[allow(dead_code)]
impl WakewordComparator {
    /// Build from a [`WakewordRef`].
    pub fn new(w: &WakewordRef, cmp: KfcComparator, mode: ScoreMode) -> Self {
        Self {
            name: w.name.clone(),
            avg_features: w.avg_features.clone().map(Arc::new),
            samples: w
                .samples_features
                .iter()
                .map(|(k, v)| (k.clone(), Arc::new(v.clone())))
                .collect(),
            threshold: w.threshold.unwrap_or(0.5),
            avg_threshold: w.avg_threshold.unwrap_or(0.0),
            rms_level: w.rms_level,
            score_mode: mode,
            kfc_cmp: cmp,
        }
    }

    /// Normalise & truncate an incoming frame window to `len` frames.
    #[inline]
    fn normalise(&self, kfc: &[Vec<f32>], len: usize) -> Vec<Vec<f32>> {
        let mut window = kfc[..len.min(kfc.len())].to_vec();
        KfcNormalizer::normalize(&mut window);
        window
    }

    /// Score one template against `frame` (already normalised).
    /// This version takes the comparator as a parameter to avoid mutable self reference.
    #[inline]
    fn score_with_comparator(
        comparator: &mut KfcComparator,
        frame: &[Vec<f32>],
        tmpl: &[Vec<f32>],
    ) -> f32 {
        comparator.compare(
            &tmpl.iter().map(|f| &f[..]).collect::<Vec<_>>(),
            &frame.iter().map(|f| &f[..]).collect::<Vec<_>>(),
        )
    }

    /// Select a unified score from all per-template scores.
    fn aggregate(&self, mut v: Vec<f32>) -> f32 {
        match self.score_mode {
            ScoreMode::Average => v.iter().sum::<f32>() / v.len() as f32,
            ScoreMode::Max => v.into_iter().fold(f32::MIN, f32::max),
            ScoreMode::Median | ScoreMode::P50 => percentile(&mut v, 50.0),
            ScoreMode::P25 => percentile(&mut v, 25.0),
            ScoreMode::P75 => percentile(&mut v, 75.0),
            ScoreMode::P80 => percentile(&mut v, 80.0),
            ScoreMode::P90 => percentile(&mut v, 90.0),
            ScoreMode::P95 => percentile(&mut v, 95.0),
            ScoreMode::Classic => v.iter().sum::<f32>() / v.len() as f32, // Use average for classic mode
        }
    }
}

/* -------- WakewordDetector impl ---------------------------------------- */

impl WakewordDetector for WakewordComparator {
    fn get_kfc_dimensions(&self) -> (u16, usize) {
        // Get the coefficients per frame (width of the first frame in the first sample)
        let coeffs = self
            .samples
            .values()
            .next()
            .and_then(|v| v.first().map(|frame| frame.len()))
            .unwrap_or(0) as u16;

        // Get the max number of frames across all samples
        let frames = self.samples.values().map(|v| v.len()).max().unwrap_or(0);

        (coeffs, frames)
    }

    fn run_detection(
        &self,
        kfc_frame: Vec<Vec<f32>>,
        cfg_avg_th: f32,
        cfg_th: f32,
    ) -> Option<KoffeeCandleDetection> {
        // 1) optional average template gating
        if let Some(avg) = &self.avg_features {
            let mut cmp = self.kfc_cmp.clone();
            let norm = self.normalise(&kfc_frame, avg.len());
            let s = cmp.compare(
                &avg.iter().map(|f| &f[..]).collect::<Vec<_>>(),
                &norm.iter().map(|f| &f[..]).collect::<Vec<_>>(),
            );
            if s < self.avg_threshold.max(cfg_avg_th) {
                return None;
            }
        }

        // 2) per-template scores
        // Clone the comparator once for the template scoring
        let cmp = self.kfc_cmp.clone();
        let scores: HashMap<String, f32> = self
            .samples
            .iter()
            .map(|(name, tmpl)| {
                let norm = self.normalise(&kfc_frame, tmpl.len());
                let mut local_cmp = cmp.clone(); // Create a local clone for each iteration
                (
                    name.clone(),
                    local_cmp.compare(
                        &tmpl.iter().map(|f| &f[..]).collect::<Vec<_>>(),
                        &norm.iter().map(|f| &f[..]).collect::<Vec<_>>(),
                    ),
                )
            })
            .collect();

        let score = self.aggregate(scores.values().copied().collect());
        if score <= self.threshold.max(cfg_th) {
            return None;
        }

        Some(KoffeeCandleDetection {
            name: self.name.clone(),
            avg_score: 0.0, // not used for comparator
            score,
            scores,
            counter: usize::MIN,
            gain: f32::NAN,
        })
    }

    fn get_rms_level(&self) -> f32 {
        self.rms_level
    }

    // Legacy getters now provided by the trait's default implementations

    fn update_config(&mut self, score_ref: f32, band: u16, mode: ScoreMode) {
        self.score_mode = mode;
        self.kfc_cmp = KfcComparator::new(score_ref, band);
    }
}

/* -------- helpers ------------------------------------------------------ */

#[inline]
#[allow(dead_code)]
fn percentile(v: &mut [f32], p: f32) -> f32 {
    v.sort_by(|a, b| a.total_cmp(b));
    let pos = p / 100.0 * (v.len() - 1) as f32;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        v[lo]
    } else {
        let d = pos - lo as f32;
        v[lo] * (1.0 - d) + v[hi] * d
    }
}
