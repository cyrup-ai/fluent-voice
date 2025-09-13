use crate::{KoffeeCandleDetection, ScoreMode};

pub(crate) trait WakewordDetector: Send + Sync {
    /// New unified getter – returns (coeffs , frames).
    #[allow(dead_code)]
    fn get_kfc_dimensions(&self) -> (u16, usize);

    /* ------------------------------------------------------------- *
     *  Back-compat shims – keep the old API signature so existing   *
     *  crates compile, but forward to the canonical method. *
     * ------------------------------------------------------------- */
    #[allow(dead_code)]
    #[inline(always)]
    fn get_kfc_size(&self) -> u16 {
        self.get_kfc_dimensions().0
    }
    #[allow(dead_code)]
    #[inline(always)]
    fn get_kfc_frame_size(&self) -> usize {
        self.get_kfc_dimensions().1
    }
    fn run_detection(
        &self,
        kfc_frame: Vec<Vec<f32>>,
        avg_threshold: f32,
        threshold: f32,
    ) -> Option<KoffeeCandleDetection>;
    #[allow(dead_code)]
    fn get_rms_level(&self) -> f32;
    fn update_config(&mut self, score_ref: f32, band_size: u16, score_mode: ScoreMode);
}
