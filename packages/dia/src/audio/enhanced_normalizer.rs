// Enhanced loudness normaliser with broadcast presets.  This is the full
// implementation previously located at `crate::enhanced_normalizer`; it now
// lives under `crate::dsp` to keep all signal-processing helpers in one
// place.

use std::collections::VecDeque;

/// Loudness preset options with broadcast standards
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LoudnessPreset {
    /// YouTube and social media (-14 LUFS, -1.0 dBTP)
    YouTube,
    /// Broadcast standards like EBU R128 (-16 LUFS, -2.0 dBTP)
    Broadcast,
    /// Music streaming standards (-18 LUFS, -1.0 dBTP)
    Streaming,
    /// Podcast AES recommendation (-23 LUFS, -1.0 dBTP)
    Podcast,
    /// Voice chat optimisation (-14 LUFS, -0.5 dBTP)
    Voice,
    /// Telephony standard (-20 LUFS, -3.0 dBTP)
    Telephony,
    /// Custom user-defined settings
    Custom,
}

impl LoudnessPreset {
    /// Returns (target LUFS, true-peak limit dBFS)
    pub fn params(&self) -> (f64, f64) {
        match self {
            Self::YouTube => (-14.0, -1.0),
            Self::Broadcast => (-16.0, -2.0),
            Self::Streaming => (-18.0, -1.0),
            Self::Podcast => (-23.0, -1.0),
            Self::Voice => (-14.0, -0.5),
            Self::Telephony => (-20.0, -3.0),
            Self::Custom => (-16.0, -1.0), // sensible default
        }
    }
}

/// Block-based loudness normaliser aimed at real-time preview in the web UI.
pub struct EnhancedNormalizer {
    // configuration
    #[allow(dead_code)]
    sample_rate: usize,
    preset: LoudnessPreset,

    // metering state
    momentary_lufs: f64,
    integrated_lufs: Option<f64>,
    lufs_history: VecDeque<f64>,
    current_gain: f32,
    current_buffer: Vec<f32>,

    // processing params
    #[allow(dead_code)]
    block_size: usize,
    #[allow(dead_code)]
    overlap: usize,

    // advanced options
    use_true_peak_limiter: bool,
    use_dynamic_compression: bool,
    #[allow(dead_code)]
    attack_time_ms: f32,
    #[allow(dead_code)]
    release_time_ms: f32,
}

impl EnhancedNormalizer {
    /// Create a new instance with sensible defaults.
    pub fn new(sample_rate: usize) -> Self {
        Self {
            sample_rate,
            preset: LoudnessPreset::Voice,
            momentary_lufs: -60.0,
            integrated_lufs: None,
            lufs_history: VecDeque::with_capacity(100),
            current_gain: 1.0,
            current_buffer: Vec::new(),
            block_size: 400, // 50 ms at 8 kHz – UI demo only
            overlap: 200,
            use_true_peak_limiter: true,
            use_dynamic_compression: true,
            attack_time_ms: 5.0,
            release_time_ms: 50.0,
        }
    }

    /// Select a different preset.
    pub fn set_preset(&mut self, preset: LoudnessPreset) {
        self.preset = preset;
        self.update_gain();
    }

    /// Process a block in-place – returns the linear gain applied.
    pub fn process(&mut self, buffer: &mut [f32]) -> f32 {
        // keep a copy for the UI graphs
        self.current_buffer = buffer.to_vec();

        // measure loudness & update history
        self.analyse_loudness(buffer);

        // recalc gain & apply
        self.update_gain();
        for s in buffer.iter_mut() {
            *s *= self.current_gain;
        }

        if self.use_true_peak_limiter {
            self.apply_true_peak_limiting(buffer);
        }

        self.current_gain
    }

    // ───────────────────────── internal helpers ───────────────────────────

    fn update_gain(&mut self) {
        let target_lufs = self.preset.params().0;

        let measured = self.integrated_lufs.unwrap_or(self.momentary_lufs);
        let gain_db = target_lufs - measured;
        let mut gain = 10.0_f32.powf((gain_db as f32) / 20.0);

        if self.use_dynamic_compression {
            gain = self.apply_dynamic_compression(gain);
        }

        self.current_gain = gain;
    }

    fn analyse_loudness(&mut self, buffer: &[f32]) {
        // quick RMS – good enough for UI visualisation
        let sum_sq: f64 = buffer.iter().map(|&s| (s as f64).powi(2)).sum();
        let rms = (sum_sq / buffer.len() as f64).sqrt();
        let lufs = if rms > 0.0 {
            20.0 * rms.log10() - 0.691
        } else {
            -100.0
        };

        self.momentary_lufs = if self.momentary_lufs < -50.0 {
            lufs
        } else {
            0.9 * self.momentary_lufs + 0.1 * lufs
        };

        self.lufs_history.push_back(self.momentary_lufs);
        if self.lufs_history.len() > 100 {
            self.lufs_history.pop_front();
        }

        // crude integrated value after 1 s of data
        if self.lufs_history.len() > 10 {
            let gated: Vec<_> = self
                .lufs_history
                .iter()
                .copied()
                .filter(|v| *v > -70.0)
                .collect();
            if !gated.is_empty() {
                self.integrated_lufs =
                    Some(gated.iter().copied().sum::<f64>() / gated.len() as f64);
            }
        }
    }

    fn apply_true_peak_limiting(&self, buffer: &mut [f32]) {
        let tp_limit = self.preset.params().1;
        let tp_lin = 10.0_f32.powf(tp_limit as f32 / 20.0);
        for s in buffer.iter_mut() {
            if s.abs() > tp_lin {
                *s = tp_lin * s.signum();
            }
        }
    }

    fn apply_dynamic_compression(&self, base: f32) -> f32 {
        let diff = self.integrated_lufs.unwrap_or(self.momentary_lufs) - self.preset.params().0;
        if diff < -10.0 {
            base * (1.0 + ((-diff - 10.0) / 20.0).min(1.0) as f32)
        } else if diff > 0.0 {
            base * (1.0 - 0.3 * (diff / 10.0).min(1.0) as f32)
        } else {
            base
        }
    }

    // ───────────── getters for the Dioxus UI overlay ─────────────────────

    pub fn get_momentary_lufs(&self) -> f64 {
        self.momentary_lufs
    }
    pub fn get_integrated_lufs(&self) -> Option<f64> {
        self.integrated_lufs
    }
    pub fn get_current_gain(&self) -> f32 {
        self.current_gain
    }
    pub fn get_true_peak(&self, buffer: &[f32]) -> f32 {
        buffer.iter().map(|&s| s.abs()).fold(0.0, f32::max)
    }
    pub fn get_current_preset(&self) -> LoudnessPreset {
        self.preset
    }
    pub fn get_lufs_history(&self) -> Vec<f64> {
        self.lufs_history.iter().copied().collect()
    }
    pub fn get_current_buffer(&self) -> Vec<f32> {
        self.current_buffer.clone()
    }

    // ─── put this somewhere inside the impl EnhancedNormalizer { … } block ─────────

    /// Enable or disable the (very light) release-shaped peak-compression stage
    /// that follows the BS.1770 gain calculation.  _True-peak limiting_ is
    /// **not** affected – that one stays on unless you call `set_true_peak_limiter(false)`.
    pub fn set_compression(&mut self, enabled: bool) {
        self.use_dynamic_compression = enabled;
    }

    /// Optional companion if you also want to expose the limiter switch
    /// (nothing else in the code calls it, so this one is … well … optional).
    pub fn set_true_peak_limiter(&mut self, enabled: bool) {
        self.use_true_peak_limiter = enabled;
    }
}
