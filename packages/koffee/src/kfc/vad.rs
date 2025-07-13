//---
// path: potter-dsp/src/vad.rs
//---
use crate::config::VADMode;
use arrayvec::ArrayVec;

/// Simple energy-based voice-activity detector.
///
/// * `W` – number of past KFC-energy frames to track
/// * `COUNTDOWN` – frames to keep returning `true` after speech detected
///
/// ```no_run
/// # use potter_dsp::vad::VadDetector;
/// let mut vad = VadDetector::<50, 500>::new(VADMode::Medium);
/// if vad.is_voice(&kfc_frame) { /* gated-audio logic */ }
/// ```
#[allow(dead_code)]
pub struct VadDetector<const W: usize, const COUNTDOWN: usize> {
    mode: VADMode,
    index: usize,
    window: ArrayVec<f32, W>,
    voice_left: usize,
}

#[allow(dead_code)]
impl<const W: usize, const COUNTDOWN: usize> VadDetector<W, COUNTDOWN> {
    #[inline]
    pub fn new(mode: VADMode) -> Self {
        Self {
            mode,
            index: 0,
            window: ArrayVec::from([f32::NAN; W]),
            voice_left: 0,
        }
    }

    /// Reset internal state (e.g. when starting a new utterance).
    #[inline]
    pub fn reset(&mut self) {
        self.window.fill(f32::NAN);
        self.index = 0;
        self.voice_left = 0;
    }

    /// Return `true` if the current KFC frame is classified as voice.
    ///
    /// Uses the absolute-value average of the KFC vector as an energy
    /// proxy.  When **> threshold** for more than 10 frames in the ring
    /// buffer, it enters a “speech” state for `COUNTDOWN` frames.
    pub fn is_voice(&mut self, kfc: &[f32]) -> bool {
        if kfc.is_empty() {
            return false;
        }

        // Energy = mean(|coeff|)
        let e = kfc.iter().map(|v| v.abs()).sum::<f32>() / kfc.len() as f32;

        // Write into ring buffer
        self.window[self.index] = e;
        self.index = (self.index + 1) % W;

        // Compute min energy over valid entries
        let mut min_e = f32::INFINITY;
        let mut high = 0;
        for &v in self.window.iter() {
            if v.is_nan() {
                continue;
            }
            if v < min_e {
                min_e = v;
            }
            if v > 0.01 && v > min_e * self.mode.get_value() {
                high += 1;
            }
        }

        // Enter voice state if 10+ frames above threshold
        if high > 10 {
            self.voice_left = COUNTDOWN;
        }

        // Countdown
        let voiced = self.voice_left > 0;
        if voiced {
            self.voice_left -= 1;
        }
        voiced
    }
}
