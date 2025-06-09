pub const WAKEWORD_MODEL_VERSION: u8 = 1;
pub const DETECTOR_INTERNAL_SAMPLE_RATE: usize = 16000;
pub const KFCS_EXTRACTOR_FRAME_LENGTH_MS: usize = 30;
pub(crate) const COMPARATOR_DEFAULT_BAND_SIZE: u16 = 5;
pub(crate) const DETECTOR_DEFAULT_AVG_THRESHOLD: f32 = 0.2;
pub(crate) const DETECTOR_DEFAULT_THRESHOLD: f32 = 0.5;
pub(crate) const DETECTOR_DEFAULT_MIN_SCORES: usize = 5;
pub(crate) const DETECTOR_DEFAULT_REFERENCE: f32 = 0.22;
pub(crate) const KFCS_EXTRACTOR_FRAME_SHIFT_MS: usize = 10;
#[allow(dead_code)]
pub(crate) const KFCS_EXTRACTOR_PRE_EMPHASIS: f32 = 0.97;
pub(crate) const KFCS_EXTRACTOR_OUT_SHIFTS: usize =
    KFCS_EXTRACTOR_FRAME_LENGTH_MS / KFCS_EXTRACTOR_FRAME_SHIFT_MS;
// ^ remains used by kfc/nn stack calculations – leave as pub(crate)
pub(crate) const NN_NONE_LABEL: &str = "none";

// KFC (Koffee-Candle) constants - same as KFC but with new naming
pub const KFC_COEFFS: usize = 13;
pub const KFC_FRAMES: usize = 32;
