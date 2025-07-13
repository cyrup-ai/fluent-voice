/// WAKEWORD_MODEL_VERSION represents the version of the wake word model.
pub const WAKEWORD_MODEL_VERSION: u8 = 1;

/// DETECTOR_INTERNAL_SAMPLE_RATE is the internal sample rate used by the detector.
pub const DETECTOR_INTERNAL_SAMPLE_RATE: usize = 16000;

/// KFCS_EXTRACTOR_FRAME_LENGTH_MS is the length of a frame in milliseconds for the KFCS extractor.
pub const KFCS_EXTRACTOR_FRAME_LENGTH_MS: usize = 30;

/// COMPARATOR_DEFAULT_BAND_SIZE is the default band size used by the comparator.
pub(crate) const COMPARATOR_DEFAULT_BAND_SIZE: u16 = 5;

/// DETECTOR_DEFAULT_AVG_THRESHOLD is the default average threshold used by the detector.
pub(crate) const DETECTOR_DEFAULT_AVG_THRESHOLD: f32 = 0.2;

/// DETECTOR_DEFAULT_THRESHOLD is the default threshold used by the detector.
pub(crate) const DETECTOR_DEFAULT_THRESHOLD: f32 = 0.5;

/// DETECTOR_DEFAULT_MIN_SCORES is the default minimum number of scores used by the detector.
pub(crate) const DETECTOR_DEFAULT_MIN_SCORES: usize = 5;

/// DETECTOR_DEFAULT_REFERENCE is the default reference value used by the detector.
pub(crate) const DETECTOR_DEFAULT_REFERENCE: f32 = 0.22;

/// KFCS_EXTRACTOR_FRAME_SHIFT_MS is the frame shift in milliseconds for the KFCS extractor.
pub(crate) const KFCS_EXTRACTOR_FRAME_SHIFT_MS: usize = 10;

/// KFCS_EXTRACTOR_PRE_EMPHASIS is the pre-emphasis value used by the KFCS extractor.
#[allow(dead_code)]
pub(crate) const KFCS_EXTRACTOR_PRE_EMPHASIS: f32 = 0.97;
/// KFCS_EXTRACTOR_OUT_SHIFTS calculates the number of output shifts for the KFCS extractor.
pub(crate) const KFCS_EXTRACTOR_OUT_SHIFTS: usize =
    KFCS_EXTRACTOR_FRAME_LENGTH_MS / KFCS_EXTRACTOR_FRAME_SHIFT_MS;

/// NN_NONE_LABEL is the label used for non-detection cases in neural network processing.
pub(crate) const NN_NONE_LABEL: &str = "none";

// KFC (Koffee-Candle) constants - same as KFC but with new naming

/// Number of KFC coefficients per frame (MFCC-like features).
pub const KFC_COEFFS: usize = 13;

/// Number of KFC frames in a detection window.
pub const KFC_FRAMES: usize = 32;
