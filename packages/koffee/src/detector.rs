//! Core compile-time constants for Kfc.
//!
//!  * All values are `pub` unless they’re strictly an implementation detail
//!    (then they stay `pub(crate)`).
//!  * “Magic numbers” are grouped logically with doc comments.
//!  * Derived constants are expressed with `const fn` to ensure the compiler
//!    checks arithmetic (no silent truncation).
//!  * Add NEW symbols introduced in the modernised code (`WAKEWORD_MODEL_VERSION`).

/* --------------------------------------------------------------------- */
/*  Global sample-rate & KFC layout                                      */

/// Internal mono PCM sample-rate (Hz).
pub const DETECTOR_INTERNAL_SAMPLE_RATE: usize = 16_000;

/// KFC frame length (milliseconds).
pub const KFC_FRAME_LEN_MS: usize = 30;

/// KFC hop / shift length (milliseconds).
pub const KFC_FRAME_SHIFT_MS: usize = 10;

/// Pre-emphasis coefficient (first-order high-pass).
pub const PRE_EMPHASIS: f32 = 0.97;

/// Number of shifts contained in one 30 ms frame.
pub const fn kfc_out_shifts() -> usize {
    KFC_FRAME_LEN_MS / KFC_FRAME_SHIFT_MS
}

/* --------------------------------------------------------------------- */
/*  Dynamic-time-warping comparator defaults                              */

/// Sakoe–Chiba band size used by DTW comparator.
pub(crate) const COMPARATOR_DEFAULT_BAND_SIZE: u16 = 5;

/// Logistic-probability baseline for *averaged* template comparison.
pub(crate) const DETECTOR_DEFAULT_AVG_THRESHOLD: f32 = 0.20;
/// Logistic-probability baseline for individual template frames.
pub(crate) const DETECTOR_DEFAULT_THRESHOLD: f32 = 0.50;
/// Minimum *partial* detections before a hit is accepted.
pub(crate) const DETECTOR_DEFAULT_MIN_SCORES: usize = 5;
/// Reference value to map DTW cost → 0-1 probability.
pub(crate) const DETECTOR_DEFAULT_REFERENCE: f32 = 0.22;

/* --------------------------------------------------------------------- */
/*  Neural-network model                                                  */

/// The label used for “non-wakeword” frames in NN models.
pub const NN_NONE_LABEL: &str = "none";

/// File-format version for `WakewordModel`.
/// Increment **whenever stored layout changes**.
pub const WAKEWORD_MODEL_VERSION: u8 = 1;

/* --------------------------------------------------------------------- */
/*  Re-exports for older call-sites (BC layer)                            */

pub use kfc_out_shifts as KFCS_EXTRACTOR_OUT_SHIFTS;
pub use KFC_FRAME_LEN_MS as KFCS_EXTRACTOR_FRAME_LENGTH_MS;
pub use KFC_FRAME_SHIFT_MS as KFCS_EXTRACTOR_FRAME_SHIFT_MS;
pub use PRE_EMPHASIS as KFCS_EXTRACTOR_PRE_EMPHASIS;
