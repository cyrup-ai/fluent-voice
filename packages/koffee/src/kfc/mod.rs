mod averager;
mod comparator;
mod dtw;
mod extractor;
mod normalizer;
mod vad;
mod wav_file_extractor;
pub(crate) use averager::KfcAverager;
pub(crate) use comparator::KfcComparator;
pub(crate) use extractor::KfcExtractor;
pub(crate) use normalizer::KfcNormalizer;
pub(crate) use vad::VadDetector;
pub(crate) use wav_file_extractor::{ExtractorError, KfcWavFileExtractor};

// Re-export a common Error type for kfc-related operations
pub use extractor::KfcError as Error;
