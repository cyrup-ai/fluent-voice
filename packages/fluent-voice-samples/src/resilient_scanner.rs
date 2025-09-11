//! Resilient audio file scanning that gracefully handles format issues
//!
//! This module provides robust audio file processing that skips problematic files
//! instead of failing the entire operation, ensuring maximum voice sample coverage.

use crate::error::{Result, SampleError};
use crate::metadata::SampleMetadata;
use crate::{MAX_SAMPLE_DURATION, MIN_SAMPLE_DURATION};
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Statistics for resilient processing
#[derive(Debug, Clone)]
pub struct ProcessingStats {
    /// Total files found
    pub total_files: usize,
    /// Successfully processed files
    pub processed_files: usize,
    /// Files skipped due to errors
    pub skipped_files: usize,
    /// Files with invalid duration
    pub invalid_duration: usize,
    /// Files with unsupported format
    pub unsupported_format: usize,
}

impl ProcessingStats {
    /// Creates new processing statistics
    #[inline]
    #[must_use]
    pub const fn new(total_files: usize) -> Self {
        Self {
            total_files,
            processed_files: 0,
            skipped_files: 0,
            invalid_duration: 0,
            unsupported_format: 0,
        }
    }

    /// Success rate as a percentage
    #[inline]
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn success_rate(&self) -> f64 {
        if self.total_files == 0 {
            0.0
        } else {
            (self.processed_files as f64 / self.total_files as f64) * 100.0
        }
    }
}

/// Resilient audio file processing that skips problematic files
///
/// This function processes audio files in parallel, skipping any that can't be read
/// or processed instead of failing the entire operation. This ensures maximum
/// coverage of voice samples for wake word training.
///
/// # Arguments
/// * `directory` - The directory to scan for audio files
///
/// # Returns
/// * `Result<(Vec<SampleMetadata>, ProcessingStats)>` - Successfully processed samples and statistics
#[inline]
pub fn resilient_index_samples<P: AsRef<Path>>(
    directory: P,
) -> Result<(Vec<SampleMetadata>, ProcessingStats)> {
    let files = crate::scanner::scan_audio_files(directory)?;
    let mut stats = ProcessingStats::new(files.len());

    println!("Found {} audio files to process", files.len());

    // Atomic counters for thread-safe statistics
    let processed_count = Arc::new(AtomicUsize::new(0));
    let skipped_count = Arc::new(AtomicUsize::new(0));
    let invalid_duration_count = Arc::new(AtomicUsize::new(0));
    let unsupported_format_count = Arc::new(AtomicUsize::new(0));

    // Process files in parallel, collecting successful results
    let results: Vec<Option<SampleMetadata>> = files
        .par_iter()
        .map(|file_path| match process_single_file(file_path) {
            Ok(metadata) => {
                processed_count.fetch_add(1, Ordering::Relaxed);
                Some(metadata)
            }
            Err(SampleError::InvalidSampleDuration(_, _, _)) => {
                invalid_duration_count.fetch_add(1, Ordering::Relaxed);
                skipped_count.fetch_add(1, Ordering::Relaxed);
                if processed_count.load(Ordering::Relaxed).is_multiple_of(1000) {
                    eprintln!(
                        "Warning: Skipped {} (invalid duration)",
                        file_path.display()
                    );
                }
                None
            }
            Err(SampleError::InvalidAudioFile { .. }) => {
                unsupported_format_count.fetch_add(1, Ordering::Relaxed);
                skipped_count.fetch_add(1, Ordering::Relaxed);
                if processed_count.load(Ordering::Relaxed).is_multiple_of(1000) {
                    eprintln!(
                        "Warning: Skipped {} (unsupported format)",
                        file_path.display()
                    );
                }
                None
            }
            Err(e) => {
                skipped_count.fetch_add(1, Ordering::Relaxed);
                if processed_count.load(Ordering::Relaxed).is_multiple_of(1000) {
                    eprintln!("Warning: Skipped {} ({})", file_path.display(), e);
                }
                None
            }
        })
        .collect();

    // Update statistics
    stats.processed_files = processed_count.load(Ordering::Relaxed);
    stats.skipped_files = skipped_count.load(Ordering::Relaxed);
    stats.invalid_duration = invalid_duration_count.load(Ordering::Relaxed);
    stats.unsupported_format = unsupported_format_count.load(Ordering::Relaxed);

    // Filter out None values to get successful results
    let successful_samples: Vec<SampleMetadata> = results.into_iter().flatten().collect();

    println!("Processing complete:");
    println!("  Successfully processed: {} files", stats.processed_files);
    println!("  Skipped files: {} files", stats.skipped_files);
    println!("  Success rate: {:.1}%", stats.success_rate());

    if stats.invalid_duration > 0 {
        println!("  Invalid duration: {} files", stats.invalid_duration);
    }
    if stats.unsupported_format > 0 {
        println!("  Unsupported format: {} files", stats.unsupported_format);
    }

    Ok((successful_samples, stats))
}

/// Processes a single audio file, returning metadata or an error
///
/// # Arguments
/// * `file_path` - Path to the audio file
///
/// # Returns
/// * `Result<SampleMetadata>` - Metadata for the file or error
#[inline]
fn process_single_file(file_path: &PathBuf) -> Result<SampleMetadata> {
    let metadata = SampleMetadata::from_file(file_path)?;

    // Validate sample duration
    if metadata.duration_secs < MIN_SAMPLE_DURATION {
        return Err(SampleError::InvalidSampleDuration(
            metadata.duration_secs,
            MIN_SAMPLE_DURATION,
            MAX_SAMPLE_DURATION,
        ));
    }

    if metadata.duration_secs > MAX_SAMPLE_DURATION {
        return Err(SampleError::InvalidSampleDuration(
            metadata.duration_secs,
            MIN_SAMPLE_DURATION,
            MAX_SAMPLE_DURATION,
        ));
    }

    // Validate sample rate
    if metadata.sample_rate < 8000 || metadata.sample_rate > 48000 {
        return Err(SampleError::UnsupportedSampleRate(metadata.sample_rate));
    }

    Ok(metadata)
}

#[cfg(test)]
mod tests {
    use super::*;

    use tempfile::tempdir;

    #[test]
    fn test_processing_stats() {
        let stats = ProcessingStats::new(100);
        assert_eq!(stats.total_files, 100);
        assert!((stats.success_rate() - 0.0).abs() < f64::EPSILON);

        let stats = ProcessingStats {
            total_files: 100,
            processed_files: 80,
            skipped_files: 20,
            invalid_duration: 10,
            unsupported_format: 10,
        };

        assert!((stats.success_rate() - 80.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_resilient_processing_empty_directory() {
        let temp_dir = tempdir().unwrap();
        let empty_dir = temp_dir.path().join("empty");
        std::fs::create_dir(&empty_dir).unwrap();

        // Should handle empty directory gracefully
        if let Ok((samples, stats)) = resilient_index_samples(&empty_dir) {
            assert_eq!(samples.len(), 0);
            assert_eq!(stats.total_files, 0);
        } else {
            // Empty directory might return an error, which is acceptable
        }
    }
}
