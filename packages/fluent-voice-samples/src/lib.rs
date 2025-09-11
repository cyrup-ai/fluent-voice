//! `fluent-voice-samples` - A library for managing and processing voice samples
//! for wake word training.
//!
//! This library provides utilities for:
//! - Scanning directories for audio files
//! - Extracting and validating audio metadata
//! - Managing sample collections
//! - Generating and validating directory structures

#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]
#![forbid(unsafe_code)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc
)]

use std::path::Path;

pub mod dirs;
pub mod error;
pub mod metadata;
/// Progress tracking utilities for long-running operations.
pub mod progress;
pub mod resilient_scanner;
pub mod sample;
pub mod scanner;
pub mod trimmer;

// Re-exports
pub use dirs::{
    cleanup_malformed_dirs, create_module_dir, sanitize_dir_name, validate_module_dir,
    validate_module_structure,
};
pub use error::{Result, SampleError};
pub use metadata::SampleMetadata;
pub use resilient_scanner::{ProcessingStats, resilient_index_samples};
pub use scanner::{process_audio_files, scan_audio_files};
pub use trimmer::{MAX_TRIM_DURATION, batch_trim_directory, trim_overlong_files};

/// The default sample rate for audio processing (16kHz)
pub const DEFAULT_SAMPLE_RATE: u32 = 16000;

/// The minimum duration of a valid sample in seconds
pub const MIN_SAMPLE_DURATION: f32 = 0.5;

/// The maximum duration of a valid sample in seconds (with tolerance for floating-point precision)
pub const MAX_SAMPLE_DURATION: f32 = 30.1;

/// Scans a directory for audio files and returns their metadata
///
/// # Arguments
/// * `directory` - The directory to scan for audio files
///
/// # Errors
/// Returns an error if the directory cannot be read or if any audio file is invalid.
#[inline]
pub fn index_samples<P: AsRef<Path>>(directory: P) -> Result<Vec<SampleMetadata>> {
    let files = scan_audio_files(directory)?;

    process_audio_files(
        &files,
        |path| {
            let metadata = SampleMetadata::from_file(path)?;

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

            // Validate sample rate (convert to Hz)
            if metadata.sample_rate < 8000 || metadata.sample_rate > 48000 {
                return Err(SampleError::UnsupportedSampleRate(metadata.sample_rate));
            }

            Ok(metadata)
        },
        "Indexing audio samples",
    )
}

/// Exports sample metadata to a YAML file
///
/// # Arguments
/// * `samples` - A slice of `SampleMetadata` to export
/// * `output_path` - The path to write the YAML file to
///
/// # Errors
/// Returns an error if the file cannot be created or written to.
pub fn export_metadata<P: AsRef<Path>>(samples: &[SampleMetadata], output_path: P) -> Result<()> {
    use std::fs::File;
    use std::io::Write;

    let yaml = serde_yaml::to_string(samples)?;

    let mut file = File::create(&output_path).map_err(|e| SampleError::Io {
        path: output_path.as_ref().to_path_buf(),
        source: e,
    })?;

    file.write_all(yaml.as_bytes())
        .map_err(|e| SampleError::Io {
            path: output_path.as_ref().to_path_buf(),
            source: e,
        })?;

    Ok(())
}

/// Imports sample metadata from a YAML file
///
/// # Arguments
/// * `path` - The path to the YAML file to import
///
/// # Errors
/// Returns an error if the file cannot be read or parsed.
pub fn import_metadata<P: AsRef<Path>>(path: P) -> Result<Vec<SampleMetadata>> {
    use std::fs::File;
    use std::io::Read;

    let mut file = File::open(&path).map_err(|e| SampleError::Io {
        path: path.as_ref().to_path_buf(),
        source: e,
    })?;

    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .map_err(|e| SampleError::Io {
            path: path.as_ref().to_path_buf(),
            source: e,
        })?;

    serde_yaml::from_str(&contents).map_err(Into::into)
}

/// Validates the samples directory structure
///
/// # Arguments
/// * `root` - The root directory to validate
///
/// # Errors
/// Returns an error if the directory structure is invalid.
pub fn validate_samples_dir(root: &Path) -> Result<()> {
    let required_dirs = &[
        "additional_tts",
        "cartoon_voices",
        "celebrities",
        "female_voices",
        "male_voices",
        "news_broadcasts",
        "radio_comedy",
        "radio_drama",
        "radio_personalities",
        "sports_announcers",
        "sports_broadcasts",
    ];

    for dir in required_dirs {
        let path = root.join(dir);
        if !path.is_dir() {
            return Err(SampleError::InvalidDirectoryStructure(format!(
                "Missing required directory: {dir}"
            )));
        }
    }

    Ok(())
}

/// Initializes a new samples directory with the required structure
///
/// # Arguments
/// * `root` - The root directory to initialize
///
/// # Errors
/// Returns an error if the directory cannot be created or if any required
/// subdirectories cannot be created.
pub fn init_samples_dir<P: AsRef<Path>>(root: P) -> Result<()> {
    let required_dirs = &[
        "additional_tts",
        "cartoon_voices",
        "celebrities",
        "female_voices",
        "male_voices",
        "news_broadcasts",
        "radio_comedy",
        "radio_drama",
        "radio_personalities",
        "sports_announcers",
        "sports_broadcasts",
    ];

    for dir in required_dirs {
        create_module_dir(&root, dir)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_init_samples_dir() {
        let temp_dir = tempdir().unwrap();
        let samples_dir = temp_dir.path().join("samples");

        init_samples_dir(&samples_dir).unwrap();

        // Verify all required directories were created
        for dir in &[
            "additional_tts",
            "cartoon_voices",
            "celebrities",
            "female_voices",
            "male_voices",
            "news_broadcasts",
            "radio_comedy",
            "radio_drama",
            "radio_personalities",
            "sports_announcers",
            "sports_broadcasts",
        ] {
            let path = samples_dir.join(dir);
            assert!(path.is_dir(), "Directory not created: {dir}");
        }
    }

    #[test]
    fn test_validate_samples_dir() {
        let temp_dir = tempdir().unwrap();
        let samples_dir = temp_dir.path().join("samples");

        // Create required directories
        for dir in &[
            "additional_tts",
            "cartoon_voices",
            "celebrities",
            "female_voices",
            "male_voices",
            "news_broadcasts",
            "radio_comedy",
            "radio_drama",
            "radio_personalities",
            "sports_announcers",
            "sports_broadcasts",
        ] {
            std::fs::create_dir_all(samples_dir.join(dir)).unwrap();
        }

        // Should pass validation
        validate_samples_dir(&samples_dir).unwrap();

        // Remove a required directory and verify validation fails
        std::fs::remove_dir(samples_dir.join("radio_drama")).unwrap();
        assert!(validate_samples_dir(&samples_dir).is_err());
    }
}
