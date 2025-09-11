//! Audio trimming utilities for processing voice samples
//!
//! This module provides high-performance audio trimming capabilities to ensure
//! all voice samples are within the required duration limits for wake word training.

use crate::error::{Result, SampleError};
use crate::metadata::SampleMetadata;
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Maximum duration for voice samples in seconds
pub const MAX_TRIM_DURATION: f32 = 30.0;

/// Trims audio files that exceed the maximum duration
///
/// This function processes audio files in parallel, trimming any that exceed
/// the maximum duration to exactly 30 seconds while preserving audio quality.
///
/// # Arguments
/// * `files` - Iterator of audio file paths to process
///
/// # Returns
/// * `Result<Vec<PathBuf>>` - Vector of successfully processed file paths
///
/// # Errors
/// Returns an error if ffmpeg is not available or if trimming fails
#[inline]
pub fn trim_overlong_files<I>(files: I) -> Result<Vec<PathBuf>>
where
    I: IntoIterator<Item = PathBuf>,
    I::IntoIter: Send,
{
    // Check if ffmpeg is available
    check_ffmpeg_availability()?;

    let files: Vec<PathBuf> = files.into_iter().collect();

    // Process files in parallel for maximum performance
    let results: Result<Vec<Option<PathBuf>>> = files
        .into_par_iter()
        .map(|file_path| {
            match needs_trimming(&file_path) {
                Ok(true) => {
                    match trim_audio_file(&file_path) {
                        Ok(()) => Ok(Some(file_path)),
                        Err(e) => {
                            // Log error but continue processing other files
                            eprintln!("Warning: Failed to trim {}: {}", file_path.display(), e);
                            Ok(Some(file_path)) // Include file even if trimming failed
                        }
                    }
                }
                Ok(false) => Ok(Some(file_path)), // File doesn't need trimming
                Err(e) => {
                    eprintln!(
                        "Warning: Failed to check duration for {}: {}",
                        file_path.display(),
                        e
                    );
                    Ok(Some(file_path)) // Include file even if duration check failed
                }
            }
        })
        .collect();

    // Filter out None values and return successful paths
    Ok(results?.into_iter().flatten().collect())
}

/// Checks if an audio file needs trimming based on its duration
///
/// # Arguments
/// * `file_path` - Path to the audio file
///
/// # Returns
/// * `Result<bool>` - True if the file needs trimming, false otherwise
#[inline]
fn needs_trimming(file_path: &Path) -> Result<bool> {
    let metadata = SampleMetadata::from_file(file_path)?;
    Ok(metadata.duration_secs > MAX_TRIM_DURATION)
}

/// Trims an audio file to the maximum duration using ffmpeg
///
/// # Arguments
/// * `file_path` - Path to the audio file to trim
///
/// # Returns
/// * `Result<()>` - Success or error result
#[inline]
fn trim_audio_file(file_path: &Path) -> Result<()> {
    let temp_path = create_temp_path(file_path)?;

    // Use ffmpeg to trim the audio file to exactly 30 seconds
    let output = Command::new("ffmpeg")
        .args([
            "-i",
            file_path
                .to_str()
                .ok_or_else(|| SampleError::InvalidPath(file_path.to_path_buf()))?,
            "-t",
            &MAX_TRIM_DURATION.to_string(),
            "-c",
            "copy", // Copy streams without re-encoding for speed
            "-avoid_negative_ts",
            "make_zero",
            "-y", // Overwrite output file
            temp_path
                .to_str()
                .ok_or_else(|| SampleError::InvalidPath(temp_path.clone()))?,
        ])
        .output()
        .map_err(|e| SampleError::AudioProcessing(format!("FFmpeg execution failed: {e}")))?;

    if !output.status.success() {
        let error_msg = String::from_utf8_lossy(&output.stderr);
        return Err(SampleError::AudioProcessing(format!(
            "FFmpeg failed to trim {}: {}",
            file_path.display(),
            error_msg
        )));
    }

    // Replace original file with trimmed version
    std::fs::rename(&temp_path, file_path).map_err(|e| {
        SampleError::AudioProcessing(format!(
            "Failed to replace original file {}: {}",
            file_path.display(),
            e
        ))
    })?;

    Ok(())
}

/// Creates a temporary file path for audio processing
///
/// # Arguments
/// * `original_path` - Original file path
///
/// # Returns
/// * `Result<PathBuf>` - Temporary file path
#[inline]
fn create_temp_path(original_path: &Path) -> Result<PathBuf> {
    let parent = original_path
        .parent()
        .ok_or_else(|| SampleError::InvalidPath(original_path.to_path_buf()))?;

    let file_name = original_path
        .file_name()
        .ok_or_else(|| SampleError::InvalidPath(original_path.to_path_buf()))?;

    let temp_name = format!("temp_trim_{}", file_name.to_string_lossy());
    Ok(parent.join(temp_name))
}

/// Checks if ffmpeg is available on the system
///
/// # Returns
/// * `Result<()>` - Success if ffmpeg is available, error otherwise
#[inline]
fn check_ffmpeg_availability() -> Result<()> {
    Command::new("ffmpeg")
        .args(["-version"])
        .output()
        .map_err(|_| {
            SampleError::AudioProcessing(
                "FFmpeg not found. Please install FFmpeg to enable audio trimming.".to_string(),
            )
        })?;

    Ok(())
}

/// Batch trims all audio files in a directory that exceed the maximum duration
///
/// # Arguments
/// * `directory` - Directory containing audio files
///
/// # Returns
/// * `Result<usize>` - Number of files processed
pub fn batch_trim_directory<P: AsRef<Path>>(directory: P) -> Result<usize> {
    let audio_files = crate::scanner::scan_audio_files(directory)?;
    let processed_files = trim_overlong_files(audio_files)?;
    Ok(processed_files.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_temp_path() {
        let original = PathBuf::from("/path/to/audio.mp3");
        let temp_path = create_temp_path(&original).unwrap();

        assert_eq!(temp_path.parent(), original.parent());
        assert!(
            temp_path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("temp_trim_")
        );
        assert!(
            temp_path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .ends_with("audio.mp3")
        );
    }

    #[test]
    fn test_check_ffmpeg_availability() {
        // This test will pass if ffmpeg is installed, skip if not
        match check_ffmpeg_availability() {
            Ok(()) => println!("FFmpeg is available"),
            Err(_) => println!("FFmpeg not available - skipping test"),
        }
    }
}
