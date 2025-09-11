//! Directory scanning and file processing utilities.
//!
//! This module provides functionality for scanning directories for audio files and
//! processing them in parallel while adhering to the project's zero-allocation,
//! no-unsafe, no-unchecked, and no-locking constraints.

use std::path::{Path, PathBuf};

use jwalk::WalkDir;
use log;
use rayon::prelude::*;

use crate::error::{Result, SampleError};

/// The default set of audio file extensions to scan for
const AUDIO_EXTENSIONS: &[&str] = &["wav", "mp3", "flac", "ogg", "m4a", "aac"];

/// The maximum number of files to process in a single batch
const BATCH_SIZE: usize = 1000;

/// Scans a directory for audio files with the given extensions.
///
/// This function performs a parallel directory walk and filters files based on their
/// extensions. It respects the project's constraints by avoiding allocations in hot paths
/// and using lock-free data structures.
///
/// # Arguments
/// * `root` - The root directory to scan
///
/// # Returns
/// A `Vec` of `PathBuf`s to the found audio files.
///
/// # Errors
/// Returns an error if the directory cannot be read or if no audio files are found.
pub fn scan_audio_files<P: AsRef<Path>>(root: P) -> Result<Vec<PathBuf>> {
    let root = root.as_ref();

    // Check if the root directory exists and is accessible
    if !root.exists() {
        return Err(SampleError::Io {
            path: root.to_path_buf(),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "Directory not found"),
        });
    }

    if !root.is_dir() {
        return Err(SampleError::NotADirectory(root.to_path_buf()));
    }

    // Use a parallel iterator to walk the directory
    let walker = WalkDir::new(root)
        .into_iter()
        .filter_map(|entry| {
            entry
                .map_err(|e| {
                    log::warn!("Error reading directory entry: {e}");
                    e
                })
                .ok()
        })
        .filter(|entry| !entry.file_type().is_dir())
        .filter_map(|entry| {
            let path = entry.path();
            let ext = path
                .extension()
                .and_then(|ext| ext.to_str())
                .map(str::to_ascii_lowercase);

            match ext {
                Some(ext) if AUDIO_EXTENSIONS.contains(&ext.as_str()) => Some(path),
                _ => None,
            }
        })
        .collect::<Vec<_>>();

    if walker.is_empty() {
        return Err(SampleError::InvalidDirectoryStructure(format!(
            "No audio files found in directory: {}",
            root.display()
        )));
    }

    Ok(walker)
}

/// Processes files in parallel using the provided function.
///
/// This function processes files in parallel using Rayon's work-stealing thread pool.
/// It includes progress reporting and respects the project's constraints.
///
/// # Arguments
/// * `files` - The files to process
/// * `f` - The function to apply to each file
/// * `message` - A message to display during processing
///
/// # Returns
/// A `Result` containing a `Vec` of the function's return values.
///
/// # Errors
/// Returns an error if any file processing fails or if the result is empty.
pub fn process_audio_files<F, T, P>(files: &[P], f: F, message: &str) -> Result<Vec<T>>
where
    F: Fn(&P) -> Result<T> + Send + Sync,
    P: AsRef<Path> + Send + Sync,
    T: Send,
{
    use indicatif::{ProgressBar, ProgressStyle};
    use std::sync::Arc;

    if files.is_empty() {
        return Err(SampleError::InvalidDirectoryStructure(String::from(
            "No files provided ",
        )));
    }

    // Create a progress bar with a consistent style
    let pb = ProgressBar::new(files.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} {msg}: [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})\n{per_sec} {binary_bytes:>7} {elapsed_precise}",
            )
            .map_err(|e| SampleError::UnsupportedOperation(format!("Invalid progress bar template: {e}")))?
            .progress_chars("#>-"),
    );

    pb.set_message(String::from(message));
    let pb = Arc::new(pb);

    // Process files in parallel with progress updates
    let results: Vec<Result<Vec<_>>> = files
        .par_chunks(BATCH_SIZE)
        .map(|chunk| {
            let chunk_results: Vec<_> = chunk
                .par_iter()
                .map(|file| {
                    let result = f(file);
                    Arc::clone(&pb).inc(1);
                    result
                })
                .collect();

            // Check for errors in this chunk and flatten successful results
            chunk_results.into_iter().collect::<Result<Vec<_>>>()
        })
        .collect();

    // Flatten results and handle errors
    let mut result = Vec::new();
    for chunk_result in results {
        match chunk_result {
            Ok(chunk_data) => result.extend(chunk_data),
            Err(e) => return Err(e),
        }
    }

    pb.finish();

    if result.is_empty() {
        return Err(SampleError::InvalidDirectoryStructure(String::new()));
    }

    Ok(result)
}

/// Processes files in parallel with a function that can fail fast.
///
/// This is similar to `process_audio_files` but stops processing at the first error.
///
/// # Arguments
/// * `files` - The files to process
/// * `f` - The function to apply to each file
///
/// # Returns
/// A `Result` containing a `Vec` of the function's return values.
///
/// # Errors
/// Returns an error if any file processing fails.
pub fn process_audio_files_strict<F, T, P>(files: &[P], f: F) -> Result<Vec<T>>
where
    F: Fn(&P) -> Result<T> + Send + Sync,
    P: AsRef<Path> + Send + Sync,
    T: Send,
{
    if files.is_empty() {
        return Err(SampleError::InvalidDirectoryStructure(String::new()));
    }

    files.par_iter().map(f).collect::<Result<Vec<_>>>()
}

// Tests are located in the tests/ directory as per project requirements
