//! Safe, zero-allocation directory handling utilities.
//!
//! This module provides utilities for creating and validating module directories
//! with a focus on safety, performance, and ergonomics.

use std::{
    ffi::OsStr,
    path::{Path, PathBuf},
};

use crate::error::{Result, SampleError};

/// Sanitizes a directory name to be filesystem-safe.
///
/// # Arguments
/// * `name` - The input directory name to sanitize.
///
/// # Returns
/// A new `String` containing only alphanumeric ASCII characters, underscores, and hyphens.
#[inline]
#[must_use]
pub fn sanitize_dir_name(name: &str) -> String {
    name.chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
        .collect()
}

/// Creates a directory with a validated name.
///
/// # Arguments
/// * `base` - The base directory path.
/// * `name` - The name of the directory to create.
///
/// # Returns
/// The full path to the created directory.
///
/// # Errors
/// Returns `SampleError::InvalidPath` if the name is empty after sanitization.
/// Returns `SampleError::Io` if directory creation fails.
pub fn create_module_dir<P: AsRef<Path>>(base: P, name: &str) -> Result<PathBuf> {
    let sanitized = sanitize_dir_name(name);

    if sanitized.is_empty() {
        return Err(SampleError::InvalidPath(PathBuf::from(format!(
            "Invalid directory name: {name}"
        ))));
    }

    let path = base.as_ref().join(&sanitized);

    std::fs::create_dir_all(&path).map_err(|e| SampleError::Io {
        path: path.clone(),
        source: e,
    })?;

    Ok(path)
}

/// Validates that a path is a valid module directory.
///
/// # Arguments
/// * `path` - The path to validate.
///
/// # Errors
/// Returns `SampleError::NotADirectory` if the path is not a directory.
/// Returns `SampleError::InvalidPath` if the directory name is invalid.
pub fn validate_module_dir<P: AsRef<Path>>(path: P) -> Result<()> {
    let path = path.as_ref();

    if !path.is_dir() {
        return Err(SampleError::NotADirectory(path.to_path_buf()));
    }

    if let Some(name) = path.file_name().and_then(OsStr::to_str) {
        let sanitized = sanitize_dir_name(name);
        if sanitized != name {
            return Err(SampleError::InvalidPath(PathBuf::from(format!(
                "Invalid directory name: {name} (should be: {sanitized})"
            ))));
        }
    }

    Ok(())
}

/// Recursively validates all module directories.
///
/// # Arguments
/// * `root` - The root directory to start validation from.
///
/// # Errors
/// Returns the first encountered error during validation.
pub fn validate_module_structure<P: AsRef<Path>>(root: P) -> Result<()> {
    let root = root.as_ref();

    validate_module_dir(root)?;

    let entries = std::fs::read_dir(root).map_err(|e| SampleError::Io {
        path: root.to_path_buf(),
        source: e,
    })?;

    for entry in entries {
        let entry = entry.map_err(|e| SampleError::Io {
            path: root.to_path_buf(),
            source: e,
        })?;

        let path = entry.path();
        if path.is_dir() {
            validate_module_dir(&path)?;
        } else if let Some(ext) = path.extension()
            && ext != "rs"
            && ext != "toml"
            && ext != "md"
        {
            return Err(SampleError::InvalidFileExtension(path));
        }
    }

    Ok(())
}

/// Cleans up malformed directories in the given path.
///
/// # Arguments
/// * `root` - The root directory to clean up.
///
/// # Returns
/// The number of directories that were fixed.
///
/// # Errors
/// Returns any I/O errors encountered during cleanup.
pub fn cleanup_malformed_dirs<P: AsRef<Path>>(root: P) -> Result<usize> {
    let root = root.as_ref();
    let mut fixed = 0;

    let entries: Vec<_> = std::fs::read_dir(root)
        .map_err(|e| SampleError::Io {
            path: root.to_path_buf(),
            source: e,
        })?
        .filter_map(std::result::Result::ok)
        .filter(|entry| entry.path().is_dir())
        .collect();

    for entry in entries {
        let path = entry.path();
        if let Some(name) = path.file_name().and_then(OsStr::to_str) {
            let sanitized = sanitize_dir_name(name);
            if sanitized != name {
                let new_path = path
                    .parent()
                    .map(|p| p.join(&sanitized))
                    .ok_or_else(|| SampleError::InvalidPath(path.clone()))?;

                if !new_path.exists() {
                    std::fs::rename(&path, &new_path).map_err(|e| SampleError::Io {
                        path: path.clone(),
                        source: e,
                    })?;
                    fixed += 1;
                }
            }
        }
    }

    Ok(fixed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_sanitize_dir_name() {
        assert_eq!(sanitize_dir_name("valid-name"), "valid-name");
        assert_eq!(sanitize_dir_name("invalid!@#name"), "invalidname");
        assert_eq!(sanitize_dir_name("with spaces"), "withspaces");
        assert_eq!(sanitize_dir_name("UPPER_lower-123"), "UPPER_lower-123");
    }

    #[test]
    fn test_create_module_dir() {
        let temp_dir = tempdir().unwrap();
        let base = temp_dir.path();

        // Test valid directory creation
        let dir = create_module_dir(base, "test-module").unwrap();
        assert!(dir.is_dir());
        assert_eq!(dir.file_name().unwrap(), "test-module");

        // Test invalid directory name
        assert!(create_module_dir(base, "!@#").is_err());
    }

    #[test]
    fn test_validate_module_dir() {
        let temp_dir = tempdir().unwrap();
        let base = temp_dir.path();

        // Create a valid directory
        let valid_dir = base.join("valid-dir");
        fs::create_dir(&valid_dir).unwrap();
        assert!(validate_module_dir(&valid_dir).is_ok());

        // Test non-existent directory
        let non_existent = base.join("nonexistent");
        assert!(validate_module_dir(&non_existent).is_err());
    }

    #[test]
    fn test_cleanup_malformed_dirs() {
        let temp_dir = tempdir().unwrap();
        let base = temp_dir.path();

        // Create some malformed directories
        fs::create_dir_all(base.join("bad,dir")).unwrap();
        fs::create_dir_all(base.join("another{bad}dir")).unwrap();

        // Clean them up
        let fixed = cleanup_malformed_dirs(base).unwrap();
        assert_eq!(fixed, 2);

        // Verify the directories were fixed
        assert!(base.join("baddir").is_dir());
        assert!(base.join("anotherbaddir").is_dir());
    }
}
