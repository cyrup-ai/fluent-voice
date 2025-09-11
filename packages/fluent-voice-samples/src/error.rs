//! Error types for the `fluent-voice-samples` crate.

use std::ffi::OsString;
use std::path::PathBuf;
use thiserror::Error;

/// The minimum allowed sample rate in Hz
pub const MIN_SAMPLE_RATE: u32 = 8000;

/// The maximum allowed sample rate in Hz
pub const MAX_SAMPLE_RATE: u32 = 48000;

/// The minimum allowed sample duration in seconds
pub const MIN_SAMPLE_DURATION: f32 = 0.5;

/// The maximum allowed sample duration in seconds
pub const MAX_SAMPLE_DURATION: f32 = 10.0;

/// Custom error type for sample-related operations.
#[derive(Debug, Error)]
pub enum SampleError {
    /// I/O error that occurred while processing a file or directory.
    #[error("I/O error for {path}: {source}")]
    Io {
        /// The path that caused the error.
        path: PathBuf,
        /// The underlying I/O error.
        source: std::io::Error,
    },

    /// The specified path is not a directory.
    #[error("Not a directory: {0:?}")]
    NotADirectory(PathBuf),

    /// The specified path is not a file.
    #[error("Not a file: {0:?}")]
    NotAFile(PathBuf),

    /// The specified path is invalid.
    #[error("Invalid path: {0}")]
    InvalidPath(PathBuf),

    /// Audio processing error (trimming, encoding, etc.)
    #[error("Audio processing error: {0}")]
    AudioProcessing(String),

    /// The file has an invalid extension.
    #[error("Invalid file extension: {0:?}")]
    InvalidFileExtension(PathBuf),

    /// The audio file format is not supported.
    #[error("Unsupported audio format: {0}")]
    UnsupportedAudioFormat(String),

    /// The audio file is corrupted or invalid.
    #[error("Invalid audio file {path}: {source}")]
    InvalidAudioFile {
        /// The path to the invalid audio file.
        path: PathBuf,
        /// The underlying error.
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// The sample rate is not supported.
    #[error("Unsupported sample rate: {0} Hz")]
    UnsupportedSampleRate(u32),

    /// The sample duration is too short or too long.
    #[error("Invalid sample duration: {0:.2}s (must be between {1:.2}s and {2:.2}s)")]
    InvalidSampleDuration(f32, f32, f32),

    /// The metadata is invalid or corrupted.
    #[error("Invalid metadata: {0}")]
    InvalidMetadata(String),

    /// The directory structure is invalid.
    #[error("Invalid directory structure: {0}")]
    InvalidDirectoryStructure(String),

    /// The directory name is invalid or contains invalid characters.
    #[error("Invalid directory name '{0:?}': {1}")]
    InvalidDirectoryName(OsString, String),

    /// The metadata file is missing or invalid.
    #[error("Metadata file error at {path}: {source}")]
    MetadataFileError {
        /// The path to the metadata file
        path: PathBuf,
        /// The underlying error
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// The sample rate is out of the allowed range.
    #[error(
        "Sample rate {0} Hz is out of range. Must be between {MIN_SAMPLE_RATE} and {MAX_SAMPLE_RATE} Hz"
    )]
    SampleRateOutOfRange(u32),

    /// The sample duration is out of the allowed range.
    #[error(
        "Sample duration {0:.2}s is out of range. Must be between {MIN_SAMPLE_DURATION} and {MAX_SAMPLE_DURATION} seconds"
    )]
    SampleDurationOutOfRange(f32),

    /// The audio file is corrupted or invalid.
    #[error("Corrupted audio file: {0}")]
    CorruptedAudioFile(PathBuf),

    /// The maximum number of samples has been exceeded.
    #[error("Maximum number of samples ({0}) exceeded")]
    MaxSamplesExceeded(usize),

    /// The operation is not supported.
    #[error("Operation not supported: {0}")]
    UnsupportedOperation(String),

    /// The operation was cancelled by the user.
    #[error("Operation cancelled by user")]
    Cancelled,

    /// Progress tracking is not initialized.
    #[error("Progress tracking not initialized")]
    ProgressNotInitialized,

    /// Progress tracking is already initialized.
    #[error("Progress tracking already initialized")]
    ProgressAlreadyInitialized,

    /// Failed to create progress bar style.
    #[error("Failed to create progress bar style")]
    ProgressStyleCreation,

    /// Audio data has not been loaded yet.
    #[error("Audio data not loaded for {path}: {reason}")]
    AudioDataNotLoaded {
        /// The path to the audio file.
        path: PathBuf,
        /// The underlying error or reason.
        reason: String,
    },
}

/// A specialized `Result` type for sample operations.
pub type Result<T> = std::result::Result<T, SampleError>;

// Implement conversion from std::io::Error to SampleError
impl From<std::io::Error> for SampleError {
    fn from(error: std::io::Error) -> Self {
        Self::Io {
            path: PathBuf::new(),
            source: error,
        }
    }
}

// Implement conversion from std::io::Error with path to SampleError
impl SampleError {
    /// Creates a new I/O error with the given path and error.
    pub fn io_error<P: Into<PathBuf>>(path: P, error: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source: error,
        }
    }

    /// Creates a new metadata file error with the given path and error.
    pub fn metadata_error<P: Into<PathBuf>, E: std::error::Error + Send + Sync + 'static>(
        path: P,
        error: E,
    ) -> Self {
        Self::MetadataFileError {
            path: path.into(),
            source: Box::new(error),
        }
    }
}

// Implement FromStr for SampleError to allow easy conversion from strings
impl std::str::FromStr for SampleError {
    type Err = Self;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(Self::InvalidMetadata(s.to_string()))
    }
}

// Implement conversion from serde_yaml::Error to SampleError
impl From<serde_yaml::Error> for SampleError {
    fn from(error: serde_yaml::Error) -> Self {
        Self::InvalidMetadata(error.to_string())
    }
}

// Implement conversion from std::num::TryFromIntError to SampleError
impl From<std::num::TryFromIntError> for SampleError {
    fn from(error: std::num::TryFromIntError) -> Self {
        Self::InvalidMetadata(error.to_string())
    }
}

// Implement conversion from std::string::FromUtf8Error to SampleError
impl From<std::string::FromUtf8Error> for SampleError {
    fn from(error: std::string::FromUtf8Error) -> Self {
        Self::InvalidMetadata(error.to_string())
    }
}

// Implement conversion from std::str::Utf8Error to SampleError
impl From<std::str::Utf8Error> for SampleError {
    fn from(error: std::str::Utf8Error) -> Self {
        Self::InvalidMetadata(error.to_string())
    }
}

// Implement conversion from std::num::ParseIntError to SampleError
impl From<std::num::ParseIntError> for SampleError {
    fn from(error: std::num::ParseIntError) -> Self {
        Self::InvalidMetadata(error.to_string())
    }
}

// Implement conversion from std::num::ParseFloatError to SampleError
impl From<std::num::ParseFloatError> for SampleError {
    fn from(error: std::num::ParseFloatError) -> Self {
        Self::InvalidMetadata(error.to_string())
    }
}

// Implement conversion from std::time::SystemTimeError to SampleError
impl From<std::time::SystemTimeError> for SampleError {
    fn from(error: std::time::SystemTimeError) -> Self {
        Self::InvalidMetadata(error.to_string())
    }
}

// Implement conversion from std::ffi::NulError to SampleError
impl From<std::ffi::NulError> for SampleError {
    fn from(error: std::ffi::NulError) -> Self {
        Self::InvalidPath(PathBuf::from(error.to_string()))
    }
}

// Implement conversion from std::path::StripPrefixError to SampleError
impl From<std::path::StripPrefixError> for SampleError {
    fn from(error: std::path::StripPrefixError) -> Self {
        Self::InvalidPath(PathBuf::from(error.to_string()))
    }
}

// Implement conversion from std::string::FromUtf16Error to SampleError
impl From<std::string::FromUtf16Error> for SampleError {
    fn from(error: std::string::FromUtf16Error) -> Self {
        Self::InvalidMetadata(error.to_string())
    }
}
