//! Comprehensive error handling for cargo-hakari-regenerate
//!
//! This module provides zero-allocation error types with rich context
//! for all operations in the workspace-hack regeneration process.

use std::path::PathBuf;

use thiserror::Error;

/// Main error type for all cargo-hakari-regenerate operations
#[derive(Error, Debug)]
pub enum HakariRegenerateError {
    #[error("Workspace error: {0}")]
    Workspace(#[from] WorkspaceError),

    #[error("Hakari operation error: {0}")]
    Hakari(#[from] HakariError),

    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),

    #[error("I/O error: {0}")]
    Io(#[from] IoError),

    #[error("Transaction error: {0}")]
    Transaction(#[from] TransactionError),

    #[error("Validation error: {0}")]
    Validation(#[from] ValidationError),
}

/// Workspace-related errors
#[derive(Error, Debug)]
pub enum WorkspaceError {
    #[error("Workspace root not found: searched from {path}")]
    RootNotFound { path: PathBuf },

    #[error("Invalid workspace structure: {reason}")]
    InvalidStructure { reason: String },

    #[error("Workspace member not found: {member}")]
    MemberNotFound { member: String },

    #[error("Package not found: {package} in workspace")]
    PackageNotFound { package: String },

    #[error("Cargo.toml parse error in {path}: {source}")]
    CargoTomlParse {
        path: PathBuf,
        source: toml_edit::TomlError,
    },

    #[error("Cargo.toml format error in {path}: {reason}")]
    CargoTomlFormat { path: PathBuf, reason: String },

    #[error("Cargo metadata error: {0}")]
    CargoMetadata(#[from] cargo_metadata::Error),
}

/// Hakari-specific operation errors
#[derive(Error, Debug)]
pub enum HakariError {
    #[error("Hakari initialization failed: {reason}")]
    InitializationFailed { reason: String },

    #[error("Metadata cache not initialized: {0}")]
    CacheNotInitialized(String),

    #[error("Hakari generation failed: {reason}")]
    GenerationFailed { reason: String },

    #[error("Hakari verification failed: {reason}")]
    VerificationFailed { reason: String },

    #[error("Hakari configuration invalid: {reason}")]
    ConfigInvalid { reason: String },

    #[error("Workspace-hack package not found at {path}")]
    WorkspaceHackNotFound { path: PathBuf },

    #[error("Workspace-hack package rename failed: {from} -> {to}")]
    RenameFailed { from: String, to: String },
}

/// Configuration-related errors
#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("Configuration file not found: {path}")]
    FileNotFound { path: PathBuf },

    #[error("Configuration parse error in {path}: {source}")]
    ParseError {
        path: PathBuf,
        source: toml_edit::TomlError,
    },

    #[error("Configuration validation error: {field} - {reason}")]
    ValidationError { field: String, reason: String },

    #[error("Configuration backup failed: {path}")]
    BackupFailed { path: PathBuf },

    #[error("Configuration restore failed: {path}")]
    RestoreFailed { path: PathBuf },

    #[error("Missing required configuration: {field}")]
    MissingRequired { field: String },

    #[error("Invalid configuration value: {field} = {value}")]
    InvalidValue { field: String, value: String },
}

/// File system and I/O errors
#[derive(Error, Debug)]
pub enum IoError {
    #[error("File operation failed: {path} - {source}")]
    FileOperation {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("Directory operation failed: {path} - {source}")]
    DirectoryOperation {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("Permission denied: {path}")]
    PermissionDenied { path: PathBuf },

    #[error("File not found: {path}")]
    FileNotFound { path: PathBuf },

    #[error("Directory not found: {path}")]
    DirectoryNotFound { path: PathBuf },

    #[error("File already exists: {path}")]
    FileExists { path: PathBuf },

    #[error("Temporary file creation failed: {source}")]
    TempFileCreation { source: std::io::Error },

    #[error("Atomic file operation failed: {path}")]
    AtomicOperation { path: PathBuf },
}

/// Transaction and rollback errors
#[derive(Error, Debug)]
pub enum TransactionError {
    #[error("Transaction already started")]
    AlreadyStarted,

    #[error("Transaction not started")]
    NotStarted,

    #[error("Transaction commit failed: {reason}")]
    CommitFailed { reason: String },

    #[error("Transaction rollback failed: {reason}")]
    RollbackFailed { reason: String },

    #[error("Transaction checkpoint failed: {name}")]
    CheckpointFailed { name: String },

    #[error("Transaction state corrupted: {reason}")]
    StateCorrupted { reason: String },

    #[error("Rollback operation failed for {path}: {source}")]
    RollbackOperation {
        path: PathBuf,
        source: std::io::Error,
    },
}

/// Validation errors
#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("Workspace validation failed: {reason}")]
    WorkspaceValidation { reason: String },

    #[error("Package validation failed: {package} - {reason}")]
    PackageValidation { package: String, reason: String },

    #[error("Configuration validation failed: {field} - {reason}")]
    ConfigValidation { field: String, reason: String },

    #[error("Dependency validation failed: {dependency} - {reason}")]
    DependencyValidation { dependency: String, reason: String },

    #[error("Hakari validation failed: {reason}")]
    HakariValidation { reason: String },
}

/// Result type alias for convenience
pub type Result<T> = std::result::Result<T, HakariRegenerateError>;

/// Helper trait for converting std::io::Error to IoError with context
pub trait IoErrorExt<T> {
    fn with_path(self, path: PathBuf) -> std::result::Result<T, IoError>;
}

impl<T> IoErrorExt<T> for std::result::Result<T, std::io::Error> {
    fn with_path(self, path: PathBuf) -> std::result::Result<T, IoError> {
        self.map_err(|source| match source.kind() {
            std::io::ErrorKind::NotFound => IoError::FileNotFound { path },
            std::io::ErrorKind::PermissionDenied => IoError::PermissionDenied { path },
            std::io::ErrorKind::AlreadyExists => IoError::FileExists { path },
            _ => IoError::FileOperation { path, source },
        })
    }
}

/// Helper trait for converting toml_edit::TomlError with context
pub trait TomlErrorExt<T> {
    fn with_path(self, path: PathBuf) -> std::result::Result<T, WorkspaceError>;
}

impl<T> TomlErrorExt<T> for std::result::Result<T, toml_edit::TomlError> {
    fn with_path(self, path: PathBuf) -> std::result::Result<T, WorkspaceError> {
        self.map_err(|source| WorkspaceError::CargoTomlParse { path, source })
    }
}

/// Helper function to create validation errors
pub fn validation_error(field: &str, reason: &str) -> ValidationError {
    ValidationError::ConfigValidation {
        field: field.to_string(),
        reason: reason.to_string(),
    }
}

/// Helper function to create configuration errors
pub fn config_error(field: &str, reason: &str) -> ConfigError {
    ConfigError::ValidationError {
        field: field.to_string(),
        reason: reason.to_string(),
    }
}
