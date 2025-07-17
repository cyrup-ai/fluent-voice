//! Simple workspace-hack regeneration library
//!
//! This library provides a working implementation for regenerating
//! workspace-hack using cargo-hakari with candle dependencies excluded.

pub mod config;
pub mod error;

pub use config::{HakariConfig, OmittedDependency, PackageInfo, WorkspaceConfig};
pub use error::{HakariRegenerateError, Result};

use std::process::Command;

/// Main regenerator for workspace-hack operations
pub struct HakariRegenerator {
    workspace_root: std::path::PathBuf,
}

impl HakariRegenerator {
    /// Create new regenerator
    pub fn new(workspace_root: std::path::PathBuf) -> Self {
        Self { workspace_root }
    }

    /// Regenerate workspace-hack
    pub async fn regenerate(&self) -> Result<()> {
        let output = Command::new("cargo")
            .arg("hakari")
            .arg("generate")
            .current_dir(&self.workspace_root)
            .output()
            .map_err(|e| {
                error::HakariRegenerateError::Io(error::IoError::FileOperation {
                    path: self.workspace_root.clone(),
                    source: e,
                })
            })?;

        if !output.status.success() {
            return Err(error::HakariRegenerateError::Hakari(
                error::HakariError::GenerationFailed {
                    reason: String::from_utf8_lossy(&output.stderr).to_string(),
                },
            ));
        }

        Ok(())
    }

    /// Verify workspace-hack
    pub async fn verify(&self) -> Result<()> {
        let output = Command::new("cargo")
            .arg("hakari")
            .arg("verify")
            .current_dir(&self.workspace_root)
            .output()
            .map_err(|e| {
                error::HakariRegenerateError::Io(error::IoError::FileOperation {
                    path: self.workspace_root.clone(),
                    source: e,
                })
            })?;

        if !output.status.success() {
            return Err(error::HakariRegenerateError::Hakari(
                error::HakariError::VerificationFailed {
                    reason: String::from_utf8_lossy(&output.stderr).to_string(),
                },
            ));
        }

        Ok(())
    }
}
