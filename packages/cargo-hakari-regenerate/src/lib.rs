//! Simple workspace-hack regeneration library
//!
//! This library provides a working implementation for regenerating
//! workspace-hack using cargo-hakari with candle dependencies excluded.

pub mod config;
pub mod error;

use std::process::Command;

pub use config::{HakariConfig, OmittedDependency, PackageInfo, WorkspaceConfig};
pub use error::{HakariRegenerateError, Result};

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
        // Always ensure workspace-hack exists first
        self.ensure_workspace_hack_exists().await?;

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

    /// Initialize workspace-hack if it doesn't exist (should be called before regenerate)
    pub async fn ensure_workspace_hack_exists(&self) -> Result<()> {
        let workspace_hack_path = self.workspace_root.join("workspace-hack");
        if !workspace_hack_path.exists() {
            println!("Workspace-hack does not exist, initializing...");
            self.initialize_workspace_hack().await?;
        }
        Ok(())
    }

    /// Initialize workspace-hack if it doesn't exist
    async fn initialize_workspace_hack(&self) -> Result<()> {
        // Move existing config temporarily so init can create fresh one
        let config_path = self.workspace_root.join(".config/hakari.toml");
        let backup_path = self.workspace_root.join(".config/hakari.toml.backup");

        if config_path.exists() {
            std::fs::rename(&config_path, &backup_path).map_err(|e| {
                error::HakariRegenerateError::Io(error::IoError::FileOperation {
                    path: config_path.clone(),
                    source: e,
                })
            })?;
        }

        let output = Command::new("cargo")
            .arg("hakari")
            .arg("init")
            .arg("--yes")
            .arg("workspace-hack")
            .current_dir(&self.workspace_root)
            .output()
            .map_err(|e| {
                error::HakariRegenerateError::Io(error::IoError::FileOperation {
                    path: self.workspace_root.clone(),
                    source: e,
                })
            })?;

        // Always restore our custom config, regardless of init success/failure
        if backup_path.exists() {
            std::fs::rename(&backup_path, &config_path).map_err(|e| {
                error::HakariRegenerateError::Io(error::IoError::FileOperation {
                    path: backup_path.clone(),
                    source: e,
                })
            })?;
        }

        if !output.status.success() {
            return Err(error::HakariRegenerateError::Hakari(
                error::HakariError::InitializationFailed {
                    reason: String::from_utf8_lossy(&output.stderr).to_string(),
                },
            ));
        }

        println!("✓ Workspace-hack initialized successfully");
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
