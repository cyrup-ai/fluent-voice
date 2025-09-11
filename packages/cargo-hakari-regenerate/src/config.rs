//! Configuration management for cargo-hakari-regenerate
//!
//! This module provides configuration handling for hakari operations.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{ConfigError, Result};

/// Main hakari configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HakariConfig {
    #[serde(rename = "hakari-package")]
    pub hakari_package: String,

    #[serde(rename = "dep-format-version")]
    pub dep_format_version: u32,

    pub resolver: String,

    #[serde(default)]
    pub platforms: Vec<String>,

    #[serde(default, rename = "exact-versions")]
    pub exact_versions: bool,

    #[serde(default, rename = "omitted-deps")]
    pub omitted_deps: Vec<OmittedDependency>,

    #[serde(default)]
    pub workspace_members: Vec<String>,
}

/// Omitted dependency configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OmittedDependency {
    pub name: String,

    #[serde(default)]
    pub version: Option<String>,

    #[serde(default)]
    pub features: Vec<String>,
}

/// Workspace configuration for package management
#[derive(Debug, Clone)]
pub struct WorkspaceConfig {
    pub root_path: PathBuf,
    pub hakari_config_path: PathBuf,
    pub workspace_hack_path: PathBuf,
    pub packages: Vec<PackageInfo>,
}

/// Package information for workspace operations
#[derive(Debug, Clone)]
pub struct PackageInfo {
    pub name: String,
    pub path: PathBuf,
    pub cargo_toml_path: PathBuf,
    pub has_workspace_hack_dep: bool,
}

impl Default for HakariConfig {
    fn default() -> Self {
        Self {
            hakari_package: "fluent-voice-workspace-hack".to_string(),
            dep_format_version: 4,
            resolver: "2".to_string(),
            platforms: Vec::new(),
            exact_versions: false,
            omitted_deps: Vec::new(),
            workspace_members: Vec::new(),
        }
    }
}

impl HakariConfig {
    /// Create default configuration with candle dependencies omitted
    pub fn new() -> Self {
        Self::default()
    }

    /// Create configuration optimized for the fluent-voice workspace
    pub fn for_fluent_voice() -> Self {
        Self {
            omitted_deps: Self::default_omitted_deps(),
            ..Self::default()
        }
    }

    /// Default omitted dependencies for excluding candle and related packages
    fn default_omitted_deps() -> Vec<OmittedDependency> {
        vec![
            OmittedDependency {
                name: "candle-core".to_string(),
                version: None,
                features: vec![],
            },
            OmittedDependency {
                name: "candle-nn".to_string(),
                version: None,
                features: vec![],
            },
            OmittedDependency {
                name: "candle-transformers".to_string(),
                version: None,
                features: vec![],
            },
            OmittedDependency {
                name: "candle-flash-attn".to_string(),
                version: None,
                features: vec![],
            },
            OmittedDependency {
                name: "candle-onnx".to_string(),
                version: None,
                features: vec![],
            },
            OmittedDependency {
                name: "candle-datasets".to_string(),
                version: None,
                features: vec![],
            },
            OmittedDependency {
                name: "cudarc".to_string(),
                version: None,
                features: vec![],
            },
            OmittedDependency {
                name: "half".to_string(),
                version: None,
                features: vec![],
            },
            OmittedDependency {
                name: "bindgen_cuda".to_string(),
                version: None,
                features: vec![],
            },
            OmittedDependency {
                name: "intel-mkl-src".to_string(),
                version: None,
                features: vec![],
            },
            OmittedDependency {
                name: "accelerate-src".to_string(),
                version: None,
                features: vec![],
            },
        ]
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.hakari_package.is_empty() {
            return Err(ConfigError::ValidationError {
                field: "hakari-package".to_string(),
                reason: "cannot be empty".to_string(),
            }
            .into());
        }

        if self.dep_format_version == 0 {
            return Err(ConfigError::ValidationError {
                field: "dep-format-version".to_string(),
                reason: "must be greater than 0".to_string(),
            }
            .into());
        }

        if self.resolver.is_empty() {
            return Err(ConfigError::ValidationError {
                field: "resolver".to_string(),
                reason: "cannot be empty".to_string(),
            }
            .into());
        }

        for (i, dep) in self.omitted_deps.iter().enumerate() {
            if dep.name.is_empty() {
                return Err(ConfigError::ValidationError {
                    field: format!("omitted-deps[{i}].name"),
                    reason: "cannot be empty".to_string(),
                }
                .into());
            }
        }

        Ok(())
    }

    /// Save configuration to file
    pub async fn save(&self, path: &Path) -> Result<()> {
        let toml_content =
            toml::to_string_pretty(self).map_err(|e| ConfigError::ValidationError {
                field: "serialization".to_string(),
                reason: e.to_string(),
            })?;

        tokio::fs::write(path, toml_content)
            .await
            .map_err(|e| ConfigError::ValidationError {
                field: "file_write".to_string(),
                reason: e.to_string(),
            })?;

        Ok(())
    }

    /// Load configuration from file
    pub async fn load(path: &Path) -> Result<Self> {
        let content =
            tokio::fs::read_to_string(path)
                .await
                .map_err(|_| ConfigError::FileNotFound {
                    path: path.to_path_buf(),
                })?;

        let config: HakariConfig =
            toml::from_str(&content).map_err(|e| ConfigError::ValidationError {
                field: "deserialization".to_string(),
                reason: e.to_string(),
            })?;

        config.validate()?;
        Ok(config)
    }
}

impl WorkspaceConfig {
    pub fn new(root_path: PathBuf) -> Self {
        Self {
            hakari_config_path: root_path.join(".hakari.toml"),
            workspace_hack_path: root_path.join("workspace-hack"),
            root_path,
            packages: Vec::new(),
        }
    }

    pub async fn discover_packages(&mut self) -> Result<()> {
        // Simple package discovery implementation
        let cargo_toml = self.root_path.join("Cargo.toml");
        let content = tokio::fs::read_to_string(&cargo_toml).await.map_err(|_| {
            ConfigError::FileNotFound {
                path: cargo_toml.clone(),
            }
        })?;

        let doc = content.parse::<toml_edit::DocumentMut>().map_err(|e| {
            ConfigError::ValidationError {
                field: "cargo_toml_parse".to_string(),
                reason: e.to_string(),
            }
        })?;

        if let Some(workspace) = doc.get("workspace")
            && let Some(members) = workspace.get("members")
            && let Some(members_array) = members.as_array()
        {
            for member in members_array {
                if let Some(member_str) = member.as_str()
                    && member_str != "workspace-hack"
                {
                    let package_path = self.root_path.join(member_str);
                    let cargo_toml_path = package_path.join("Cargo.toml");

                    if cargo_toml_path.exists() {
                        let has_workspace_hack_dep =
                            self.check_workspace_hack_dep(&cargo_toml_path).await?;

                        self.packages.push(PackageInfo {
                            name: member_str.to_string(),
                            path: package_path,
                            cargo_toml_path,
                            has_workspace_hack_dep,
                        });
                    }
                }
            }
        }

        Ok(())
    }

    async fn check_workspace_hack_dep(&self, cargo_toml_path: &Path) -> Result<bool> {
        let content = tokio::fs::read_to_string(cargo_toml_path)
            .await
            .map_err(|_| ConfigError::FileNotFound {
                path: cargo_toml_path.to_path_buf(),
            })?;

        Ok(content.contains("fluent-voice-workspace-hack"))
    }

    pub fn validate(&self) -> Result<()> {
        if !self.root_path.exists() {
            return Err(ConfigError::ValidationError {
                field: "root_path".to_string(),
                reason: "does not exist".to_string(),
            }
            .into());
        }

        let cargo_toml = self.root_path.join("Cargo.toml");
        if !cargo_toml.exists() {
            return Err(ConfigError::ValidationError {
                field: "cargo_toml".to_string(),
                reason: "Cargo.toml not found in root".to_string(),
            }
            .into());
        }

        Ok(())
    }
}
