//! Workspace operations for Cargo.toml manipulation and package management
//!
//! This module provides high-performance workspace operations with zero-allocation
//! patterns for managing Cargo.toml files and workspace structure.

use std::borrow::Cow;
use std::path::{Path, PathBuf};
use smallvec::SmallVec;
use regex::Regex;
use toml_edit::{Document, Array, value, Item, Table};
use tokio::fs;
use walkdir::WalkDir;

use crate::config::{WorkspaceConfig, PackageInfo};
use crate::error::{WorkspaceError, Result, IoErrorExt, TomlErrorExt};
use crate::transaction::Transaction;

/// Workspace manager for high-performance operations
pub struct WorkspaceManager {
    root_path: PathBuf,
    config: WorkspaceConfig,
    comment_patterns: CommentPatterns}

/// Pre-compiled regex patterns for commenting/uncommenting
struct CommentPatterns {
    workspace_hack_member: Regex,
    workspace_hack_dep: Regex,
    commented_workspace_hack_member: Regex,
    commented_workspace_hack_dep: Regex}

impl CommentPatterns {
    /// Create new comment patterns
    fn new() -> Result<Self, crate::error::WorkspaceError> {
        Ok(Self {
            workspace_hack_member: Regex::new(r#"^\s*"workspace-hack",?\s*$"#)
                .map_err(|e| crate::error::WorkspaceError::InvalidStructure { 
                    reason: format!("Invalid regex pattern: {}", e) 
                })?,
            workspace_hack_dep: Regex::new(r#"^fluent-voice-workspace-hack\s*="#)
                .map_err(|e| crate::error::WorkspaceError::InvalidStructure { 
                    reason: format!("Invalid regex pattern: {}", e) 
                })?,
            commented_workspace_hack_member: Regex::new(r#"^\s*#\s*"workspace-hack",?\s*$"#)
                .map_err(|e| crate::error::WorkspaceError::InvalidStructure { 
                    reason: format!("Invalid regex pattern: {}", e) 
                })?,
            commented_workspace_hack_dep: Regex::new(r#"^#\s*fluent-voice-workspace-hack\s*="#)
                .map_err(|e| crate::error::WorkspaceError::InvalidStructure { 
                    reason: format!("Invalid regex pattern: {}", e) 
                })?})
    }
}

impl WorkspaceManager {
    /// Create new workspace manager
    pub async fn new(root_path: PathBuf) -> Result<Self> {
        let mut config = WorkspaceConfig::new(root_path.clone());
        config.discover_packages().await?;
        config.validate()?;
        
        Ok(Self {
            root_path,
            config,
            comment_patterns: CommentPatterns::new()?})
    }
    
    /// Get workspace configuration
    pub fn config(&self) -> &WorkspaceConfig {
        &self.config
    }
    
    /// Find workspace root from given path
    pub async fn find_workspace_root(start_path: &Path) -> Result<PathBuf> {
        let mut current = start_path.to_path_buf();
        
        loop {
            let cargo_toml = current.join("Cargo.toml");
            
            if cargo_toml.exists() {
                let content = fs::read_to_string(&cargo_toml)
                    .await
                    .with_path(cargo_toml.clone())?;
                
                if content.contains("[workspace]") {
                    return Ok(current);
                }
            }
            
            match current.parent() {
                Some(parent) => current = parent.to_path_buf(),
                None => break}
        }
        
        Err(WorkspaceError::RootNotFound {
            path: start_path.to_path_buf()}.into())
    }
    
    /// Parse workspace Cargo.toml
    pub async fn parse_workspace_cargo_toml(&self) -> Result<Document> {
        let cargo_toml_path = self.root_path.join("Cargo.toml");
        let content = fs::read_to_string(&cargo_toml_path)
            .await
            .with_path(cargo_toml_path.clone())?;
        
        content.parse::<Document>()
            .with_path(cargo_toml_path)?
    }
    
    /// Comment workspace-hack member in root Cargo.toml
    pub async fn comment_workspace_hack_member(&self, transaction: &mut Transaction) -> Result<()> {
        let cargo_toml_path = self.root_path.join("Cargo.toml");
        
        transaction.record_file_modified(cargo_toml_path.clone()).await?;
        
        let content = fs::read_to_string(&cargo_toml_path)
            .await
            .with_path(cargo_toml_path.clone())?;
        
        let modified_content = self.comment_patterns.workspace_hack_member
            .replace_all(&content, |caps: &regex::Captures| {
                format!("    # {}", caps[0].trim())
            });
        
        transaction.atomic_write(&cargo_toml_path, modified_content.as_bytes()).await?;
        
        Ok(())
    }
    
    /// Uncomment workspace-hack member in root Cargo.toml
    pub async fn uncomment_workspace_hack_member(&self, transaction: &mut Transaction) -> Result<()> {
        let cargo_toml_path = self.root_path.join("Cargo.toml");
        
        transaction.record_file_modified(cargo_toml_path.clone()).await?;
        
        let content = fs::read_to_string(&cargo_toml_path)
            .await
            .with_path(cargo_toml_path.clone())?;
        
        let modified_content = self.comment_patterns.commented_workspace_hack_member
            .replace_all(&content, r#"    "workspace-hack","#);
        
        transaction.atomic_write(&cargo_toml_path, modified_content.as_bytes()).await?;
        
        Ok(())
    }
    
    /// Comment workspace-hack dependencies in all packages
    pub async fn comment_workspace_hack_dependencies(&self, transaction: &mut Transaction) -> Result<()> {
        for package in &self.config.packages {
            if package.has_workspace_hack_dep {
                self.comment_workspace_hack_dependency_in_package(&package.cargo_toml_path, transaction).await?;
            }
        }
        
        Ok(())
    }
    
    /// Comment workspace-hack dependency in specific package
    async fn comment_workspace_hack_dependency_in_package(&self, cargo_toml_path: &Path, transaction: &mut Transaction) -> Result<()> {
        transaction.record_file_modified(cargo_toml_path.to_path_buf()).await?;
        
        let content = fs::read_to_string(cargo_toml_path)
            .await
            .with_path(cargo_toml_path.to_path_buf())?;
        
        let modified_content = self.comment_patterns.workspace_hack_dep
            .replace_all(&content, "# fluent-voice-workspace-hack =");
        
        transaction.atomic_write(cargo_toml_path, modified_content.as_bytes()).await?;
        
        Ok(())
    }
    
    /// Uncomment workspace-hack dependencies in all packages
    pub async fn uncomment_workspace_hack_dependencies(&self, transaction: &mut Transaction) -> Result<()> {
        for package in &self.config.packages {
            self.uncomment_workspace_hack_dependency_in_package(&package.cargo_toml_path, transaction).await?;
        }
        
        Ok(())
    }
    
    /// Uncomment workspace-hack dependency in specific package
    async fn uncomment_workspace_hack_dependency_in_package(&self, cargo_toml_path: &Path, transaction: &mut Transaction) -> Result<()> {
        transaction.record_file_modified(cargo_toml_path.to_path_buf()).await?;
        
        let content = fs::read_to_string(cargo_toml_path)
            .await
            .with_path(cargo_toml_path.to_path_buf())?;
        
        let modified_content = self.comment_patterns.commented_workspace_hack_dep
            .replace_all(&content, "fluent-voice-workspace-hack =");
        
        transaction.atomic_write(cargo_toml_path, modified_content.as_bytes()).await?;
        
        Ok(())
    }
    
    /// Validate workspace structure
    pub async fn validate_workspace(&self) -> Result<()> {
        let cargo_toml_path = self.root_path.join("Cargo.toml");
        
        if !cargo_toml_path.exists() {
            return Err(WorkspaceError::InvalidStructure {
                reason: "workspace Cargo.toml not found".to_string()}.into());
        }
        
        let content = fs::read_to_string(&cargo_toml_path)
            .await
            .with_path(cargo_toml_path.clone())?;
        
        if !content.contains("[workspace]") {
            return Err(WorkspaceError::InvalidStructure {
                reason: "not a workspace (missing [workspace] section)".to_string()}.into());
        }
        
        // Validate packages exist
        let doc = content.parse::<Document>()
            .with_path(cargo_toml_path)?;
        
        if let Some(workspace) = doc.get("workspace") {
            if let Some(members) = workspace.get("members") {
                if let Some(members_array) = members.as_array() {
                    for member in members_array {
                        if let Some(member_str) = member.as_str() {
                            if member_str != "workspace-hack" {
                                let member_path = self.root_path.join(member_str);
                                if !member_path.exists() {
                                    return Err(WorkspaceError::MemberNotFound {
                                        member: member_str.to_string()}.into());
                                }
                            }
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Get all workspace members
    pub async fn get_workspace_members(&self) -> Result<SmallVec<[Cow<'static, str>; 16]>> {
        let cargo_toml_path = self.root_path.join("Cargo.toml");
        let content = fs::read_to_string(&cargo_toml_path)
            .await
            .with_path(cargo_toml_path.clone())?;
        
        let doc = content.parse::<Document>()
            .with_path(cargo_toml_path)?;
        
        let mut members = SmallVec::new();
        
        if let Some(workspace) = doc.get("workspace") {
            if let Some(members_item) = workspace.get("members") {
                if let Some(members_array) = members_item.as_array() {
                    for member in members_array {
                        if let Some(member_str) = member.as_str() {
                            members.push(Cow::Owned(member_str.to_string()));
                        }
                    }
                }
            }
        }
        
        Ok(members)
    }
    
    /// Add workspace member
    pub async fn add_workspace_member(&self, member: &str, transaction: &mut Transaction) -> Result<()> {
        let cargo_toml_path = self.root_path.join("Cargo.toml");
        
        transaction.record_file_modified(cargo_toml_path.clone()).await?;
        
        let content = fs::read_to_string(&cargo_toml_path)
            .await
            .with_path(cargo_toml_path.clone())?;
        
        let mut doc = content.parse::<Document>()
            .with_path(cargo_toml_path.clone())?;
        
        // Get or create workspace section
        let workspace = doc.entry("workspace").or_insert_with(|| {
            let mut table = Table::new();
            table.set_implicit(true);
            Item::Table(table)
        });
        
        // Get or create members array
        let members = workspace.entry("members").or_insert_with(|| {
            Item::Value(value(Array::new()))
        });
        
        if let Some(members_array) = members.as_array_mut() {
            // Check if member already exists
            let member_exists = members_array.iter().any(|m| {
                m.as_str().map_or(false, |s| s == member)
            });
            
            if !member_exists {
                members_array.push(member);
            }
        }
        
        let modified_content = doc.to_string();
        transaction.atomic_write(&cargo_toml_path, modified_content.as_bytes()).await?;
        
        Ok(())
    }
    
    /// Remove workspace member
    pub async fn remove_workspace_member(&self, member: &str, transaction: &mut Transaction) -> Result<()> {
        let cargo_toml_path = self.root_path.join("Cargo.toml");
        
        transaction.record_file_modified(cargo_toml_path.clone()).await?;
        
        let content = fs::read_to_string(&cargo_toml_path)
            .await
            .with_path(cargo_toml_path.clone())?;
        
        let mut doc = content.parse::<Document>()
            .with_path(cargo_toml_path.clone())?;
        
        if let Some(workspace) = doc.get_mut("workspace") {
            if let Some(members) = workspace.get_mut("members") {
                if let Some(members_array) = members.as_array_mut() {
                    members_array.retain(|m| {
                        m.as_str().map_or(true, |s| s != member)
                    });
                }
            }
        }
        
        let modified_content = doc.to_string();
        transaction.atomic_write(&cargo_toml_path, modified_content.as_bytes()).await?;
        
        Ok(())
    }
    
    /// Check if workspace has workspace-hack member
    pub async fn has_workspace_hack_member(&self) -> Result<bool> {
        let members = self.get_workspace_members().await?;
        Ok(members.iter().any(|m| m == "workspace-hack"))
    }
    
    /// Update workspace-hack member (ensure it exists exactly once)
    pub async fn ensure_workspace_hack_member(&self, transaction: &mut Transaction) -> Result<()> {
        let has_member = self.has_workspace_hack_member().await?;
        
        if !has_member {
            self.add_workspace_member("workspace-hack", transaction).await?;
        }
        
        Ok(())
    }
    
    /// Refresh package information
    pub async fn refresh_packages(&mut self) -> Result<()> {
        self.config.packages.clear();
        self.config.discover_packages().await?;
        
        Ok(())
    }
    
    /// Get package by name
    pub fn get_package(&self, name: &str) -> Option<&PackageInfo> {
        self.config.get_package(name)
    }
    
    /// Get all packages
    pub fn packages(&self) -> &[PackageInfo] {
        &self.config.packages
    }
    
    /// Get packages with workspace-hack dependency
    pub fn packages_with_workspace_hack(&self) -> impl Iterator<Item = &PackageInfo> {
        self.config.packages_with_workspace_hack()
    }
    
    /// Create backup of workspace state
    pub async fn backup_workspace_state(&self, backup_dir: &Path) -> Result<()> {
        if !backup_dir.exists() {
            fs::create_dir_all(backup_dir)
                .await
                .with_path(backup_dir.to_path_buf())?;
        }
        
        // Backup root Cargo.toml
        let root_cargo_toml = self.root_path.join("Cargo.toml");
        let backup_root_cargo_toml = backup_dir.join("Cargo.toml");
        
        fs::copy(&root_cargo_toml, &backup_root_cargo_toml)
            .await
            .with_path(root_cargo_toml)?;
        
        // Backup package Cargo.toml files
        for package in &self.config.packages {
            let relative_path = package.cargo_toml_path.strip_prefix(&self.root_path)
                .map_err(|_| WorkspaceError::InvalidStructure {
                    reason: format!("package path not in workspace: {:?}", package.cargo_toml_path)})?;
            
            let backup_path = backup_dir.join(relative_path);
            
            if let Some(parent) = backup_path.parent() {
                fs::create_dir_all(parent)
                    .await
                    .with_path(parent.to_path_buf())?;
            }
            
            fs::copy(&package.cargo_toml_path, &backup_path)
                .await
                .with_path(package.cargo_toml_path.clone())?;
        }
        
        Ok(())
    }
    
    /// Restore workspace state from backup
    pub async fn restore_workspace_state(&self, backup_dir: &Path) -> Result<()> {
        if !backup_dir.exists() {
            return Err(WorkspaceError::InvalidStructure {
                reason: "backup directory does not exist".to_string()}.into());
        }
        
        // Restore root Cargo.toml
        let backup_root_cargo_toml = backup_dir.join("Cargo.toml");
        let root_cargo_toml = self.root_path.join("Cargo.toml");
        
        if backup_root_cargo_toml.exists() {
            fs::copy(&backup_root_cargo_toml, &root_cargo_toml)
                .await
                .with_path(root_cargo_toml)?;
        }
        
        // Restore package Cargo.toml files
        for entry in WalkDir::new(backup_dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name() == "Cargo.toml" && e.path() != backup_root_cargo_toml)
        {
            let relative_path = entry.path().strip_prefix(backup_dir)
                .map_err(|_| WorkspaceError::InvalidStructure {
                    reason: format!("backup file not in backup dir: {:?}", entry.path())})?;
            
            let target_path = self.root_path.join(relative_path);
            
            fs::copy(entry.path(), &target_path)
                .await
                .with_path(target_path)?;
        }
        
        Ok(())
    }
}

/// High-performance package discovery
pub struct PackageDiscovery {
    root_path: PathBuf,
    packages_cache: SmallVec<[PackageInfo; 32]>,
    cache_valid: bool}

impl PackageDiscovery {
    /// Create new package discovery
    pub fn new(root_path: PathBuf) -> Self {
        Self {
            root_path,
            packages_cache: SmallVec::new(),
            cache_valid: false}
    }
    
    /// Discover all packages in workspace
    pub async fn discover_all(&mut self) -> Result<&[PackageInfo]> {
        if self.cache_valid {
            return Ok(&self.packages_cache);
        }
        
        self.packages_cache.clear();
        
        let packages_dir = self.root_path.join("packages");
        
        if packages_dir.exists() {
            self.discover_in_directory(&packages_dir).await?;
        }
        
        self.cache_valid = true;
        
        Ok(&self.packages_cache)
    }
    
    /// Discover packages in specific directory
    async fn discover_in_directory(&mut self, dir: &Path) -> Result<()> {
        let mut read_dir = fs::read_dir(dir)
            .await
            .with_path(dir.to_path_buf())?;
        
        while let Some(entry) = read_dir.next_entry()
            .await
            .with_path(dir.to_path_buf())?
        {
            let path = entry.path();
            
            if path.is_dir() {
                let cargo_toml_path = path.join("Cargo.toml");
                
                if cargo_toml_path.exists() {
                    let name = path.file_name()
                        .and_then(|n| n.to_str())
                        .map(|s| Cow::Owned(s.to_string()))
                        .unwrap_or_else(|| Cow::Borrowed("unknown"));
                    
                    let has_workspace_hack_dep = self.check_workspace_hack_dependency(&cargo_toml_path).await?;
                    
                    self.packages_cache.push(PackageInfo {
                        name,
                        path,
                        cargo_toml_path,
                        has_workspace_hack_dep});
                }
            }
        }
        
        Ok(())
    }
    
    /// Check if package has workspace-hack dependency
    async fn check_workspace_hack_dependency(&self, cargo_toml_path: &Path) -> Result<bool> {
        let content = fs::read_to_string(cargo_toml_path)
            .await
            .with_path(cargo_toml_path.to_path_buf())?;
        
        Ok(content.contains("fluent-voice-workspace-hack"))
    }
    
    /// Invalidate cache
    pub fn invalidate_cache(&mut self) {
        self.cache_valid = false;
    }
}