//! Cargo-hakari API integration for workspace-hack operations
//!
//! This module provides high-performance integration with the cargo-hakari
//! API for workspace-hack initialization, generation, and verification.

use std::path::{Path, PathBuf};
use std::process::Command;
use tokio::fs;
use cargo_metadata::MetadataCommand;
use smallvec::SmallVec;

use crate::config::{HakariConfig, ConfigManager};
use crate::error::{HakariError, Result, IoErrorExt};
use crate::transaction::Transaction;
use crate::workspace::WorkspaceManager;

/// Hakari operations manager
pub struct HakariManager {
    workspace_root: PathBuf,
    workspace_hack_path: PathBuf,
    config_manager: ConfigManager,
    metadata_cache: Option<cargo_metadata::Metadata>}

/// Hakari operation options
#[derive(Debug, Clone)]
pub struct HakariOptions {
    pub verbose: bool,
    pub dry_run: bool,
    pub skip_verification: bool,
    pub force_regenerate: bool}

impl Default for HakariOptions {
    fn default() -> Self {
        Self {
            verbose: false,
            dry_run: false,
            skip_verification: false,
            force_regenerate: false}
    }
}

/// Hakari operation result
#[derive(Debug, Clone)]
pub struct HakariResult {
    pub success: bool,
    pub operations_performed: SmallVec<[String; 8]>,
    pub warnings: SmallVec<[String; 4]>,
    pub duration: std::time::Duration}

impl HakariManager {
    /// Create new hakari manager
    pub fn new(workspace_root: PathBuf) -> Self {
        let workspace_hack_path = workspace_root.join("workspace-hack");
        let config_path = workspace_root.join(".config").join("hakari.toml");
        let config_manager = ConfigManager::new(config_path);
        
        Self {
            workspace_root,
            workspace_hack_path,
            config_manager,
            metadata_cache: None}
    }
    
    /// Get cargo metadata with caching
    pub async fn get_metadata(&mut self) -> Result<&cargo_metadata::Metadata> {
        if self.metadata_cache.is_none() {
            let metadata = MetadataCommand::new()
                .manifest_path(self.workspace_root.join("Cargo.toml"))
                .exec()
                .map_err(|e| HakariError::InitializationFailed {
                    reason: format!("failed to get cargo metadata: {}", e)})?;
            
            self.metadata_cache = Some(metadata);
        }
        
        Ok(self.metadata_cache.as_ref().unwrap())
    }
    
    /// Initialize workspace-hack using cargo-hakari
    pub async fn initialize_workspace_hack(&mut self, transaction: &mut Transaction, options: &HakariOptions) -> Result<HakariResult> {
        let start_time = std::time::Instant::now();
        let mut operations = SmallVec::new();
        let mut warnings = SmallVec::new();
        
        // Remove existing workspace-hack if it exists
        if self.workspace_hack_path.exists() {
            transaction.record_file_deleted(self.workspace_hack_path.clone()).await?;
            
            if !options.dry_run {
                fs::remove_dir_all(&self.workspace_hack_path)
                    .await
                    .with_path(self.workspace_hack_path.clone())?;
            }
            
            operations.push("Removed existing workspace-hack".to_string());
        }
        
        // Create workspace-hack directory
        if !options.dry_run {
            fs::create_dir_all(&self.workspace_hack_path)
                .await
                .with_path(self.workspace_hack_path.clone())?;
        }
        
        transaction.record_workspace_hack_init(self.workspace_hack_path.clone())?;
        
        // Run cargo hakari init
        let init_result = self.run_hakari_init(options).await?;
        
        if !init_result.success {
            return Err(HakariError::InitializationFailed {
                reason: "cargo hakari init failed".to_string()}.into());
        }
        
        operations.push("Initialized workspace-hack with cargo hakari".to_string());
        
        // Rename package if needed
        if !options.dry_run {
            self.rename_workspace_hack_package(transaction).await?;
            operations.push("Renamed workspace-hack package".to_string());
        }
        
        Ok(HakariResult {
            success: true,
            operations_performed: operations,
            warnings,
            duration: start_time.elapsed()})
    }
    
    /// Run cargo hakari init command
    async fn run_hakari_init(&self, options: &HakariOptions) -> Result<HakariResult> {
        let mut cmd = Command::new("cargo");
        cmd.arg("hakari")
            .arg("init")
            .arg(&self.workspace_hack_path)
            .arg("--yes")
            .current_dir(&self.workspace_root);
        
        if options.verbose {
            cmd.arg("--verbose");
        }
        
        if options.dry_run {
            cmd.arg("--dry-run");
        }
        
        let output = cmd.output()
            .map_err(|e| HakariError::InitializationFailed {
                reason: format!("failed to run cargo hakari init: {}", e)})?;
        
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(HakariError::InitializationFailed {
                reason: format!("cargo hakari init failed: {}", stderr)}.into());
        }
        
        Ok(HakariResult {
            success: true,
            operations_performed: SmallVec::from_vec(vec!["cargo hakari init".to_string()]),
            warnings: SmallVec::new(),
            duration: std::time::Duration::from_millis(0), // Not tracking in subprocess
        })
    }
    
    /// Rename workspace-hack package to fluent-voice-workspace-hack
    async fn rename_workspace_hack_package(&self, transaction: &mut Transaction) -> Result<()> {
        let cargo_toml_path = self.workspace_hack_path.join("Cargo.toml");
        
        if !cargo_toml_path.exists() {
            return Err(HakariError::WorkspaceHackNotFound {
                path: cargo_toml_path}.into());
        }
        
        transaction.record_file_modified(cargo_toml_path.clone()).await?;
        
        let content = fs::read_to_string(&cargo_toml_path)
            .await
            .with_path(cargo_toml_path.clone())?;
        
        let modified_content = content.replace(
            r#"name = "workspace-hack""#,
            r#"name = "fluent-voice-workspace-hack""#,
        );
        
        transaction.atomic_write(&cargo_toml_path, modified_content.as_bytes()).await?;
        
        transaction.record_package_renamed(
            cargo_toml_path,
            "workspace-hack".to_string(),
            "fluent-voice-workspace-hack".to_string(),
        )?;
        
        Ok(())
    }
    
    /// Generate workspace-hack using cargo-hakari with custom configuration
    pub async fn generate_workspace_hack(&mut self, transaction: &mut Transaction, options: &HakariOptions) -> Result<HakariResult> {
        let start_time = std::time::Instant::now();
        let mut operations = SmallVec::new();
        let mut warnings = SmallVec::new();
        
        // Load configuration
        let config = self.config_manager.load_or_default().await?;
        
        // Validate configuration
        config.validate()?;
        
        // Backup existing configuration
        self.config_manager.backup().await?;
        transaction.record_config_backup(
            self.config_manager.config_path.clone(),
            self.config_manager.backup_path.clone(),
        )?;
        
        // Save updated configuration
        if !options.dry_run {
            self.config_manager.save(&config).await?;
        }
        
        operations.push("Updated hakari configuration".to_string());
        
        // Run cargo hakari generate
        let generate_result = self.run_hakari_generate(options).await?;
        
        if !generate_result.success {
            return Err(HakariError::GenerationFailed {
                reason: "cargo hakari generate failed".to_string()}.into());
        }
        
        operations.extend(generate_result.operations_performed);
        warnings.extend(generate_result.warnings);
        
        // Verify if not skipped
        if !options.skip_verification {
            let verify_result = self.verify_workspace_hack(options).await?;
            
            if !verify_result.success {
                warnings.push("Verification failed".to_string());
            } else {
                operations.push("Verified workspace-hack".to_string());
            }
        }
        
        Ok(HakariResult {
            success: true,
            operations_performed: operations,
            warnings,
            duration: start_time.elapsed()})
    }
    
    /// Run cargo hakari generate command
    async fn run_hakari_generate(&self, options: &HakariOptions) -> Result<HakariResult> {
        let mut cmd = Command::new("cargo");
        cmd.arg("hakari")
            .arg("generate")
            .current_dir(&self.workspace_root);
        
        if options.verbose {
            cmd.arg("--verbose");
        }
        
        if options.dry_run {
            cmd.arg("--dry-run");
        }
        
        let output = cmd.output()
            .map_err(|e| HakariError::GenerationFailed {
                reason: format!("failed to run cargo hakari generate: {}", e)})?;
        
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(HakariError::GenerationFailed {
                reason: format!("cargo hakari generate failed: {}", stderr)}.into());
        }
        
        let stdout = String::from_utf8_lossy(&output.stdout);
        let operations = if stdout.contains("no changes detected") {
            SmallVec::from_vec(vec!["No changes detected in workspace-hack".to_string()])
        } else {
            SmallVec::from_vec(vec!["Generated workspace-hack contents".to_string()])
        };
        
        Ok(HakariResult {
            success: true,
            operations_performed: operations,
            warnings: SmallVec::new(),
            duration: std::time::Duration::from_millis(0), // Not tracking in subprocess
        })
    }
    
    /// Verify workspace-hack using cargo-hakari
    pub async fn verify_workspace_hack(&self, options: &HakariOptions) -> Result<HakariResult> {
        let start_time = std::time::Instant::now();
        
        let mut cmd = Command::new("cargo");
        cmd.arg("hakari")
            .arg("verify")
            .current_dir(&self.workspace_root);
        
        if options.verbose {
            cmd.arg("--verbose");
        }
        
        let output = cmd.output()
            .map_err(|e| HakariError::VerificationFailed {
                reason: format!("failed to run cargo hakari verify: {}", e)})?;
        
        let success = output.status.success();
        let operations = if success {
            SmallVec::from_vec(vec!["Workspace-hack verification passed".to_string()])
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            SmallVec::from_vec(vec![format!("Workspace-hack verification failed: {}", stderr)])
        };
        
        Ok(HakariResult {
            success,
            operations_performed: operations,
            warnings: SmallVec::new(),
            duration: start_time.elapsed()})
    }
    
    /// Complete workspace-hack regeneration workflow
    pub async fn regenerate_workspace_hack(&mut self, workspace_manager: &mut WorkspaceManager, options: &HakariOptions) -> Result<HakariResult> {
        let start_time = std::time::Instant::now();
        let mut all_operations = SmallVec::new();
        let mut all_warnings = SmallVec::new();
        
        // Create transaction for atomic operations
        let mut transaction = crate::transaction::TransactionBuilder::new()
            .auto_rollback(true)
            .build()?;
        
        // Step 1: Comment out workspace-hack dependencies
        workspace_manager.comment_workspace_hack_dependencies(&mut transaction).await?;
        all_operations.push("Commented workspace-hack dependencies".to_string());
        
        // Step 2: Comment out workspace-hack member
        workspace_manager.comment_workspace_hack_member(&mut transaction).await?;
        all_operations.push("Commented workspace-hack member".to_string());
        
        // Step 3: Initialize workspace-hack
        let init_result = self.initialize_workspace_hack(&mut transaction, options).await?;
        all_operations.extend(init_result.operations_performed);
        all_warnings.extend(init_result.warnings);
        
        // Step 4: Uncomment workspace-hack member
        workspace_manager.uncomment_workspace_hack_member(&mut transaction).await?;
        all_operations.push("Uncommented workspace-hack member".to_string());
        
        // Step 5: Uncomment workspace-hack dependencies
        workspace_manager.uncomment_workspace_hack_dependencies(&mut transaction).await?;
        all_operations.push("Uncommented workspace-hack dependencies".to_string());
        
        // Step 6: Generate workspace-hack with custom configuration
        let generate_result = self.generate_workspace_hack(&mut transaction, options).await?;
        all_operations.extend(generate_result.operations_performed);
        all_warnings.extend(generate_result.warnings);
        
        // Commit transaction
        transaction.commit().await?;
        all_operations.push("Committed all changes".to_string());
        
        Ok(HakariResult {
            success: true,
            operations_performed: all_operations,
            warnings: all_warnings,
            duration: start_time.elapsed()})
    }
    
    /// Check if workspace-hack exists and is valid
    pub async fn is_workspace_hack_valid(&self) -> Result<bool> {
        let cargo_toml_path = self.workspace_hack_path.join("Cargo.toml");
        
        if !cargo_toml_path.exists() {
            return Ok(false);
        }
        
        let content = fs::read_to_string(&cargo_toml_path)
            .await
            .with_path(cargo_toml_path)?;
        
        // Check if it's a valid workspace-hack
        let is_valid = content.contains("fluent-voice-workspace-hack") &&
                      content.contains("workspace-hack package, managed by hakari");
        
        Ok(is_valid)
    }
    
    /// Get workspace-hack package information
    pub async fn get_workspace_hack_info(&self) -> Result<Option<WorkspaceHackInfo>> {
        let cargo_toml_path = self.workspace_hack_path.join("Cargo.toml");
        
        if !cargo_toml_path.exists() {
            return Ok(None);
        }
        
        let content = fs::read_to_string(&cargo_toml_path)
            .await
            .with_path(cargo_toml_path)?;
        
        let doc = content.parse::<toml_edit::Document>()
            .map_err(|e| HakariError::ConfigInvalid {
                reason: format!("failed to parse workspace-hack Cargo.toml: {}", e)})?;
        
        let name = doc.get("package")
            .and_then(|p| p.get("name"))
            .and_then(|n| n.as_str())
            .unwrap_or("unknown");
        
        let version = doc.get("package")
            .and_then(|p| p.get("version"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        
        let dependency_count = doc.get("dependencies")
            .and_then(|d| d.as_table())
            .map(|t| t.len())
            .unwrap_or(0);
        
        Ok(Some(WorkspaceHackInfo {
            name: name.to_string(),
            version: version.to_string(),
            path: self.workspace_hack_path.clone(),
            dependency_count}))
    }
    
    /// Clean up temporary files and backups
    pub async fn cleanup(&self) -> Result<()> {
        self.config_manager.cleanup().await?;
        
        Ok(())
    }
    
    /// Reset metadata cache
    pub fn reset_cache(&mut self) {
        self.metadata_cache = None;
    }
}

/// Workspace-hack information
#[derive(Debug, Clone)]
pub struct WorkspaceHackInfo {
    pub name: String,
    pub version: String,
    pub path: PathBuf,
    pub dependency_count: usize}

/// Hakari configuration validator
pub struct ConfigValidator {
    config: HakariConfig}

impl ConfigValidator {
    /// Create new validator
    pub fn new(config: HakariConfig) -> Self {
        Self { config }
    }
    
    /// Validate configuration comprehensively
    pub fn validate_comprehensive(&self) -> Result<ValidationReport> {
        let mut report = ValidationReport::new();
        
        // Basic validation
        if let Err(e) = self.config.validate() {
            report.add_error(format!("Basic validation failed: {}", e));
        }
        
        // Check omitted dependencies
        self.validate_omitted_dependencies(&mut report);
        
        // Check resolver version
        self.validate_resolver_version(&mut report);
        
        // Check package name
        self.validate_package_name(&mut report);
        
        Ok(report)
    }
    
    /// Validate omitted dependencies
    fn validate_omitted_dependencies(&self, report: &mut ValidationReport) {
        let expected_omitted = [
            "candle-core",
            "candle-nn", 
            "candle-transformers",
            "candle-flash-attn",
            "candle-onnx",
            "candle-datasets",
            "cudarc",
            "bindgen_cuda",
            "half",
            "accelerate-src",
            "intel-mkl-src",
        ];
        
        for expected in &expected_omitted {
            if !self.config.is_dependency_omitted(expected) {
                report.add_warning(format!("Dependency '{}' is not omitted", expected));
            }
        }
    }
    
    /// Validate resolver version
    fn validate_resolver_version(&self, report: &mut ValidationReport) {
        if self.config.resolver != "2" && self.config.resolver != "3" {
            report.add_error(format!("Invalid resolver version: {}", self.config.resolver));
        }
    }
    
    /// Validate package name
    fn validate_package_name(&self, report: &mut ValidationReport) {
        if self.config.hakari_package != "fluent-voice-workspace-hack" {
            report.add_warning(format!("Package name is not 'fluent-voice-workspace-hack': {}", self.config.hakari_package));
        }
    }
}

/// Validation report
#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub errors: SmallVec<[String; 4]>,
    pub warnings: SmallVec<[String; 8]>}

impl ValidationReport {
    /// Create new validation report
    pub fn new() -> Self {
        Self {
            errors: SmallVec::new(),
            warnings: SmallVec::new()}
    }
    
    /// Add error
    pub fn add_error(&mut self, error: String) {
        self.errors.push(error);
    }
    
    /// Add warning
    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }
    
    /// Check if validation passed
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }
    
    /// Get summary
    pub fn summary(&self) -> String {
        format!("Errors: {}, Warnings: {}", self.errors.len(), self.warnings.len())
    }
}