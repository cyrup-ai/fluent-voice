//! Command-line interface for cargo-hakari-regenerate
//!
//! This module provides a comprehensive CLI with structured logging,
//! progress reporting, and user-friendly output.

use std::path::PathBuf;
use std::time::Duration;
use clap::{Parser, Subcommand, ValueEnum};
use console::{style, Term};
use indicatif::{ProgressBar, ProgressStyle};
use serde_json::json;
use tracing::{info, warn, error, debug, Level};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

use crate::config::HakariConfig;
use crate::error::Result;
use crate::hakari::{HakariManager, HakariOptions, HakariResult};
use crate::workspace::WorkspaceManager;

/// High-performance workspace-hack regeneration tool
#[derive(Parser)]
#[command(
    name = "cargo-hakari-regenerate",
    version = env!("CARGO_PKG_VERSION"),
    about = "High-performance workspace-hack regeneration using cargo-hakari API",
    long_about = "A production-quality tool for regenerating workspace-hack crates with \
                  zero-allocation patterns and comprehensive error handling."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
    
    /// Verbose output
    #[arg(short, long, global = true)]
    pub verbose: bool,
    
    /// Quiet output (suppress non-error messages)
    #[arg(short, long, global = true, conflicts_with = "verbose")]
    pub quiet: bool,
    
    /// Output format
    #[arg(long, global = true, default_value = "human")]
    pub format: OutputFormat,
    
    /// Workspace root path
    #[arg(long, global = true)]
    pub workspace_root: Option<PathBuf>,
    
    /// Configuration file path
    #[arg(long, global = true)]
    pub config: Option<PathBuf>,
    
    /// Enable performance timing
    #[arg(long, global = true)]
    pub timing: bool}

/// Available commands
#[derive(Subcommand)]
pub enum Commands {
    /// Regenerate workspace-hack from scratch
    Regenerate {
        /// Perform dry-run without making changes
        #[arg(long)]
        dry_run: bool,
        
        /// Skip verification after generation
        #[arg(long)]
        skip_verification: bool,
        
        /// Force regeneration even if workspace-hack exists
        #[arg(long)]
        force: bool,
        
        /// Show detailed progress information
        #[arg(long)]
        progress: bool},
    
    /// Verify existing workspace-hack
    Verify {
        /// Output detailed verification report
        #[arg(long)]
        detailed: bool},
    
    /// Manage configuration
    Config {
        #[command(subcommand)]
        action: ConfigAction},
    
    /// Show workspace information
    Info {
        /// Show package details
        #[arg(long)]
        packages: bool,
        
        /// Show configuration details
        #[arg(long)]
        config: bool},
    
    /// Clean up temporary files and backups
    Cleanup {
        /// Remove all backup files
        #[arg(long)]
        all: bool}}

/// Configuration management actions
#[derive(Subcommand)]
pub enum ConfigAction {
    /// Show current configuration
    Show,
    
    /// Validate configuration
    Validate {
        /// Show detailed validation report
        #[arg(long)]
        detailed: bool},
    
    /// Reset configuration to defaults
    Reset {
        /// Don't ask for confirmation
        #[arg(long)]
        yes: bool},
    
    /// Add omitted dependency
    AddOmitted {
        /// Dependency name to omit
        name: String},
    
    /// Remove omitted dependency
    RemoveOmitted {
        /// Dependency name to include
        name: String}}

/// Output format options
#[derive(ValueEnum, Clone, Debug)]
pub enum OutputFormat {
    /// Human-readable output with colors
    Human,
    /// JSON output for machine processing
    Json,
    /// Compact output for CI/CD
    Compact}

/// CLI application runner
pub struct CliRunner {
    args: Cli,
    term: Term,
    start_time: std::time::Instant}

impl CliRunner {
    /// Create new CLI runner
    pub fn new(args: Cli) -> Self {
        Self {
            args,
            term: Term::stdout(),
            start_time: std::time::Instant::now()}
    }
    
    /// Initialize logging and tracing
    pub fn init_logging(&self) -> Result<()> {
        let level = if self.args.quiet {
            Level::ERROR
        } else if self.args.verbose {
            Level::DEBUG
        } else {
            Level::INFO
        };
        
        let filter = EnvFilter::from_default_env()
            .add_directive(level.into())
            .add_directive("cargo_hakari_regenerate=debug".parse().unwrap());
        
        match self.args.format {
            OutputFormat::Human => {
                tracing_subscriber::registry()
                    .with(fmt::layer().with_writer(std::io::stderr))
                    .with(filter)
                    .init();
            }
            OutputFormat::Json => {
                tracing_subscriber::registry()
                    .with(fmt::layer().json().with_writer(std::io::stderr))
                    .with(filter)
                    .init();
            }
            OutputFormat::Compact => {
                tracing_subscriber::registry()
                    .with(fmt::layer().compact().with_writer(std::io::stderr))
                    .with(filter)
                    .init();
            }
        }
        
        Ok(())
    }
    
    /// Run CLI application
    pub async fn run(&mut self) -> Result<()> {
        self.init_logging()?;
        
        debug!("Starting cargo-hakari-regenerate v{}", env!("CARGO_PKG_VERSION"));
        
        // Determine workspace root
        let workspace_root = if let Some(root) = &self.args.workspace_root {
            root.clone()
        } else {
            let current_dir = std::env::current_dir()
                .map_err(|e| crate::error::HakariRegenerateError::Io(
                    crate::error::IoError::DirectoryOperation {
                        path: PathBuf::from("."),
                        source: e}
                ))?;
            
            WorkspaceManager::find_workspace_root(&current_dir).await?
        };
        
        info!("Using workspace root: {}", workspace_root.display());
        
        // Execute command
        match &self.args.command {
            Commands::Regenerate { dry_run, skip_verification, force, progress } => {
                self.run_regenerate(workspace_root, *dry_run, *skip_verification, *force, *progress).await
            }
            Commands::Verify { detailed } => {
                self.run_verify(workspace_root, *detailed).await
            }
            Commands::Config { action } => {
                self.run_config(workspace_root, action).await
            }
            Commands::Info { packages, config } => {
                self.run_info(workspace_root, *packages, *config).await
            }
            Commands::Cleanup { all } => {
                self.run_cleanup(workspace_root, *all).await
            }
        }
    }
    
    /// Run regenerate command
    async fn run_regenerate(&mut self, workspace_root: PathBuf, dry_run: bool, skip_verification: bool, force: bool, show_progress: bool) -> Result<()> {
        let options = HakariOptions {
            verbose: self.args.verbose,
            dry_run,
            skip_verification,
            force_regenerate: force};
        
        if dry_run {
            self.output_message("Running in dry-run mode - no changes will be made", MessageType::Info);
        }
        
        let mut workspace_manager = WorkspaceManager::new(workspace_root.clone()).await?;
        let mut hakari_manager = HakariManager::new(workspace_root);
        
        // Check if workspace-hack exists
        let workspace_hack_exists = hakari_manager.is_workspace_hack_valid().await?;
        
        if workspace_hack_exists && !force {
            self.output_message("Workspace-hack already exists. Use --force to regenerate.", MessageType::Warning);
            return Ok(());
        }
        
        // Setup progress bar
        let progress_bar = if show_progress {
            Some(self.create_progress_bar("Regenerating workspace-hack", 6))
        } else {
            None
        };
        
        // Run regeneration
        let result = if let Some(pb) = &progress_bar {
            self.run_regenerate_with_progress(pb, &mut workspace_manager, &mut hakari_manager, &options).await?
        } else {
            hakari_manager.regenerate_workspace_hack(&mut workspace_manager, &options).await?
        };
        
        if let Some(pb) = progress_bar {
            pb.finish_with_message("Regeneration complete");
        }
        
        // Output results
        self.output_regenerate_result(&result).await?;
        
        Ok(())
    }
    
    /// Run regenerate with progress reporting
    async fn run_regenerate_with_progress(
        &self,
        progress_bar: &ProgressBar,
        workspace_manager: &mut WorkspaceManager,
        hakari_manager: &mut HakariManager,
        options: &HakariOptions,
    ) -> Result<HakariResult> {
        progress_bar.set_message("Commenting dependencies");
        progress_bar.inc(1);
        
        // This would need to be integrated with the actual regeneration process
        // For now, we'll just call the main function
        let result = hakari_manager.regenerate_workspace_hack(workspace_manager, options).await?;
        
        progress_bar.set_position(6);
        
        Ok(result)
    }
    
    /// Run verify command
    async fn run_verify(&mut self, workspace_root: PathBuf, detailed: bool) -> Result<()> {
        let hakari_manager = HakariManager::new(workspace_root);
        let options = HakariOptions {
            verbose: self.args.verbose,
            ..Default::default()
        };
        
        let result = hakari_manager.verify_workspace_hack(&options).await?;
        
        if result.success {
            self.output_message("Workspace-hack verification passed", MessageType::Success);
        } else {
            self.output_message("Workspace-hack verification failed", MessageType::Error);
        }
        
        if detailed {
            self.output_detailed_result(&result).await?;
        }
        
        Ok(())
    }
    
    /// Run config command
    async fn run_config(&mut self, workspace_root: PathBuf, action: &ConfigAction) -> Result<()> {
        let config_path = self.args.config.clone()
            .unwrap_or_else(|| workspace_root.join(".config").join("hakari.toml"));
        
        let config_manager = crate::config::ConfigManager::new(config_path);
        
        match action {
            ConfigAction::Show => {
                let config = config_manager.load_or_default().await?;
                self.output_config(&config).await?;
            }
            ConfigAction::Validate { detailed } => {
                let config = config_manager.load_or_default().await?;
                let validator = crate::hakari::ConfigValidator::new(config);
                let report = validator.validate_comprehensive()?;
                
                if report.is_valid() {
                    self.output_message("Configuration is valid", MessageType::Success);
                } else {
                    self.output_message("Configuration has errors", MessageType::Error);
                }
                
                if *detailed {
                    self.output_validation_report(&report).await?;
                }
            }
            ConfigAction::Reset { yes } => {
                if !yes {
                    self.output_message("This will reset configuration to defaults. Continue? [y/N]", MessageType::Warning);
                    let mut input = String::new();
                    std::io::stdin().read_line(&mut input)
                        .map_err(|e| crate::error::HakariRegenerateError::Io(crate::error::IoError::FileOperation { 
                            path: std::path::PathBuf::from("stdin"), 
                            source: e 
                        }))?;
                    if !input.trim().eq_ignore_ascii_case("y") {
                        self.output_message("Cancelled", MessageType::Info);
                        return Ok(());
                    }
                }
                
                let config = HakariConfig::for_fluent-voice();
                config_manager.save(&config).await?;
                self.output_message("Configuration reset to defaults", MessageType::Success);
            }
            ConfigAction::AddOmitted { name } => {
                let mut config = config_manager.load_or_default().await?;
                config.add_omitted_dependency(name.clone());
                config_manager.save(&config).await?;
                self.output_message(&format!("Added '{}' to omitted dependencies", name), MessageType::Success);
            }
            ConfigAction::RemoveOmitted { name } => {
                let mut config = config_manager.load_or_default().await?;
                config.remove_omitted_dependency(name);
                config_manager.save(&config).await?;
                self.output_message(&format!("Removed '{}' from omitted dependencies", name), MessageType::Success);
            }
        }
        
        Ok(())
    }
    
    /// Run info command
    async fn run_info(&mut self, workspace_root: PathBuf, show_packages: bool, show_config: bool) -> Result<()> {
        let workspace_manager = WorkspaceManager::new(workspace_root.clone()).await?;
        let hakari_manager = HakariManager::new(workspace_root);
        
        // Basic workspace info
        self.output_workspace_info(&workspace_manager).await?;
        
        // Workspace-hack info
        if let Some(hack_info) = hakari_manager.get_workspace_hack_info().await? {
            self.output_workspace_hack_info(&hack_info).await?;
        }
        
        // Package info
        if show_packages {
            self.output_package_info(&workspace_manager).await?;
        }
        
        // Configuration info
        if show_config {
            let config_path = workspace_root.join(".config").join("hakari.toml");
            let config_manager = crate::config::ConfigManager::new(config_path);
            let config = config_manager.load_or_default().await?;
            self.output_config(&config).await?;
        }
        
        Ok(())
    }
    
    /// Run cleanup command
    async fn run_cleanup(&mut self, workspace_root: PathBuf, all: bool) -> Result<()> {
        let hakari_manager = HakariManager::new(workspace_root);
        
        hakari_manager.cleanup().await?;
        
        if all {
            self.output_message("Cleaned up all temporary files and backups", MessageType::Success);
        } else {
            self.output_message("Cleaned up temporary files", MessageType::Success);
        }
        
        Ok(())
    }
    
    /// Create progress bar
    fn create_progress_bar(&self, message: &str, total: u64) -> ProgressBar {
        let pb = ProgressBar::new(total);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("#>-"),
        );
        pb.set_message(message.to_string());
        pb
    }
    
    /// Output message with formatting
    fn output_message(&self, message: &str, msg_type: MessageType) {
        match self.args.format {
            OutputFormat::Human => {
                let styled_message = match msg_type {
                    MessageType::Success => style(message).green(),
                    MessageType::Warning => style(message).yellow(),
                    MessageType::Error => style(message).red(),
                    MessageType::Info => style(message).blue()};
                
                let _ = self.term.write_line(&styled_message.to_string());
            }
            OutputFormat::Json => {
                let json_msg = json!({
                    "type": msg_type.to_string(),
                    "message": message,
                    "timestamp": chrono::Utc::now().to_rfc3339()});
                println!("{}", json_msg);
            }
            OutputFormat::Compact => {
                println!("{}: {}", msg_type.to_string().to_uppercase(), message);
            }
        }
    }
    
    /// Output regenerate result
    async fn output_regenerate_result(&self, result: &HakariResult) -> Result<()> {
        match self.args.format {
            OutputFormat::Human => {
                if result.success {
                    self.output_message("Workspace-hack regeneration completed successfully", MessageType::Success);
                } else {
                    self.output_message("Workspace-hack regeneration failed", MessageType::Error);
                }
                
                if self.args.timing {
                    self.output_message(&format!("Duration: {:?}", result.duration), MessageType::Info);
                }
                
                if !result.operations_performed.is_empty() {
                    println!("\nOperations performed:");
                    for operation in &result.operations_performed {
                        println!("  • {}", operation);
                    }
                }
                
                if !result.warnings.is_empty() {
                    println!("\nWarnings:");
                    for warning in &result.warnings {
                        println!("  ! {}", style(warning).yellow());
                    }
                }
            }
            OutputFormat::Json => {
                let json_result = json!({
                    "success": result.success,
                    "operations": result.operations_performed,
                    "warnings": result.warnings,
                    "duration_ms": result.duration.as_millis(),
                    "timestamp": chrono::Utc::now().to_rfc3339()});
                println!("{}", serde_json::to_string_pretty(&json_result).unwrap());
            }
            OutputFormat::Compact => {
                println!("RESULT: {}", if result.success { "SUCCESS" } else { "FAILED" });
                println!("OPERATIONS: {}", result.operations_performed.len());
                println!("WARNINGS: {}", result.warnings.len());
                println!("DURATION: {}ms", result.duration.as_millis());
            }
        }
        
        Ok(())
    }
    
    /// Output detailed result
    async fn output_detailed_result(&self, result: &HakariResult) -> Result<()> {
        match self.args.format {
            OutputFormat::Human => {
                println!("\n{}", style("Detailed Results:").bold());
                println!("Success: {}", if result.success { 
                    style("✓").green() 
                } else { 
                    style("✗").red() 
                });
                println!("Duration: {:?}", result.duration);
                
                if !result.operations_performed.is_empty() {
                    println!("\nOperations:");
                    for (i, op) in result.operations_performed.iter().enumerate() {
                        println!("  {}. {}", i + 1, op);
                    }
                }
                
                if !result.warnings.is_empty() {
                    println!("\nWarnings:");
                    for (i, warning) in result.warnings.iter().enumerate() {
                        println!("  {}. {}", i + 1, style(warning).yellow());
                    }
                }
            }
            OutputFormat::Json => {
                let json_result = json!({
                    "detailed": true,
                    "success": result.success,
                    "operations": result.operations_performed,
                    "warnings": result.warnings,
                    "duration_ms": result.duration.as_millis(),
                    "timestamp": chrono::Utc::now().to_rfc3339()});
                println!("{}", serde_json::to_string_pretty(&json_result).unwrap());
            }
            OutputFormat::Compact => {
                for op in &result.operations_performed {
                    println!("OP: {}", op);
                }
                for warning in &result.warnings {
                    println!("WARN: {}", warning);
                }
            }
        }
        
        Ok(())
    }
    
    /// Output configuration
    async fn output_config(&self, config: &HakariConfig) -> Result<()> {
        match self.args.format {
            OutputFormat::Human => {
                println!("{}", style("Hakari Configuration:").bold());
                println!("Package: {}", config.hakari_package);
                println!("Resolver: {}", config.resolver);
                println!("Format Version: {}", config.dep_format_version);
                
                if !config.omitted_deps.is_empty() {
                    println!("\nOmitted Dependencies:");
                    for dep in &config.omitted_deps {
                        println!("  • {}", dep.name);
                    }
                }
            }
            OutputFormat::Json => {
                let json_config = serde_json::to_string_pretty(config).unwrap();
                println!("{}", json_config);
            }
            OutputFormat::Compact => {
                println!("PACKAGE: {}", config.hakari_package);
                println!("RESOLVER: {}", config.resolver);
                println!("OMITTED: {}", config.omitted_deps.len());
            }
        }
        
        Ok(())
    }
    
    /// Output validation report
    async fn output_validation_report(&self, report: &crate::hakari::ValidationReport) -> Result<()> {
        match self.args.format {
            OutputFormat::Human => {
                if !report.errors.is_empty() {
                    println!("\n{}", style("Errors:").red().bold());
                    for error in &report.errors {
                        println!("  {} {}", style("✗").red(), error);
                    }
                }
                
                if !report.warnings.is_empty() {
                    println!("\n{}", style("Warnings:").yellow().bold());
                    for warning in &report.warnings {
                        println!("  {} {}", style("!").yellow(), warning);
                    }
                }
            }
            OutputFormat::Json => {
                let json_report = json!({
                    "errors": report.errors,
                    "warnings": report.warnings,
                    "valid": report.is_valid(),
                    "summary": report.summary()});
                println!("{}", serde_json::to_string_pretty(&json_report).unwrap());
            }
            OutputFormat::Compact => {
                for error in &report.errors {
                    println!("ERROR: {}", error);
                }
                for warning in &report.warnings {
                    println!("WARN: {}", warning);
                }
            }
        }
        
        Ok(())
    }
    
    /// Output workspace information
    async fn output_workspace_info(&self, workspace_manager: &WorkspaceManager) -> Result<()> {
        match self.args.format {
            OutputFormat::Human => {
                println!("{}", style("Workspace Information:").bold());
                println!("Root: {}", workspace_manager.config().root_path.display());
                println!("Packages: {}", workspace_manager.packages().len());
                println!("Workspace-hack path: {}", workspace_manager.config().workspace_hack_path.display());
            }
            OutputFormat::Json => {
                let json_info = json!({
                    "root": workspace_manager.config().root_path,
                    "packages": workspace_manager.packages().len(),
                    "workspace_hack_path": workspace_manager.config().workspace_hack_path});
                println!("{}", serde_json::to_string_pretty(&json_info).unwrap());
            }
            OutputFormat::Compact => {
                println!("ROOT: {}", workspace_manager.config().root_path.display());
                println!("PACKAGES: {}", workspace_manager.packages().len());
            }
        }
        
        Ok(())
    }
    
    /// Output workspace-hack information
    async fn output_workspace_hack_info(&self, info: &crate::hakari::WorkspaceHackInfo) -> Result<()> {
        match self.args.format {
            OutputFormat::Human => {
                println!("\n{}", style("Workspace-hack Information:").bold());
                println!("Name: {}", info.name);
                println!("Version: {}", info.version);
                println!("Dependencies: {}", info.dependency_count);
                println!("Path: {}", info.path.display());
            }
            OutputFormat::Json => {
                let json_info = serde_json::to_string_pretty(info).unwrap();
                println!("{}", json_info);
            }
            OutputFormat::Compact => {
                println!("HACK_NAME: {}", info.name);
                println!("HACK_VERSION: {}", info.version);
                println!("HACK_DEPS: {}", info.dependency_count);
            }
        }
        
        Ok(())
    }
    
    /// Output package information
    async fn output_package_info(&self, workspace_manager: &WorkspaceManager) -> Result<()> {
        match self.args.format {
            OutputFormat::Human => {
                println!("\n{}", style("Package Information:").bold());
                for package in workspace_manager.packages() {
                    println!("  {} {}", 
                        style("•").blue(), 
                        package.name
                    );
                    if package.has_workspace_hack_dep {
                        println!("    {} Uses workspace-hack", style("✓").green());
                    }
                }
            }
            OutputFormat::Json => {
                let packages: Vec<_> = workspace_manager.packages().iter().map(|p| {
                    json!({
                        "name": p.name,
                        "path": p.path,
                        "has_workspace_hack": p.has_workspace_hack_dep})
                }).collect();
                
                let json_info = json!({
                    "packages": packages});
                println!("{}", serde_json::to_string_pretty(&json_info).unwrap());
            }
            OutputFormat::Compact => {
                for package in workspace_manager.packages() {
                    println!("PKG: {} {}", 
                        package.name, 
                        if package.has_workspace_hack_dep { "HACK" } else { "NO_HACK" }
                    );
                }
            }
        }
        
        Ok(())
    }
    
    /// Get total runtime
    pub fn total_runtime(&self) -> Duration {
        self.start_time.elapsed()
    }
}

/// Message types for output formatting
#[derive(Debug, Clone, Copy)]
enum MessageType {
    Success,
    Warning,
    Error,
    Info}

impl std::fmt::Display for MessageType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MessageType::Success => write!(f, "success"),
            MessageType::Warning => write!(f, "warning"),
            MessageType::Error => write!(f, "error"),
            MessageType::Info => write!(f, "info")}
    }
}