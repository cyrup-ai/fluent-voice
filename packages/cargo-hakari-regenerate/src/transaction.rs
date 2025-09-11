//! Transaction system for atomic operations and rollback capability
//!
//! This module provides a comprehensive transaction system that ensures
//! workspace operations can be rolled back safely in case of failures.

use std::path::{Path, PathBuf};
use scopeguard::defer;
use smallvec::SmallVec;
use tempfile::NamedTempFile;
use tokio::fs;
use crate::error::{TransactionError, Result, IoErrorExt};

/// Transaction operation types
#[derive(Debug, Clone)]
pub enum Operation {
    /// File was created
    FileCreated { path: PathBuf },
    
    /// File was modified (stores original content)
    FileModified { path: PathBuf, original_content: Vec<u8> },
    
    /// File was deleted (stores original content)
    FileDeleted { path: PathBuf, original_content: Vec<u8> },
    
    /// Directory was created
    DirectoryCreated { path: PathBuf },
    
    /// Directory was deleted
    DirectoryDeleted { path: PathBuf },
    
    /// Configuration was backed up
    ConfigBackup { from: PathBuf, to: PathBuf },
    
    /// Workspace hack was initialized
    WorkspaceHackInit { path: PathBuf },
    
    /// Package was renamed
    PackageRenamed { path: PathBuf, old_name: String, new_name: String }}

/// Transaction checkpoint for nested operations
#[derive(Debug, Clone)]
pub struct Checkpoint {
    name: String,
    operation_count: usize,
    created_at: std::time::Instant}

/// High-performance transaction system with rollback capability
pub struct Transaction {
    /// Operations performed in this transaction
    operations: SmallVec<[Operation; 32]>,
    
    /// Named checkpoints for partial rollback
    checkpoints: HashMap<String, Checkpoint>,
    
    /// Temporary files created during transaction
    temp_files: SmallVec<[NamedTempFile; 8]>,
    
    /// Transaction state
    state: TransactionState,
    
    /// Started timestamp
    started_at: std::time::Instant,
    
    /// Whether to automatically rollback on drop
    auto_rollback: bool}

/// Transaction state
#[derive(Debug, Clone, Copy, PartialEq)]
enum TransactionState {
    NotStarted,
    Active,
    Committed,
    RolledBack,
    Failed}

impl Transaction {
    /// Create new transaction
    pub fn new() -> Self {
        Self {
            operations: SmallVec::new(),
            checkpoints: HashMap::new(),
            temp_files: SmallVec::new(),
            state: TransactionState::NotStarted,
            started_at: std::time::Instant::now(),
            auto_rollback: true}
    }
    
    /// Start transaction
    pub fn start(&mut self) -> Result<()> {
        if self.state != TransactionState::NotStarted {
            return Err(TransactionError::AlreadyStarted.into());
        }
        
        self.state = TransactionState::Active;
        self.started_at = std::time::Instant::now();
        
        Ok(())
    }
    
    /// Create checkpoint
    pub fn checkpoint(&mut self, name: impl Into<String>) -> Result<()> {
        if self.state != TransactionState::Active {
            return Err(TransactionError::NotStarted.into());
        }
        
        let name = name.into();
        let checkpoint = Checkpoint {
            name: name.clone(),
            operation_count: self.operations.len(),
            created_at: std::time::Instant::now()};
        
        self.checkpoints.insert(name, checkpoint);
        
        Ok(())
    }
    
    /// Record file creation
    pub fn record_file_created(&mut self, path: PathBuf) -> Result<()> {
        if self.state != TransactionState::Active {
            return Err(TransactionError::NotStarted.into());
        }
        
        self.operations.push(Operation::FileCreated { path });
        
        Ok(())
    }
    
    /// Record file modification (preserves original content)
    pub async fn record_file_modified(&mut self, path: PathBuf) -> Result<()> {
        if self.state != TransactionState::Active {
            return Err(TransactionError::NotStarted.into());
        }
        
        let original_content = if path.exists() {
            fs::read(&path)
                .await
                .with_path(path.clone())?
        } else {
            Vec::new()
        };
        
        self.operations.push(Operation::FileModified { path, original_content });
        
        Ok(())
    }
    
    /// Record file deletion
    pub async fn record_file_deleted(&mut self, path: PathBuf) -> Result<()> {
        if self.state != TransactionState::Active {
            return Err(TransactionError::NotStarted.into());
        }
        
        let original_content = fs::read(&path)
            .await
            .with_path(path.clone())?;
        
        self.operations.push(Operation::FileDeleted { path, original_content });
        
        Ok(())
    }
    
    /// Record directory creation
    pub fn record_directory_created(&mut self, path: PathBuf) -> Result<()> {
        if self.state != TransactionState::Active {
            return Err(TransactionError::NotStarted.into());
        }
        
        self.operations.push(Operation::DirectoryCreated { path });
        
        Ok(())
    }
    
    /// Record configuration backup
    pub fn record_config_backup(&mut self, from: PathBuf, to: PathBuf) -> Result<()> {
        if self.state != TransactionState::Active {
            return Err(TransactionError::NotStarted.into());
        }
        
        self.operations.push(Operation::ConfigBackup { from, to });
        
        Ok(())
    }
    
    /// Record workspace hack initialization
    pub fn record_workspace_hack_init(&mut self, path: PathBuf) -> Result<()> {
        if self.state != TransactionState::Active {
            return Err(TransactionError::NotStarted.into());
        }
        
        self.operations.push(Operation::WorkspaceHackInit { path });
        
        Ok(())
    }
    
    /// Record package rename
    pub fn record_package_renamed(&mut self, path: PathBuf, old_name: String, new_name: String) -> Result<()> {
        if self.state != TransactionState::Active {
            return Err(TransactionError::NotStarted.into());
        }
        
        self.operations.push(Operation::PackageRenamed { path, old_name, new_name });
        
        Ok(())
    }
    
    /// Create temporary file within transaction
    pub fn create_temp_file(&mut self) -> Result<&NamedTempFile> {
        if self.state != TransactionState::Active {
            return Err(TransactionError::NotStarted.into());
        }
        
        let temp_file = NamedTempFile::new()
            .map_err(|e| TransactionError::CheckpointFailed {
                name: "temp_file_creation".to_string()})?;
        
        self.temp_files.push(temp_file);
        
        Ok(self.temp_files.last().unwrap())
    }
    
    /// Perform atomic file write
    pub async fn atomic_write(&mut self, path: &Path, content: &[u8]) -> Result<()> {
        if self.state != TransactionState::Active {
            return Err(TransactionError::NotStarted.into());
        }
        
        // Record the modification first
        self.record_file_modified(path.to_path_buf()).await?;
        
        // Create temporary file
        let temp_file = self.create_temp_file()?;
        
        // Write content to temporary file
        fs::write(temp_file.path(), content)
            .await
            .with_path(temp_file.path().to_path_buf())?;
        
        // Atomically move temporary file to target
        fs::rename(temp_file.path(), path)
            .await
            .with_path(path.to_path_buf())?;
        
        Ok(())
    }
    
    /// Commit transaction
    pub async fn commit(&mut self) -> Result<()> {
        if self.state != TransactionState::Active {
            return Err(TransactionError::NotStarted.into());
        }
        
        self.state = TransactionState::Committed;
        self.auto_rollback = false;
        
        // Clean up temporary files
        self.temp_files.clear();
        
        Ok(())
    }
    
    /// Rollback transaction
    pub async fn rollback(&mut self) -> Result<()> {
        if self.state != TransactionState::Active && self.state != TransactionState::Failed {
            return Err(TransactionError::NotStarted.into());
        }
        
        let mut rollback_errors = Vec::new();
        
        // Rollback operations in reverse order
        for operation in self.operations.iter().rev() {
            if let Err(e) = self.rollback_operation(operation).await {
                rollback_errors.push(e);
            }
        }
        
        self.state = TransactionState::RolledBack;
        
        // Clean up temporary files
        self.temp_files.clear();
        
        if !rollback_errors.is_empty() {
            return Err(TransactionError::RollbackFailed {
                reason: format!("Multiple rollback errors: {:?}", rollback_errors)}.into());
        }
        
        Ok(())
    }
    
    /// Rollback to specific checkpoint
    pub async fn rollback_to_checkpoint(&mut self, checkpoint_name: &str) -> Result<()> {
        if self.state != TransactionState::Active {
            return Err(TransactionError::NotStarted.into());
        }
        
        let checkpoint = self.checkpoints.get(checkpoint_name)
            .ok_or_else(|| TransactionError::CheckpointFailed {
                name: checkpoint_name.to_string()})?;
        
        let target_count = checkpoint.operation_count;
        
        // Rollback operations added after checkpoint
        let operations_to_rollback = self.operations.split_off(target_count);
        
        for operation in operations_to_rollback.iter().rev() {
            self.rollback_operation(operation).await?;
        }
        
        Ok(())
    }
    
    /// Rollback single operation
    async fn rollback_operation(&self, operation: &Operation) -> Result<()> {
        match operation {
            Operation::FileCreated { path } => {
                if path.exists() {
                    fs::remove_file(path)
                        .await
                        .with_path(path.clone())?;
                }
            }
            
            Operation::FileModified { path, original_content } => {
                if original_content.is_empty() && path.exists() {
                    fs::remove_file(path)
                        .await
                        .with_path(path.clone())?;
                } else if !original_content.is_empty() {
                    fs::write(path, original_content)
                        .await
                        .with_path(path.clone())?;
                }
            }
            
            Operation::FileDeleted { path, original_content } => {
                fs::write(path, original_content)
                    .await
                    .with_path(path.clone())?;
            }
            
            Operation::DirectoryCreated { path } => {
                if path.exists() {
                    fs::remove_dir_all(path)
                        .await
                        .with_path(path.clone())?;
                }
            }
            
            Operation::DirectoryDeleted { path } => {
                fs::create_dir_all(path)
                    .await
                    .with_path(path.clone())?;
            }
            
            Operation::ConfigBackup { from, to } => {
                if to.exists() {
                    fs::copy(to, from)
                        .await
                        .with_path(from.clone())?;
                }
            }
            
            Operation::WorkspaceHackInit { path } => {
                if path.exists() {
                    fs::remove_dir_all(path)
                        .await
                        .with_path(path.clone())?;
                }
            }
            
            Operation::PackageRenamed { path, old_name, .. } => {
                // This would require more complex rollback logic
                // For now, log the issue
                tracing::warn!("Package rename rollback not implemented for {:?}", path);
            }
        }
        
        Ok(())
    }
    
    /// Get transaction duration
    pub fn duration(&self) -> std::time::Duration {
        self.started_at.elapsed()
    }
    
    /// Get operation count
    pub fn operation_count(&self) -> usize {
        self.operations.len()
    }
    
    /// Check if transaction is active
    pub fn is_active(&self) -> bool {
        self.state == TransactionState::Active
    }
    
    /// Check if transaction is committed
    pub fn is_committed(&self) -> bool {
        self.state == TransactionState::Committed
    }
    
    /// Disable automatic rollback on drop
    pub fn disable_auto_rollback(&mut self) {
        self.auto_rollback = false;
    }
}

impl Drop for Transaction {
    fn drop(&mut self) {
        if self.auto_rollback && self.state == TransactionState::Active {
            // Use defer to ensure rollback happens even if panic occurs
            defer! {
                if let Err(e) = futures::executor::block_on(self.rollback()) {
                    tracing::error!("Failed to rollback transaction on drop: {}", e);
                }
            }
        }
    }
}

/// Transaction builder for fluent interface
pub struct TransactionBuilder {
    auto_rollback: bool}

impl TransactionBuilder {
    /// Create new transaction builder
    pub fn new() -> Self {
        Self {
            auto_rollback: true}
    }
    
    /// Enable automatic rollback on drop
    pub fn auto_rollback(mut self, enabled: bool) -> Self {
        self.auto_rollback = enabled;
        self
    }
    
    /// Build and start transaction
    pub fn build(self) -> Result<Transaction> {
        let mut transaction = Transaction::new();
        transaction.auto_rollback = self.auto_rollback;
        transaction.start()?;
        
        Ok(transaction)
    }
}

/// Convenience macro for creating transactions
#[macro_export]
macro_rules! transaction {
    ($($config:tt)*) => {
        $crate::transaction::TransactionBuilder::new()
            $($config)*
            .build()
    };
}

/// Async transaction guard that ensures rollback on panic
pub struct AsyncTransactionGuard {
    transaction: Option<Transaction>}

impl AsyncTransactionGuard {
    /// Create new transaction guard
    pub fn new(transaction: Transaction) -> Self {
        Self {
            transaction: Some(transaction)}
    }
    
    /// Get mutable reference to transaction
    pub fn transaction(&mut self) -> &mut Transaction {
        self.transaction.as_mut().unwrap()
    }
    
    /// Commit transaction and consume guard
    pub async fn commit(mut self) -> Result<()> {
        if let Some(mut transaction) = self.transaction.take() {
            transaction.commit().await?;
        }
        
        Ok(())
    }
    
    /// Rollback transaction and consume guard
    pub async fn rollback(mut self) -> Result<()> {
        if let Some(mut transaction) = self.transaction.take() {
            transaction.rollback().await?;
        }
        
        Ok(())
    }
}

impl Drop for AsyncTransactionGuard {
    fn drop(&mut self) {
        if let Some(mut transaction) = self.transaction.take() {
            if transaction.is_active() {
                tokio::task::block_in_place(|| {
                    futures::executor::block_on(async {
                        if let Err(e) = transaction.rollback().await {
                            tracing::error!("Failed to rollback transaction in guard: {}", e);
                        }
                    })
                });
            }
        }
    }
}