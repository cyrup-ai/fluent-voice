use crate::error::{Result, SampleError};
use indicatif::{ProgressBar, ProgressStyle};
use std::sync::{
    OnceLock,
    atomic::{AtomicBool, Ordering},
};

static PROGRESS_BAR: OnceLock<ProgressBar> = OnceLock::new();
static IS_INITIALIZED: AtomicBool = AtomicBool::new(false);
static IS_FINISHED: AtomicBool = AtomicBool::new(false);

/// A progress tracker for displaying progress bars during long-running operations.
pub struct ProgressTracker {
    _private: (),
}

impl ProgressTracker {
    /// Creates a new progress tracker with the specified total count and message.
    pub fn new(total: u64, message: &str) -> Result<Self> {
        let pb = ProgressBar::new(total);

        let style = ProgressStyle::default_bar()
            .template("{spinner:.green} {msg}\n[{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})")
            .map_err(|_| SampleError::ProgressStyleCreation)?
            .progress_chars("#>-");

        pb.set_style(style);
        pb.set_message(message.to_string());

        match PROGRESS_BAR.set(pb) {
            Ok(()) => {
                IS_INITIALIZED.store(true, Ordering::Release);
                IS_FINISHED.store(false, Ordering::Release);
                Ok(Self { _private: () })
            }
            Err(_) => Err(SampleError::ProgressAlreadyInitialized),
        }
    }

    #[inline]
    /// Increments the progress bar by one step.
    pub fn inc(&self) -> Result<()> {
        if !IS_INITIALIZED.load(Ordering::Acquire) || IS_FINISHED.load(Ordering::Acquire) {
            return Err(SampleError::ProgressNotInitialized);
        }

        PROGRESS_BAR.get().map_or_else(
            || Err(SampleError::ProgressNotInitialized),
            |pb| {
                pb.inc(1);
                Ok(())
            },
        )
    }

    /// Finishes the progress bar with the specified completion message.
    pub fn finish(&self, message: &str) -> Result<()> {
        if !IS_INITIALIZED.load(Ordering::Acquire) || IS_FINISHED.load(Ordering::Acquire) {
            return Err(SampleError::ProgressNotInitialized);
        }

        PROGRESS_BAR.get().map_or_else(
            || Err(SampleError::ProgressNotInitialized),
            |pb| {
                pb.finish_with_message(message.to_string());
                IS_FINISHED.store(true, Ordering::Release);
                Ok(())
            },
        )
    }

    #[inline]
    /// Checks if the progress tracker has been initialized.
    pub fn is_initialized() -> bool {
        IS_INITIALIZED.load(Ordering::Acquire) && !IS_FINISHED.load(Ordering::Acquire)
    }
}

impl Drop for ProgressTracker {
    fn drop(&mut self) {
        if IS_INITIALIZED.load(Ordering::Acquire)
            && !IS_FINISHED.load(Ordering::Acquire)
            && let Some(pb) = PROGRESS_BAR.get()
        {
            pb.finish_and_clear();
            IS_FINISHED.store(true, Ordering::Release);
        }
    }
}
