// Utility functions for common tasks in the LiveKit client
use std::fmt::Debug;

/// Extension trait for Result to log errors without disrupting the error flow
pub trait ResultExt<T, E: Debug> {
    /// Log an error if the result is an Err, then return the original result
    fn log_err(self) -> Self;
}

impl<T, E: Debug> ResultExt<T, E> for Result<T, E> {
    fn log_err(self) -> Self {
        if let Err(ref e) = self {
            tracing::error!("Error occurred: {:?}", e);
        }
        self
    }
}

/// Create a deferred action that will run when dropped
pub fn defer<F>(f: F) -> impl Drop
where
    F: FnOnce(),
{
    struct Defer<F>(Option<F>)
    where
        F: FnOnce();

    impl<F> Drop for Defer<F>
    where
        F: FnOnce(),
    {
        fn drop(&mut self) {
            if let Some(f) = self.0.take() {
                f();
            }
        }
    }

    Defer(Some(f))
}

/// Safe wrapper for optional execution
pub mod maybe {
    /// Try to execute a closure with a value that might be None
    pub fn try_with<T, F, R>(value: Option<&T>, f: F) -> Option<R>
    where
        F: FnOnce(&T) -> R,
    {
        value.map(f)
    }

    /// Execute a closure with a value if it exists, returning a result
    pub fn with<T, F, R, E>(value: Option<&T>, f: F) -> Result<Option<R>, E>
    where
        F: FnOnce(&T) -> Result<R, E>,
    {
        match value {
            Some(v) => f(v).map(Some),
            None => Ok(None),
        }
    }
}
