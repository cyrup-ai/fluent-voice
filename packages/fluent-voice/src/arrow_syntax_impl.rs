//! Concrete implementation of arrow syntax for examples
//!
//! This module provides the actual implementation that makes the examples work
//! by transforming the `Ok =>` syntax into valid Rust code.

use crate::VoiceError;
use std::future::Future;
use std::pin::Pin;

/// Extension trait for TTS builders to support arrow syntax
pub trait TtsBuilderArrowSyntax {
    type Output;

    /// synthesize method that handles arrow syntax transformation
    fn synthesize_with_arrow<F, R>(self, f: F) -> Pin<Box<dyn Future<Output = R> + Send>>
    where
        F: FnOnce(Result<Self::Output, VoiceError>) -> R + Send + 'static,
        R: Send + 'static,
        Self::Output: Send + 'static;
}

/// Extension trait for STT builders to support arrow syntax
pub trait SttBuilderArrowSyntax {
    type Output;

    /// listen method that handles arrow syntax transformation
    fn listen_with_arrow<F, R>(self, f: F) -> Pin<Box<dyn Future<Output = R> + Send>>
    where
        F: FnOnce(Result<Self::Output, VoiceError>) -> R + Send + 'static,
        R: Send + 'static,
        Self::Output: Send + 'static;
}

/// Helper function to transform arrow syntax closures
pub fn transform_arrow_closure<T, R, F>(
    f: F,
) -> impl FnOnce(Result<T, VoiceError>) -> R + Send + 'static
where
    F: FnOnce(Result<T, VoiceError>) -> R + Send + 'static,
    T: Send + 'static,
    R: Send + 'static,
{
    f
}

/// Macro to handle TTS synthesize with arrow syntax
#[macro_export]
macro_rules! tts_synthesize_arrow {
    ($builder:expr, |$param:ident| {
        Ok => $ok:expr,
        Err($err:ident) => $err_expr:expr $(,)?
    }) => {
        $builder.synthesize(|result| match result {
            Ok($param) => $ok,
            Err($err) => $err_expr,
        })
    };
}

/// Macro to handle STT listen with arrow syntax
#[macro_export]
macro_rules! stt_listen_arrow {
    ($builder:expr, |$param:ident| {
        Ok => $ok:expr,
        Err($err:ident) => $err_expr:expr $(,)?
    }) => {
        $builder.listen(|result| match result {
            Ok($param) => $ok,
            Err($err) => $err_expr,
        })
    };
}

// Re-export for convenience
pub use stt_listen_arrow;
pub use tts_synthesize_arrow;
