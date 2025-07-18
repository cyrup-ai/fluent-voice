//! Arrow syntax support for fluent-voice examples
//!
//! This module provides macro support for the `Ok =>` syntax used in examples.

/// Macro to enable arrow syntax in TTS on_chunk closures
/// 
/// Transforms:
/// ```ignore
/// |synthesis_chunk| {
///     Ok => synthesis_chunk.into(),
///     Err(e) => Err(e),
/// }
/// ```
/// 
/// Into a proper closure that the builder API can handle.
#[macro_export]
macro_rules! tts_on_chunk_arrow {
    (|$param:ident| {
        Ok => $ok:expr,
        Err($err:ident) => $err_expr:expr $(,)?
    }) => {
        |result: Result<_, _>| match result {
            Ok($param) => $ok,
            Err($err) => $err_expr,
        }
    };
}

/// Macro to enable arrow syntax in TTS synthesize closures
/// 
/// Transforms:
/// ```ignore
/// |conversation| {
///     Ok => conversation.into_stream(),
///     Err(e) => Err(e),
/// }
/// ```
/// 
/// Into a proper closure that the builder API can handle.
#[macro_export]
macro_rules! tts_synthesize_arrow {
    (|$param:ident| {
        Ok => $ok:expr,
        Err($err:ident) => $err_expr:expr $(,)?
    }) => {
        |result: Result<_, _>| match result {
            Ok($param) => $ok,
            Err($err) => $err_expr,
        }
    };
}

/// Helper trait to enable arrow syntax on builder methods
pub trait ArrowSyntaxSupport {
    /// Enable arrow syntax for on_chunk method
    fn on_chunk_arrow<F>(self, f: F) -> Self
    where
        F: FnOnce(Result<Vec<u8>, crate::VoiceError>) -> Vec<u8> + Send + 'static;
    
    /// Enable arrow syntax for synthesize method
    fn synthesize_arrow<F, R>(self, f: F) -> impl std::future::Future<Output = R> + Send
    where
        F: FnOnce(Result<Self, crate::VoiceError>) -> R + Send + 'static,
        R: Send + 'static,
        Self: Sized;
}

// Re-export macros for convenience
pub use tts_on_chunk_arrow;
pub use tts_synthesize_arrow;
