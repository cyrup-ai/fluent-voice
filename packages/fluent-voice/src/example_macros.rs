//! Macros specifically designed to support the exact syntax used in examples
//!
//! This module provides procedural macros that transform the syntax used in
//! examples into valid Rust code that works with the builder API.

/// Macro to handle the TTS synthesize syntax used in examples
///
/// This macro needs to be applied at the method call site to transform
/// the `Ok =>` syntax into a proper closure.
#[macro_export]
macro_rules! handle_tts_synthesize {
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

/// Macro to handle the STT on_result syntax used in examples
///
/// This extends the cyrup-sugars on_result! macro to handle the specific
/// syntax patterns used in the STT example.
#[macro_export]
macro_rules! fluent_on_result {
    (
        Ok => $ok:expr,
        Err($err:ident) => $err_expr:expr $(,)?
    ) => {
        |result| match result {
            Ok(value) => $ok,
            Err($err) => $err_expr,
        }
    };
}

/// Helper macro for STT listen method
#[macro_export]
macro_rules! handle_stt_listen {
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
pub use fluent_on_result;
pub use handle_stt_listen;
pub use handle_tts_synthesize;
