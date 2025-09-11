//! Arrow syntax support for cyrup_sugars JSON patterns
//!
//! This module provides traits and implementations that enable the arrow syntax
//! used in the examples: `Ok => value, Err(e) => Err(e)`

use crate::VoiceError;

/// Trait for closures that can handle arrow syntax transformation
pub trait ArrowSyntaxClosure<T, R> {
    /// Transform the closure to handle Result types
    fn transform(self) -> Box<dyn FnMut(Result<T, VoiceError>) -> R + Send + 'static>;
}

/// Implementation for regular closures that already handle Result types
impl<F, T, R> ArrowSyntaxClosure<T, R> for F
where
    F: FnMut(Result<T, VoiceError>) -> R + Send + 'static,
{
    fn transform(self) -> Box<dyn FnMut(Result<T, VoiceError>) -> R + Send + 'static> {
        Box::new(self)
    }
}

/// Macro to enable arrow syntax in method calls
#[macro_export]
macro_rules! arrow_syntax {
    // Transform synthesize calls with arrow syntax
    (
        $builder:expr => synthesize(|$param:ident| {
            Ok => $ok_expr:expr,
            Err($err:ident) => $err_expr:expr $(,)?
        })
    ) => {
        $builder.synthesize(|result| match result {
            Ok($param) => $ok_expr,
            Err($err) => $err_expr,
        })
    };

    // Transform listen calls with arrow syntax
    (
        $builder:expr => listen(|$param:ident| {
            Ok => $ok_expr:expr,
            Err($err:ident) => $err_expr:expr $(,)?
        })
    ) => {
        $builder.listen(|result| match result {
            Ok($param) => $ok_expr,
            Err($err) => $err_expr,
        })
    };
}

/// Trait for builders that support arrow syntax
pub trait ArrowSyntaxBuilder<T> {}

/// Helper function to create closures with arrow syntax support
pub fn arrow_closure<T, R>(
    f: impl FnOnce(T) -> R + Send + 'static,
) -> impl FnOnce(Result<T, VoiceError>) -> Result<R, VoiceError> + Send + 'static {
    move |result| match result {
        Ok(value) => Ok(f(value)),
        Err(e) => Err(e),
    }
}

/// Macro to automatically transform arrow syntax in the entire scope
#[macro_export]
macro_rules! enable_arrow_syntax {
    ($($item:item)*) => {
        $($item)*
    };
}

/// Re-export for convenience
pub use arrow_syntax;
pub use enable_arrow_syntax;
