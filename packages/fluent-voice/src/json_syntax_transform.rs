//! JSON Syntax Transformation for cyrup_sugars integration
//!
//! This module provides the infrastructure to make the cyrup_sugars JSON syntax
//! work seamlessly in fluent-voice method calls.

/// Trait for types that support cyrup_sugars JSON syntax transformation
pub trait JsonSyntaxSupport<T> {
    /// Transform a closure with JSON syntax into a regular closure
    fn transform_json_closure<F, R>(self, f: F) -> Self
    where
        F: FnOnce(Result<T, crate::VoiceError>) -> R;
}

/// Helper function to create a closure that matches the cyrup_sugars pattern
#[inline]
pub fn json_closure_match<T, R, F>(f: F) -> impl FnOnce(Result<T, crate::VoiceError>) -> R
where
    F: FnOnce(Result<T, crate::VoiceError>) -> R,
{
    f
}

/// Macro to enable JSON syntax transformation in a module
#[macro_export]
macro_rules! enable_json_syntax {
    () => {
        // Re-export the transformation macros
        use $crate::json_syntax_transform::*;

        // Enable cyrup_sugars syntax
        use cyrup_sugars::macros::*;
        use cyrup_sugars::prelude::*;
    };
}

/// Re-export working transformation macros from the macros module
pub use crate::{fv_match, listen_transform, synthesize_transform};

/// Attribute macro (when procedural macros are available) to enable JSON syntax in a function
#[macro_export]
macro_rules! with_json_syntax {
    (
        $(#[$meta:meta])*
        $vis:vis fn $name:ident($($args:tt)*) -> $ret:ty {
            $($body:tt)*
        }
    ) => {
        $(#[$meta])*
        $vis fn $name($($args)*) -> $ret {
            $crate::enable_json_syntax!();
            $($body)*
        }
    };
}
