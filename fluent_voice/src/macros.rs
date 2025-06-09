//! Internal helper for the `Ok ⇒ / Err ⇒` matcher block.

/// Internal helper macro for the matcher closure syntax.
///
/// This macro transforms the `{ Ok => ..., Err(e) => ... }` syntax
/// into a proper async closure that can be used by engine implementations.
/// Users typically don't interact with this macro directly.
#[doc(hidden)]
#[macro_export]
macro_rules! fv_match {
    (|$conv:ident| {
        Ok  => $ok:expr,
        Err($err:ident) => $errexpr:expr $(,)?
    }) => {
        |result| async move {
            match result {
                Ok($conv) => $ok,
                Err($err) => $errexpr,
            }
        }
    };
}
