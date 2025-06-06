//! fluent_voice/src/internal_macro.rs
//! ----------------------------------
//! Internal macro for handler syntax sugar

/// **INTERNAL** helper macro.  Rewrites the
/// `Ok => … , Err(err) => …` fragment used inside `.play( … )`.
#[doc(hidden)]
#[macro_export]
macro_rules! __play_handler {
    (|$audio:ident| {
        Ok => $ok:expr,
        Err($err:ident) => $errexpr:expr $(,)?
    }) => {
        |result| async move {
            match result {
                Ok($audio) => { $ok },
                Err($err)  => { $errexpr },
            }
        }
    };
}