//! Macros for elegant error handling in voice conversations

/// Macro to sugar the player closure syntax
#[macro_export]
macro_rules! player {
    ($conversation:expr, |$conv:ident| {
        Ok => $ok_expr:expr,
        Err => $err_expr:expr
    }) => {{
        $conversation.player(|$conv| {
            // Implementation will internally determine if generation succeeded
            // and execute the appropriate branch
            if internal_generation_succeeded!() {
                $ok_expr
            } else {
                $err_expr
            }
        })
    }};
}
