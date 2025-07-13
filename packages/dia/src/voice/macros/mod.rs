//! Macros for elegant error handling in the voice API

pub mod voice_conversation;

/// Macro for handling Result in conversation speak method
#[macro_export]
macro_rules! speak_handler {
    ($conversation:expr, |$conv:ident| {
        Ok => $ok_expr:expr,
        Err => $err_expr:expr
    }) => {{
        match $conversation.internal_generate().await {
            Ok(player) => {
                let $conv = $conversation;
                $ok_expr
            }
            Err(e) => {
                let $conv = ($conversation, e);
                $err_expr
            }
        }
    }};
}

/// Alternative syntax with type annotations
#[macro_export]
macro_rules! speak_with {
    ($conversation:expr, {
        Ok($player:ident: VoicePlayer) => $ok_block:block,
        Err($error:ident: VoiceError) => $err_block:block
    }) => {{
        match $conversation.internal_generate().await {
            Ok($player) => $ok_block,
            Err($error) => $err_block,
        }
    }};
}
