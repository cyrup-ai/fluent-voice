//! Internal helpers for the fluent builder syntax.

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
        |result| match result {
            Ok($conv) => $ok,
            Err($err) => $errexpr,
        }
    };
}

/// Macro for synthesize method with cyrup_sugars JSON syntax support.
///
/// This macro enables the exact syntax shown in examples:
/// ```ignore
/// .synthesize(|conversation| {
///     Ok => conversation.into_stream(),
///     Err(e) => Err(e),
/// })
/// ```
#[macro_export]
macro_rules! synthesize_transform {
    (|$conv:ident| {
        Ok => $ok:expr,
        Err($err:ident) => $errexpr:expr $(,)?
    }) => {
        |result: Result<_, _>| match result {
            Ok($conv) => $ok,
            Err($err) => $errexpr,
        }
    };
}

/// Macro for listen method with cyrup_sugars JSON syntax support.
///
/// This macro enables the exact syntax shown in examples:
/// ```ignore
/// .listen(|segment| {
///     Ok => segment.text(),
///     Err(e) => Err(e),
/// })
/// ```
#[macro_export]
macro_rules! listen_transform {
    (|$segment:ident| {
        Ok => $ok:expr,
        Err($err:ident) => $errexpr:expr $(,)?
    }) => {
        |result: Result<_, _>| match result {
            Ok($segment) => $ok,
            Err($err) => $errexpr,
        }
    };
}

/// Macro for TTS conversation builder pattern.
///
/// This macro supports the fluent builder pattern for TTS operations,
/// allowing for a clean syntax with a single await point.
#[macro_export]
macro_rules! tts_conversation_builder {
    ($engine:expr) => {{
        use $crate::builders::TtsConversationBuilderImpl;
        use $crate::tts_conversation::TtsConversationBuilder;

        TtsConversationBuilderImpl::new($engine)
    }};
}

/// TTS synthesize method macro that enables arrow syntax.
///
/// This macro allows TTS builders to use arrow syntax in closures:
/// ```ignore
/// .synthesize(|conversation| {
///     Ok => conversation.into_stream(),
///     Err(e) => Err(e),
/// })
/// ```
#[macro_export]
macro_rules! tts_synthesize {
    ($builder:expr, |$conv:ident| {
        Ok => $ok:expr,
        Err($err:ident) => $errexpr:expr $(,)?
    }) => {
        $builder.synthesize(synthesize_transform!(|$conv| {
            Ok => $ok,
            Err($err) => $errexpr,
        }))
    };
}

// STT conversation builder macro removed - was orphaned code using removed SttConversationBuilderImpl

/// Macro for the listen method on STT microphone builders.
///
/// This macro allows for the simplified `Ok => ..., Err(e) => ...` syntax
/// in the listen method of STT microphone builders.
#[macro_export]
macro_rules! stt_listen {
    ($self:expr, |$conv:ident| {
        Ok  => $ok:expr,
        Err($err:ident) => $errexpr:expr $(,)?
    }) => {
        $self.listen(|result| match result {
            Ok($conv) => $ok,
            Err($err) => $errexpr,
        })
    };
}

/// Macro for the emit method on STT transcription builders.
///
/// This macro allows for the simplified `Ok => ..., Err(e) => ...` syntax
/// in the emit method of STT transcription builders.
#[macro_export]
macro_rules! stt_emit {
    ($self:expr, |$transcript:ident| {
        Ok  => $ok:expr,
        Err($err:ident) => $errexpr:expr $(,)?
    }) => {
        $self.emit(|result| match result {
            Ok($transcript) => $ok,
            Err($err) => $errexpr,
        })
    };
}

/// Macro for the collect_with method on STT transcription builders.
///
/// This macro allows for the simplified `Ok => ..., Err(e) => ...` syntax
/// in the collect_with method of STT transcription builders.
#[macro_export]
macro_rules! stt_collect_with {
    ($self:expr, |$transcript:ident| {
        Ok  => $ok:expr,
        Err($err:ident) => $errexpr:expr $(,)?
    }) => {
        $self.collect_with(|result| match result {
            Ok($transcript) => $ok,
            Err($err) => $errexpr,
        })
    };
}

/// Helper macro to create a speaker.
///
/// This macro simplifies the creation of speaker objects with default values.
#[macro_export]
macro_rules! speaker {
    ($id:expr) => {{
        use $crate::builders::SpeakerLineBuilder;
        use $crate::speaker_builder::SpeakerBuilder;

        SpeakerLineBuilder::new($id.into())
    }};
}

/// Simplified syntax for TTS engine creation.
///
/// This macro provides a clean way to create and configure a TTS engine.
#[macro_export]
macro_rules! tts {
    () => {{
        use $crate::fluent_voice::FluentVoice;
        use $crate::tts_engine::TtsEngine;

        FluentVoice::tts()
    }};
}

/// Simplified syntax for STT engine creation.
///
/// This macro provides a clean way to create and configure an STT engine.
#[macro_export]
macro_rules! stt {
    () => {{
        use $crate::fluent_voice::FluentVoice;
        use $crate::stt_engine::SttEngine;

        FluentVoice::stt()
    }};
}

/// Macro for synthesize method JSON/arrow syntax support
///
/// This macro transforms the JSON syntax used in TTS examples into proper closures:
/// ```ignore
/// synthesize!(Ok => conversation.into_stream(), Err => e)
/// ```
#[macro_export]
macro_rules! synthesize {
    (Ok => $ok:expr, Err => $err:expr) => {
        move |__res| match __res {
            Ok(chunk) => Ok($ok),
            Err(err) => Err($err),
        }
    };
}

/// Macro for listen method JSON/arrow syntax support
///
/// This macro transforms the JSON syntax used in STT examples into proper closures:
/// ```ignore
/// listen!(Ok => conversation.into_stream(), Err => e)
/// ```
#[macro_export]
macro_rules! listen {
    (Ok => $ok:expr, Err => $err:expr) => {
        move |__res| match __res {
            Ok(chunk) => Ok($ok),
            Err(err) => Err($err),
        }
    };
}
