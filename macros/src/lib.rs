#![forbid(missing_docs)]

//! Builder-for-builders macro kit for fluent voice engine implementations.

/* ──────────────────────────────── TTS macro ───────────────────────────────────────── */

/// Generate a complete TTS engine implementation with minimal boilerplate.
#[macro_export]
macro_rules! tts_engine {
    (
        engine  = $engine:ident,
        voice   = $voice_ty:ty,
        audio   = $audio_stream_ty:ty,
        $(#[$meta:meta])*
    ) => {
        $(#[$meta])*
        pub struct $engine;

        /* ----- SpeakerBuilder (minimal demo) ----- */
        #[derive(Clone)]
        pub struct SpeakerLine { pub id: Option<String>, pub text: String }
        impl fluent_voice::speaker::Speaker for SpeakerLine {
            fn id(&self) -> &str { self.id.as_deref().unwrap_or("spk") }
        }
        pub struct SpeakerLineBuilder(SpeakerLine);
        impl fluent_voice::speaker_builder::SpeakerBuilder for SpeakerLineBuilder {
            type Output = SpeakerLine;
            fn named(name: impl Into<String>) -> Self {
                SpeakerLineBuilder(SpeakerLine { id: Some(name.into()), text: String::new() })
            }
            fn voice_id(self, _id: fluent_voice::voice_id::VoiceId) -> Self { self }
            fn language(self, _l: fluent_voice::language::Language) -> Self { self }
            fn with_speed_modifier(self, _m: fluent_voice::vocal_speed::VocalSpeedMod) -> Self { self }
            fn with_pitch_range(self, _r: fluent_voice::pitch_range::PitchRange) -> Self { self }
            fn speak(mut self, txt: impl Into<String>) -> Self { self.0.text = txt.into(); self }
            fn build(self) -> Self::Output { self.0 }
        }
        impl fluent_voice::speaker_builder::SpeakerExt for $engine {
            fn speaker(name: impl Into<String>) -> impl fluent_voice::speaker_builder::SpeakerBuilder {
                SpeakerLineBuilder::named(name)
            }
        }

        /* ----- Conversation object + async helper stub ----- */
        pub struct Conv { pub lines: Vec<SpeakerLine> }
        impl fluent_voice::tts_conversation::TtsConversation for Conv {
            type AudioStream = $audio_stream_ty;
            fn into_stream(self) -> Self::AudioStream { self.synth_inner() }
        }
        impl Conv {
            pub fn synth_inner(self) -> $audio_stream_ty {
                todo!("Engine-specific streaming synthesis")
            }
        }

        /* ----- Builder impl ----- */
        pub struct ConvBuilder { lines: Vec<SpeakerLine> }
        impl fluent_voice::tts_conversation::TtsConversationBuilder for ConvBuilder {
            type Conversation = Conv;

            fn with_speaker<S: fluent_voice::speaker::Speaker>(mut self, s: S) -> Self {
                self.lines.push(SpeakerLine { id: Some(s.id().into()), text: String::new() }); self
            }
            fn language(self, _l: fluent_voice::language::Language) -> Self { self }
            fn synthesize<F, R>(self, m: F) -> impl core::future::Future<Output = R> + Send
            where
                F: FnOnce(Result<Self::Conversation, fluent_voice::voice_error::VoiceError>) -> R + Send + 'static,
            {
                async move { m(Ok(Conv { lines: self.lines })) }
            }
        }

        /* ----- Static entry:  MyTtsEngine::builder()  ----- */
        impl fluent_voice::tts_conversation::TtsConversationExt for $engine {
            fn builder() -> impl fluent_voice::tts_conversation::TtsConversationBuilder {
                ConvBuilder { lines: Vec::new() }
            }
        }

        /* ----- Value entry via SttEngine trait stays unchanged ----- */
        impl fluent_voice::tts_engine::TtsEngine for $engine {
            type Conv = ConvBuilder;
            fn conversation(&self) -> Self::Conv { ConvBuilder { lines: Vec::new() } }
        }
    };
}

/* ──────────────────────────────── STT macro ───────────────────────────────────────── */

/// Generate a complete STT engine implementation with minimal boilerplate.
#[macro_export]
macro_rules! stt_engine {
    (
        engine  = $engine:ident,
        segment = $seg_ty:ty,
        stream  = $stream_ty:ty,
        $(#[$meta:meta])*
    ) => {
        $(#[$meta])*
        pub struct $engine;

        /* ----- Session object + helper ----- */
        pub struct Session;
        impl fluent_voice::stt_conversation::SttConversation for Session {
            type Stream = $stream_ty;
            fn into_stream(self) -> Self::Stream { self.transcribe_inner() }
        }
        impl Session {
            pub fn transcribe_inner(&self) -> $stream_ty {
                todo!("Engine-specific transcription stream")
            }
        }

        /* ----- Builder impl ----- */
        pub struct SessBuilder;
        impl fluent_voice::stt_conversation::SttConversationBuilder for SessBuilder {
            type Conversation = Session;

            fn with_source(self, _: fluent_voice::speech_source::SpeechSource) -> Self { self }
            fn vad_mode(self, _: fluent_voice::vad_mode::VadMode) -> Self { self }
            fn noise_reduction(self, _: fluent_voice::noise_reduction::NoiseReduction) -> Self { self }
            fn language_hint(self, _: fluent_voice::language::Language) -> Self { self }
            fn diarization(self, _: fluent_voice::timestamps::Diarization) -> Self { self }
            fn word_timestamps(self, _: fluent_voice::timestamps::WordTimestamps) -> Self { self }
            fn timestamps_granularity(self, _: fluent_voice::timestamps::TimestampsGranularity) -> Self { self }
            fn punctuation(self, _: fluent_voice::timestamps::Punctuation) -> Self { self }
            fn listen<F, R>(self, m: F) -> impl core::future::Future<Output = R> + Send
            where
                F: FnOnce(Result<Self::Conversation, fluent_voice::voice_error::VoiceError>) -> R + Send + 'static,
            {
                async move { m(Ok(Session)) }
            }
        }

        /* ----- Static entry: MySttEngine::builder() ----- */
        impl fluent_voice::stt_conversation::SttConversationExt for $engine {
            fn builder() -> impl fluent_voice::stt_conversation::SttConversationBuilder {
                SessBuilder
            }
        }

        /* ----- Value-based SttEngine impl (unchanged) ----- */
        impl fluent_voice::stt_engine::SttEngine for $engine {
            type Conv = SessBuilder;
            fn conversation(&self) -> Self::Conv { SessBuilder }
        }
    };
}
