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

        /* ----- Concrete Speaker Implementation ----- */
        #[derive(Clone, Debug)]
        pub struct SpeakerLine {
            pub id: String,
            pub text: String,
            pub voice_id: Option<fluent_voice_domain::voice_id::VoiceId>,
            pub language: Option<fluent_voice_domain::language::Language>,
            pub speed_modifier: Option<fluent_voice_domain::vocal_speed::VocalSpeedMod>,
            pub pitch_range: Option<fluent_voice_domain::pitch_range::PitchRange>,
        }

        impl fluent_voice_domain::speaker::Speaker for SpeakerLine {
            fn id(&self) -> &str { &self.id }
        }

        /* ----- SpeakerBuilder Implementation ----- */
        #[derive(Clone, Debug)]
        pub struct SpeakerLineBuilder {
            id: String,
            text: String,
            voice_id: Option<fluent_voice_domain::voice_id::VoiceId>,
            language: Option<fluent_voice_domain::language::Language>,
            speed_modifier: Option<fluent_voice_domain::vocal_speed::VocalSpeedMod>,
            pitch_range: Option<fluent_voice_domain::pitch_range::PitchRange>,
        }

        impl fluent_voice_domain::speaker_builder::SpeakerBuilder for SpeakerLineBuilder {
            type Output = SpeakerLine;

            fn named(name: impl Into<String>) -> Self {
                SpeakerLineBuilder {
                    id: name.into(),
                    text: String::new(),
                    voice_id: None,
                    language: None,
                    speed_modifier: None,
                    pitch_range: None,
                }
            }

            fn voice_id(mut self, id: fluent_voice_domain::voice_id::VoiceId) -> Self {
                self.voice_id = Some(id);
                self
            }

            fn language(mut self, lang: fluent_voice_domain::language::Language) -> Self {
                self.language = Some(lang);
                self
            }

            fn with_speed_modifier(mut self, m: fluent_voice_domain::vocal_speed::VocalSpeedMod) -> Self {
                self.speed_modifier = Some(m);
                self
            }

            fn with_pitch_range(mut self, range: fluent_voice_domain::pitch_range::PitchRange) -> Self {
                self.pitch_range = Some(range);
                self
            }

            fn speak(mut self, text: impl Into<String>) -> Self {
                self.text = text.into();
                self
            }

            fn build(self) -> Self::Output {
                SpeakerLine {
                    id: self.id,
                    text: self.text,
                    voice_id: self.voice_id,
                    language: self.language,
                    speed_modifier: self.speed_modifier,
                    pitch_range: self.pitch_range,
                }
            }
        }

        /* ----- SpeakerExt Implementation ----- */
        impl fluent_voice_domain::speaker_builder::SpeakerExt for $engine {
            fn speaker(name: impl Into<String>) -> impl fluent_voice_domain::speaker_builder::SpeakerBuilder {
                SpeakerLineBuilder::named(name)
            }
        }

        /* ----- Conversation Implementation ----- */
        pub struct Conv {
            pub lines: Vec<SpeakerLine>,
            pub global_language: Option<fluent_voice_domain::language::Language>,
        }

        impl fluent_voice_domain::tts_conversation::TtsConversation for Conv {
            type AudioStream = $audio_stream_ty;

            fn into_stream(self) -> Self::AudioStream {
                self.synth_inner()
            }
        }

        impl Conv {
            /// Engine-specific synthesis implementation point.
            /// Engines override this method to provide actual TTS functionality.
            pub fn synth_inner(self) -> $audio_stream_ty {
                // Default implementation - engines should override this
                panic!("Engine must implement synth_inner() method")
            }
        }

        /* ----- ConversationBuilder Implementation ----- */
        pub struct ConvBuilder {
            lines: Vec<SpeakerLine>,
            global_language: Option<fluent_voice_domain::language::Language>,
        }

        impl fluent_voice_domain::tts_conversation::TtsConversationBuilder for ConvBuilder {
            type Conversation = Conv;

            fn with_speaker<S: fluent_voice_domain::speaker::Speaker>(mut self, speaker: S) -> Self {
                // For macro-generated engines, we create a SpeakerLine from any Speaker
                let speaker_line = SpeakerLine {
                    id: speaker.id().to_string(),
                    text: String::new(), // Will be set when speak() is called on the builder
                    voice_id: None,
                    language: None,
                    speed_modifier: None,
                    pitch_range: None,
                };
                self.lines.push(speaker_line);
                self
            }

            fn language(mut self, lang: fluent_voice_domain::language::Language) -> Self {
                self.global_language = Some(lang);
                self
            }

            fn synthesize<F, R>(self, matcher: F) -> impl core::future::Future<Output = R> + Send
            where
                F: FnOnce(Result<Self::Conversation, fluent_voice_domain::voice_error::VoiceError>) -> R + Send + 'static,
            {
                async move {
                    let conversation = Conv {
                        lines: self.lines,
                        global_language: self.global_language,
                    };
                    matcher(Ok(conversation))
                }
            }
        }

        /* ----- TtsConversationExt Implementation ----- */
        impl fluent_voice_domain::tts_conversation::TtsConversationExt for $engine {
            fn builder() -> impl fluent_voice_domain::tts_conversation::TtsConversationBuilder {
                ConvBuilder {
                    lines: Vec::new(),
                    global_language: None,
                }
            }
        }

        /* ----- TtsEngine Implementation ----- */
        impl fluent_voice_domain::tts_engine::TtsEngine for $engine {
            type Conv = ConvBuilder;

            fn conversation(&self) -> Self::Conv {
                ConvBuilder {
                    lines: Vec::new(),
                    global_language: None,
                }
            }
        }

        /* ----- FluentVoice Implementation (TTS only) ----- */
        impl fluent_voice_domain::fluent_voice_domain::FluentVoice for $engine {
            fn tts() -> impl fluent_voice_domain::tts_conversation::TtsConversationBuilder {
                ConvBuilder {
                    lines: Vec::new(),
                    global_language: None,
                }
            }

            fn stt() -> impl fluent_voice_domain::stt_conversation::SttConversationBuilder {
                // Return a dummy builder that panics when used
                struct DummySttBuilder;
                struct DummySegment;
                impl fluent_voice_domain::transcript::TranscriptSegment for DummySegment {
                    fn start_ms(&self) -> u32 { 0 }
                    fn end_ms(&self) -> u32 { 0 }
                    fn text(&self) -> &str { "" }
                    fn speaker_id(&self) -> Option<&str> { None }
                }
                struct DummyStream;
                impl futures_core::Stream for DummyStream {
                    type Item = Result<DummySegment, fluent_voice_domain::voice_error::VoiceError>;
                    fn poll_next(self: core::pin::Pin<&mut Self>, _: &mut core::task::Context<'_>) -> core::task::Poll<Option<Self::Item>> {
                        core::task::Poll::Ready(None)
                    }
                }
                impl Unpin for DummyStream {}
                struct DummyTextStream;
                impl futures_core::Stream for DummyTextStream {
                    type Item = String;
                    fn poll_next(self: core::pin::Pin<&mut Self>, _: &mut core::task::Context<'_>) -> core::task::Poll<Option<Self::Item>> {
                        core::task::Poll::Ready(None)
                    }
                }
                impl Unpin for DummyTextStream {}

                impl fluent_voice_domain::stt_conversation::SttConversationBuilder for DummySttBuilder {
                    type Conversation = DummySession;
                    fn with_source(self, _: fluent_voice_domain::speech_source::SpeechSource) -> Self { self }
                    fn vad_mode(self, _: fluent_voice_domain::vad_mode::VadMode) -> Self { self }
                    fn noise_reduction(self, _: fluent_voice_domain::noise_reduction::NoiseReduction) -> Self { self }
                    fn language_hint(self, _: fluent_voice_domain::language::Language) -> Self { self }
                    fn diarization(self, _: fluent_voice_domain::timestamps::Diarization) -> Self { self }
                    fn word_timestamps(self, _: fluent_voice_domain::timestamps::WordTimestamps) -> Self { self }
                    fn timestamps_granularity(self, _: fluent_voice_domain::timestamps::TimestampsGranularity) -> Self { self }
                    fn punctuation(self, _: fluent_voice_domain::timestamps::Punctuation) -> Self { self }
                    fn with_microphone(self, _: impl Into<String>) -> impl fluent_voice_domain::stt_conversation::MicrophoneBuilder { DummyMicBuilder }
                    fn transcribe(self, _: impl Into<String>) -> impl fluent_voice_domain::stt_conversation::TranscriptionBuilder { DummyTransBuilder }
                    fn listen<F, R>(self, _: F) -> impl core::future::Future<Output = R> + Send
                    where F: FnOnce(Result<Self::Conversation, fluent_voice_domain::voice_error::VoiceError>) -> R + Send + 'static,
                    { async move { panic!("TTS-only engine does not support STT") } }
                }
                struct DummySession;
                impl fluent_voice_domain::stt_conversation::SttConversation for DummySession {
                    type Stream = DummyStream;
                    fn into_stream(self) -> Self::Stream { DummyStream }
                }
                struct DummyMicBuilder;
                impl fluent_voice_domain::stt_conversation::MicrophoneBuilder for DummyMicBuilder {
                    type Conversation = DummySession;
                    fn vad_mode(self, _: fluent_voice_domain::vad_mode::VadMode) -> Self { self }
                    fn noise_reduction(self, _: fluent_voice_domain::noise_reduction::NoiseReduction) -> Self { self }
                    fn language_hint(self, _: fluent_voice_domain::language::Language) -> Self { self }
                    fn diarization(self, _: fluent_voice_domain::timestamps::Diarization) -> Self { self }
                    fn word_timestamps(self, _: fluent_voice_domain::timestamps::WordTimestamps) -> Self { self }
                    fn timestamps_granularity(self, _: fluent_voice_domain::timestamps::TimestampsGranularity) -> Self { self }
                    fn punctuation(self, _: fluent_voice_domain::timestamps::Punctuation) -> Self { self }
                    fn listen<F, R>(self, _: F) -> impl core::future::Future<Output = R> + Send
                    where F: FnOnce(Result<Self::Conversation, fluent_voice_domain::voice_error::VoiceError>) -> R + Send + 'static,
                    { async move { panic!("TTS-only engine does not support STT") } }
                }
                struct DummyTransBuilder;
                impl fluent_voice_domain::stt_conversation::TranscriptionBuilder for DummyTransBuilder {
                    type Transcript = ();
                    fn vad_mode(self, _: fluent_voice_domain::vad_mode::VadMode) -> Self { self }
                    fn noise_reduction(self, _: fluent_voice_domain::noise_reduction::NoiseReduction) -> Self { self }
                    fn language_hint(self, _: fluent_voice_domain::language::Language) -> Self { self }
                    fn diarization(self, _: fluent_voice_domain::timestamps::Diarization) -> Self { self }
                    fn word_timestamps(self, _: fluent_voice_domain::timestamps::WordTimestamps) -> Self { self }
                    fn timestamps_granularity(self, _: fluent_voice_domain::timestamps::TimestampsGranularity) -> Self { self }
                    fn punctuation(self, _: fluent_voice_domain::timestamps::Punctuation) -> Self { self }
                    fn with_progress<S: Into<String>>(self, _: S) -> Self { self }
                    fn emit<F, R>(self, _: F) -> impl core::future::Future<Output = R> + Send
                    where F: FnOnce(Result<Self::Transcript, fluent_voice_domain::voice_error::VoiceError>) -> R + Send + 'static,
                    { async move { panic!("TTS-only engine does not support STT") } }
                    fn collect(self) -> impl core::future::Future<Output = Result<Self::Transcript, fluent_voice_domain::voice_error::VoiceError>> + Send
                    { async move { panic!("TTS-only engine does not support STT") } }
                    fn collect_with<F, R>(self, _: F) -> impl core::future::Future<Output = R> + Send
                    where F: FnOnce(Result<Self::Transcript, fluent_voice_domain::voice_error::VoiceError>) -> R + Send + 'static,
                    { async move { panic!("TTS-only engine does not support STT") } }
                    fn as_text(self) -> impl futures_core::Stream<Item = String> + Send
                    { DummyTextStream }
                }
                DummySttBuilder
            }
        }
    };
}

/* ──────────────────────────────── STT macro ───────────────────────────────────────── */

/// Generate a complete STT engine implementation with polymorphic builders.
#[macro_export]
macro_rules! stt_engine {
    (
        engine     = $engine:ident,
        segment    = $seg_ty:ty,
        stream     = $stream_ty:ty,
        transcript = $transcript_ty:ty,
        $(#[$meta:meta])*
    ) => {
        $(#[$meta])*
        pub struct $engine;

        /* ----- Session/Conversation Implementation ----- */
        pub struct Session {
            pub config: SessionConfig,
        }

        #[derive(Debug)]
        pub struct SessionConfig {
            pub source: Option<fluent_voice_domain::speech_source::SpeechSource>,
            pub vad_mode: Option<fluent_voice_domain::vad_mode::VadMode>,
            pub noise_reduction: Option<fluent_voice_domain::noise_reduction::NoiseReduction>,
            pub language_hint: Option<fluent_voice_domain::language::Language>,
            pub diarization: Option<fluent_voice_domain::timestamps::Diarization>,
            pub word_timestamps: Option<fluent_voice_domain::timestamps::WordTimestamps>,
            pub timestamps_granularity: Option<fluent_voice_domain::timestamps::TimestampsGranularity>,
            pub punctuation: Option<fluent_voice_domain::timestamps::Punctuation>,
        }

        impl Default for SessionConfig {
            fn default() -> Self {
                Self {
                    source: None,
                    vad_mode: None,
                    noise_reduction: None,
                    language_hint: None,
                    diarization: None,
                    word_timestamps: None,
                    timestamps_granularity: None,
                    punctuation: None,
                }
            }
        }

        impl Clone for SessionConfig {
            fn clone(&self) -> Self {
                Self {
                    source: None, // SpeechSource doesn't implement Clone, so we reset this
                    vad_mode: self.vad_mode,
                    noise_reduction: self.noise_reduction,
                    language_hint: self.language_hint.clone(),
                    diarization: self.diarization,
                    word_timestamps: self.word_timestamps,
                    timestamps_granularity: self.timestamps_granularity,
                    punctuation: self.punctuation,
                }
            }
        }

        impl fluent_voice_domain::stt_conversation::SttConversation for Session {
            type Stream = $stream_ty;

            fn into_stream(self) -> Self::Stream {
                self.stream_inner()
            }
        }

        impl Session {
            /// Engine-specific stream implementation point.
            /// Engines override this method to provide actual STT functionality.
            pub fn stream_inner(self) -> $stream_ty {
                panic!("Engine must implement stream_inner() method")
            }
        }

        /* ----- Base STT Builder ----- */
        pub struct SttBuilder {
            config: SessionConfig,
        }

        impl fluent_voice_domain::stt_conversation::SttConversationBuilder for SttBuilder {
            type Conversation = Session;

            fn with_source(mut self, src: fluent_voice_domain::speech_source::SpeechSource) -> Self {
                self.config.source = Some(src);
                self
            }

            fn vad_mode(mut self, mode: fluent_voice_domain::vad_mode::VadMode) -> Self {
                self.config.vad_mode = Some(mode);
                self
            }

            fn noise_reduction(mut self, level: fluent_voice_domain::noise_reduction::NoiseReduction) -> Self {
                self.config.noise_reduction = Some(level);
                self
            }

            fn language_hint(mut self, lang: fluent_voice_domain::language::Language) -> Self {
                self.config.language_hint = Some(lang);
                self
            }

            fn diarization(mut self, d: fluent_voice_domain::timestamps::Diarization) -> Self {
                self.config.diarization = Some(d);
                self
            }

            fn word_timestamps(mut self, w: fluent_voice_domain::timestamps::WordTimestamps) -> Self {
                self.config.word_timestamps = Some(w);
                self
            }

            fn timestamps_granularity(mut self, g: fluent_voice_domain::timestamps::TimestampsGranularity) -> Self {
                self.config.timestamps_granularity = Some(g);
                self
            }

            fn punctuation(mut self, p: fluent_voice_domain::timestamps::Punctuation) -> Self {
                self.config.punctuation = Some(p);
                self
            }

            fn with_microphone(self, device: impl Into<String>) -> impl fluent_voice_domain::stt_conversation::MicrophoneBuilder {
                MicBuilder {
                    config: self.config,
                    device: device.into(),
                }
            }

            fn transcribe(self, path: impl Into<String>) -> impl fluent_voice_domain::stt_conversation::TranscriptionBuilder {
                TransBuilder {
                    config: self.config,
                    path: path.into(),
                    progress_template: None,
                }
            }

            fn listen<F, R>(self, matcher: F) -> impl core::future::Future<Output = R> + Send
            where
                F: FnOnce(Result<Self::Conversation, fluent_voice_domain::voice_error::VoiceError>) -> R + Send + 'static,
            {
                async move {
                    let session = Session { config: self.config };
                    matcher(Ok(session))
                }
            }
        }

        /* ----- Microphone Builder ----- */
        pub struct MicBuilder {
            config: SessionConfig,
            device: String,
        }

        impl fluent_voice_domain::stt_conversation::MicrophoneBuilder for MicBuilder {
            type Conversation = Session;

            fn vad_mode(mut self, mode: fluent_voice_domain::vad_mode::VadMode) -> Self {
                self.config.vad_mode = Some(mode);
                self
            }

            fn noise_reduction(mut self, level: fluent_voice_domain::noise_reduction::NoiseReduction) -> Self {
                self.config.noise_reduction = Some(level);
                self
            }

            fn language_hint(mut self, lang: fluent_voice_domain::language::Language) -> Self {
                self.config.language_hint = Some(lang);
                self
            }

            fn diarization(mut self, d: fluent_voice_domain::timestamps::Diarization) -> Self {
                self.config.diarization = Some(d);
                self
            }

            fn word_timestamps(mut self, w: fluent_voice_domain::timestamps::WordTimestamps) -> Self {
                self.config.word_timestamps = Some(w);
                self
            }

            fn timestamps_granularity(mut self, g: fluent_voice_domain::timestamps::TimestampsGranularity) -> Self {
                self.config.timestamps_granularity = Some(g);
                self
            }

            fn punctuation(mut self, p: fluent_voice_domain::timestamps::Punctuation) -> Self {
                self.config.punctuation = Some(p);
                self
            }

            fn listen<F, R>(self, matcher: F) -> impl core::future::Future<Output = R> + Send
            where
                F: FnOnce(Result<Self::Conversation, fluent_voice_domain::voice_error::VoiceError>) -> R + Send + 'static,
            {
                async move {
                    let mut config = self.config;
                    // Note: Microphone source would be set here in a real implementation
                    // For now, engines override this behavior
                    let session = Session { config };
                    matcher(Ok(session))
                }
            }
        }

        /* ----- Transcription Builder ----- */
        pub struct TransBuilder {
            config: SessionConfig,
            path: String,
            progress_template: Option<String>,
        }

        impl fluent_voice_domain::stt_conversation::TranscriptionBuilder for TransBuilder {
            type Transcript = $transcript_ty;

            fn vad_mode(mut self, mode: fluent_voice_domain::vad_mode::VadMode) -> Self {
                self.config.vad_mode = Some(mode);
                self
            }

            fn noise_reduction(mut self, level: fluent_voice_domain::noise_reduction::NoiseReduction) -> Self {
                self.config.noise_reduction = Some(level);
                self
            }

            fn language_hint(mut self, lang: fluent_voice_domain::language::Language) -> Self {
                self.config.language_hint = Some(lang);
                self
            }

            fn diarization(mut self, d: fluent_voice_domain::timestamps::Diarization) -> Self {
                self.config.diarization = Some(d);
                self
            }

            fn word_timestamps(mut self, w: fluent_voice_domain::timestamps::WordTimestamps) -> Self {
                self.config.word_timestamps = Some(w);
                self
            }

            fn timestamps_granularity(mut self, g: fluent_voice_domain::timestamps::TimestampsGranularity) -> Self {
                self.config.timestamps_granularity = Some(g);
                self
            }

            fn punctuation(mut self, p: fluent_voice_domain::timestamps::Punctuation) -> Self {
                self.config.punctuation = Some(p);
                self
            }

            fn with_progress<S: Into<String>>(mut self, template: S) -> Self {
                self.progress_template = Some(template.into());
                self
            }

            fn emit<F, R>(self, matcher: F) -> impl core::future::Future<Output = R> + Send
            where
                F: FnOnce(Result<Self::Transcript, fluent_voice_domain::voice_error::VoiceError>) -> R + Send + 'static,
            {
                async move {
                    let result = self.transcribe_inner().await;
                    matcher(result)
                }
            }

            fn collect(self) -> impl core::future::Future<Output = Result<Self::Transcript, fluent_voice_domain::voice_error::VoiceError>> + Send {
                async move {
                    self.transcribe_inner().await
                }
            }

            fn collect_with<F, R>(self, handler: F) -> impl core::future::Future<Output = R> + Send
            where
                F: FnOnce(Result<Self::Transcript, fluent_voice_domain::voice_error::VoiceError>) -> R + Send + 'static,
            {
                async move {
                    let result = self.collect().await;
                    handler(result)
                }
            }

            fn as_text(self) -> impl futures_core::Stream<Item = String> + Send {
                self.text_stream_inner()
            }
        }

        impl TransBuilder {
            /// Engine-specific transcription implementation point.
            /// Engines override this method to provide actual transcription functionality.
            pub async fn transcribe_inner(self) -> Result<$transcript_ty, fluent_voice_domain::voice_error::VoiceError> {
                panic!("Engine must implement transcribe_inner() method")
            }

            /// Engine-specific text stream implementation point.
            /// Engines override this method to provide text streaming functionality.
            pub fn text_stream_inner(self) -> impl futures_core::Stream<Item = String> + Send {
                EmptyTextStream
            }
        }

        pub struct EmptyTextStream;

        impl futures_core::Stream for EmptyTextStream {
            type Item = String;

            fn poll_next(self: core::pin::Pin<&mut Self>, _cx: &mut core::task::Context<'_>) -> core::task::Poll<Option<Self::Item>> {
                core::task::Poll::Ready(None)
            }
        }

        /* ----- Engine Trait Implementations ----- */
        impl fluent_voice_domain::stt_conversation::SttConversationExt for $engine {
            fn builder() -> impl fluent_voice_domain::stt_conversation::SttConversationBuilder {
                SttBuilder {
                    config: SessionConfig::default(),
                }
            }
        }

        impl fluent_voice_domain::stt_engine::SttEngine for $engine {
            type Conv = SttBuilder;

            fn conversation(&self) -> Self::Conv {
                SttBuilder {
                    config: SessionConfig::default(),
                }
            }
        }

        /* ----- FluentVoice Implementation (STT only) ----- */
        impl fluent_voice_domain::fluent_voice_domain::FluentVoice for $engine {
            fn tts() -> impl fluent_voice_domain::tts_conversation::TtsConversationBuilder {
                // Return a dummy builder that panics when used
                struct DummyTtsBuilder;
                struct DummyAudioStream;
                impl futures_core::Stream for DummyAudioStream {
                    type Item = i16;
                    fn poll_next(self: core::pin::Pin<&mut Self>, _: &mut core::task::Context<'_>) -> core::task::Poll<Option<Self::Item>> {
                        core::task::Poll::Ready(None)
                    }
                }
                impl Unpin for DummyAudioStream {}

                impl fluent_voice_domain::tts_conversation::TtsConversationBuilder for DummyTtsBuilder {
                    type Conversation = DummyConv;
                    fn with_speaker<S: fluent_voice_domain::speaker::Speaker>(self, _: S) -> Self { self }
                    fn language(self, _: fluent_voice_domain::language::Language) -> Self { self }
                    fn synthesize<F, R>(self, _: F) -> impl core::future::Future<Output = R> + Send
                    where F: FnOnce(Result<Self::Conversation, fluent_voice_domain::voice_error::VoiceError>) -> R + Send + 'static,
                    { async move { panic!("STT-only engine does not support TTS") } }
                }
                struct DummyConv;
                impl fluent_voice_domain::tts_conversation::TtsConversation for DummyConv {
                    type AudioStream = DummyAudioStream;
                    fn into_stream(self) -> Self::AudioStream { DummyAudioStream }
                }
                DummyTtsBuilder
            }

            fn stt() -> impl fluent_voice_domain::stt_conversation::SttConversationBuilder {
                SttBuilder {
                    config: SessionConfig::default(),
                }
            }
        }
    };
}
