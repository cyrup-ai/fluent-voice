//! Concrete sound effects builder implementation.

use core::future::Future;
use fluent_voice_domain::audio_format::AudioFormat;
use fluent_voice_domain::VoiceError;
use futures_core::Stream;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

/// Builder trait for sound effects functionality.
pub trait SoundEffectsBuilder: Sized + Send {
    /// The session type produced by this builder.
    type Session: SoundEffectsSession;

    /// Set the audio data to process.
    fn with_audio_data(self, data: Vec<u8>) -> Self;

    /// Add reverb effect.
    fn add_reverb(self, room_size: f32, damping: f32) -> Self;

    /// Add echo effect.
    fn add_echo(self, delay_ms: u32, decay: f32) -> Self;

    /// Add pitch shift effect.
    fn pitch_shift(self, semitones: f32) -> Self;

    /// Add low pass filter.
    fn low_pass(self, frequency: u32) -> Self;

    /// Add high pass filter.
    fn high_pass(self, frequency: u32) -> Self;

    /// Add chorus effect.
    fn add_chorus(self, speed: f32, depth: f32) -> Self;

    /// Set output audio format.
    fn output_format(self, format: AudioFormat) -> Self;

    /// Process the audio effects with a matcher closure.
    fn process<F, R>(self, matcher: F) -> impl Future<Output = R> + Send
    where
        F: FnOnce(Result<Self::Session, VoiceError>) -> R + Send + 'static;
}

/// Session trait for sound effects operations.
pub trait SoundEffectsSession: Send {
    /// The audio stream type produced by this session.
    type AudioStream: Stream<Item = fluent_voice_domain::AudioChunk> + Send + Unpin;

    /// Convert this session into an audio stream.
    fn into_stream(self) -> Self::AudioStream;
}

/// Concrete sound effects session implementation.
pub struct SoundEffectsSessionImpl {
    input_audio: Vec<u8>,
    effect_type: EffectType,
}

#[derive(Debug, Clone)]
enum EffectType {
    Reverb { room_size: f32, damping: f32 },
    Echo { delay_ms: u32, decay: f32 },
    PitchShift { semitones: f32 },
    LowPass { frequency: u32 },
    HighPass { frequency: u32 },
    Chorus { speed: f32, depth: f32 },
}

impl SoundEffectsSession for SoundEffectsSessionImpl {
    type AudioStream = crate::audio_stream::AudioStream;

    fn into_stream(self) -> Self::AudioStream {
        let (tx, rx) = mpsc::unbounded_channel::<fluent_voice_domain::AudioChunk>();

        tokio::spawn(async move {
            match self.apply_ffmpeg_effects().await {
                Ok(processed_audio) => {
                    let chunk = fluent_voice_domain::AudioChunk::with_metadata(
                        processed_audio,
                        0,
                        0,
                        Some("processed_audio".to_string()),
                        Some(format!("Applied effect: {:?}", self.effect_type)),
                        Some(fluent_voice_domain::AudioFormat::Pcm24Khz),
                    );
                    let _ = tx.send(chunk);
                }
                Err(e) => {
                    let error_chunk = fluent_voice_domain::AudioChunk::with_metadata(
                        Vec::new(),
                        0,
                        0,
                        None,
                        Some(format!("[EFFECTS_ERROR] {}", e)),
                        None,
                    );
                    let _ = tx.send(error_chunk);
                }
            }
        });

        let stream = UnboundedReceiverStream::new(rx);
        crate::audio_stream::AudioStream::new(Box::pin(stream))
    }
}

impl SoundEffectsSessionImpl {
    async fn apply_ffmpeg_effects(&self) -> Result<Vec<u8>, VoiceError> {
        let temp_input = std::env::temp_dir().join("input_effects.wav");
        let temp_output = std::env::temp_dir().join("output_effects.wav");

        tokio::fs::write(&temp_input, &self.input_audio)
            .await
            .map_err(|e| VoiceError::AudioProcessing(format!("Failed to write input: {}", e)))?;

        let filter_spec = match &self.effect_type {
            EffectType::Reverb { room_size, damping } => {
                format!(
                    "areverb=roomsize={}:damping={}:stereospread=1",
                    room_size, damping
                )
            }
            EffectType::Echo { delay_ms, decay } => {
                format!("aecho=0.8:0.9:{}:{}", delay_ms, decay)
            }
            EffectType::PitchShift { semitones } => {
                let rate_factor = 2.0_f32.powf(*semitones / 12.0);
                format!("asetrate=r=44100*{},aresample=44100", rate_factor)
            }
            EffectType::LowPass { frequency } => format!("lowpass=frequency={}", frequency),
            EffectType::HighPass { frequency } => format!("highpass=frequency={}", frequency),
            EffectType::Chorus { speed, depth } => {
                format!("chorus=0.7:0.9:55:0.4:{}:{}", speed, depth)
            }
        };

        let mut cmd = tokio::process::Command::new("ffmpeg");
        cmd.arg("-i")
            .arg(&temp_input)
            .arg("-af")
            .arg(&filter_spec)
            .arg("-y")
            .arg(&temp_output);

        let output = cmd
            .output()
            .await
            .map_err(|e| VoiceError::AudioProcessing(format!("FFmpeg execution failed: {}", e)))?;

        if !output.status.success() {
            return Err(VoiceError::AudioProcessing(format!(
                "FFmpeg processing failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        let result = tokio::fs::read(&temp_output)
            .await
            .map_err(|e| VoiceError::AudioProcessing(format!("Failed to read output: {}", e)))?;

        tokio::fs::remove_file(&temp_input).await.ok();
        tokio::fs::remove_file(&temp_output).await.ok();

        Ok(result)
    }
}

/// Concrete sound effects builder implementation.
pub struct SoundEffectsBuilderImpl {
    audio_data: Option<Vec<u8>>,
    effects: Vec<EffectType>,
    output_format: Option<AudioFormat>,
}

impl SoundEffectsBuilderImpl {
    /// Create a new sound effects builder.
    pub fn new() -> Self {
        Self {
            audio_data: None,
            effects: Vec::new(),
            output_format: None,
        }
    }
}

impl Default for SoundEffectsBuilderImpl {
    fn default() -> Self {
        Self::new()
    }
}

impl SoundEffectsBuilder for SoundEffectsBuilderImpl {
    type Session = SoundEffectsSessionImpl;

    fn with_audio_data(mut self, data: Vec<u8>) -> Self {
        self.audio_data = Some(data);
        self
    }

    fn add_reverb(mut self, room_size: f32, damping: f32) -> Self {
        self.effects.push(EffectType::Reverb { room_size, damping });
        self
    }

    fn add_echo(mut self, delay_ms: u32, decay: f32) -> Self {
        self.effects.push(EffectType::Echo { delay_ms, decay });
        self
    }

    fn pitch_shift(mut self, semitones: f32) -> Self {
        self.effects.push(EffectType::PitchShift { semitones });
        self
    }

    fn low_pass(mut self, frequency: u32) -> Self {
        self.effects.push(EffectType::LowPass { frequency });
        self
    }

    fn high_pass(mut self, frequency: u32) -> Self {
        self.effects.push(EffectType::HighPass { frequency });
        self
    }

    fn add_chorus(mut self, speed: f32, depth: f32) -> Self {
        self.effects.push(EffectType::Chorus { speed, depth });
        self
    }

    fn output_format(mut self, format: AudioFormat) -> Self {
        self.output_format = Some(format);
        self
    }

    fn process<F, R>(self, matcher: F) -> impl Future<Output = R> + Send
    where
        F: FnOnce(Result<Self::Session, VoiceError>) -> R + Send + 'static,
    {
        async move {
            match self.validate_and_prepare() {
                Ok((audio_data, effect)) => {
                    let session = SoundEffectsSessionImpl {
                        input_audio: audio_data,
                        effect_type: effect,
                    };
                    matcher(Ok(session))
                }
                Err(e) => matcher(Err(e)),
            }
        }
    }
}

impl SoundEffectsBuilderImpl {
    fn validate_and_prepare(&self) -> Result<(Vec<u8>, EffectType), VoiceError> {
        let audio_data = self
            .audio_data
            .clone()
            .ok_or_else(|| VoiceError::Configuration("Missing audio data".to_string()))?;

        let effect = if self.effects.is_empty() {
            return Err(VoiceError::Configuration(
                "No effects specified".to_string(),
            ));
        } else {
            // Use the first effect for now
            self.effects[0].clone()
        };

        Ok((audio_data, effect))
    }
}
