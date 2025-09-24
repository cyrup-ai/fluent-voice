//! Concrete speech-to-speech builder implementation.

use core::future::Future;
use fluent_voice_domain::VoiceError;
use fluent_voice_domain::{audio_format::AudioFormat, model_id::ModelId, voice_id::VoiceId};

use futures_core::Stream;
use std::path::PathBuf;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

/// Builder trait for speech-to-speech functionality.
pub trait SpeechToSpeechBuilder: Sized + Send {
    /// The session type produced by this builder.
    type Session: SpeechToSpeechSession;

    /// Set the audio source file or URL.
    fn with_audio_source(self, source: impl Into<String>) -> Self;

    /// Set the audio data directly.
    fn with_audio_data(self, data: Vec<u8>) -> Self;

    /// Set the target voice ID.
    fn target_voice(self, voice_id: VoiceId) -> Self;

    /// Set the model ID.
    fn model(self, model: ModelId) -> Self;

    /// Configure emotion preservation.
    fn preserve_emotion(self, preserve: bool) -> Self;

    /// Configure style preservation.
    fn preserve_style(self, preserve: bool) -> Self;

    /// Configure timing preservation.
    fn preserve_timing(self, preserve: bool) -> Self;

    /// Set output audio format.
    fn output_format(self, format: AudioFormat) -> Self;

    /// Set voice stability parameter.
    fn stability(self, stability: f32) -> Self;

    /// Set similarity boost parameter.
    fn similarity_boost(self, similarity: f32) -> Self;

    /// Set the voice reference file path for voice cloning.
    fn with_voice_reference_path<P: Into<PathBuf>>(self, path: P) -> Self;

    /// Convert the speech with a matcher closure.
    fn convert<F, R>(self, matcher: F) -> impl Future<Output = R> + Send
    where
        F: FnOnce(Result<Self::Session, VoiceError>) -> R + Send + 'static;
}

/// Session trait for speech-to-speech operations.
pub trait SpeechToSpeechSession: Send {
    /// The audio stream type produced by this session.
    type AudioStream: Stream<Item = fluent_voice_domain::AudioChunk> + Send + Unpin;

    /// Convert this session into an audio stream.
    fn into_stream(self) -> Self::AudioStream;
}

/// Concrete speech-to-speech session implementation.
pub struct SpeechToSpeechSessionImpl {
    source_audio: Vec<u8>,
    target_voice_id: VoiceId,
    voice_reference_path: Option<PathBuf>,
    #[allow(dead_code)]
    conversion_config: ConversionConfig,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ConversionConfig {
    preserve_emotion: bool,
    preserve_style: bool,
    preserve_timing: bool,
    stability: f32,
    similarity_boost: f32,
    model: Option<ModelId>,
}

impl SpeechToSpeechSession for SpeechToSpeechSessionImpl {
    type AudioStream = crate::audio_stream::AudioStream;

    fn into_stream(self) -> Self::AudioStream {
        let (tx, rx) = mpsc::unbounded_channel::<fluent_voice_domain::AudioChunk>();

        tokio::spawn(async move {
            match self.perform_voice_conversion().await {
                Ok(converted_bytes) => {
                    let chunk = fluent_voice_domain::AudioChunk::with_metadata(
                        converted_bytes,
                        0,
                        0,
                        Some(format!("voice_conversion_{}", self.target_voice_id)),
                        Some("Speech-to-speech conversion".to_string()),
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
                        Some(format!("[VOICE_CONVERSION_ERROR] {}", e)),
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

impl SpeechToSpeechSessionImpl {
    async fn perform_voice_conversion(&self) -> Result<Vec<u8>, VoiceError> {
        let transcription = self.transcribe_with_whisper().await?;
        let voice_reference = self.get_voice_reference().await?;
        let converted_audio = self
            .synthesize_with_tortoise(&transcription, &voice_reference)
            .await?;
        Ok(converted_audio)
    }

    async fn transcribe_with_whisper(&self) -> Result<String, VoiceError> {
        use fluent_voice_domain::SpeechSource;
        use fluent_voice_whisper::WhisperSttBuilder;

        // Use native Rust Whisper implementation instead of Python subprocess
        let speech_source = SpeechSource::Memory {
            data: self.source_audio.clone(),
            format: fluent_voice_domain::AudioFormat::Pcm16Khz,
            sample_rate: 16000,
        };

        // Use WhisperSttBuilder for native transcription
        let conversation = WhisperSttBuilder::new()
            .with_source(speech_source)
            .transcribe(|conversation_result| conversation_result)
            .await
            .map_err(|e| {
                VoiceError::AudioProcessing(format!("Native Whisper transcription failed: {}", e))
            })?;

        let transcription = conversation.collect().await.map_err(|e| {
            VoiceError::AudioProcessing(format!("Whisper text collection failed: {}", e))
        })?;

        Ok(transcription)
    }

    async fn get_voice_reference(&self) -> Result<Vec<u8>, VoiceError> {
        let voice_ref_path = self.voice_reference_path.as_ref().ok_or_else(|| {
            VoiceError::Configuration("Voice reference path not configured".to_string())
        })?;

        if !voice_ref_path.exists() {
            return Err(VoiceError::Configuration(format!(
                "Voice reference file does not exist: {}",
                voice_ref_path.display()
            )));
        }

        tokio::fs::read(voice_ref_path).await.map_err(|e| {
            VoiceError::AudioProcessing(format!("Failed to load voice reference: {}", e))
        })
    }

    async fn synthesize_with_tortoise(
        &self,
        text: &str,
        voice_ref: &[u8],
    ) -> Result<Vec<u8>, VoiceError> {
        let voice_ref_path = std::env::temp_dir().join("voice_reference.wav");
        let output_path = std::env::temp_dir().join("converted_speech.wav");

        tokio::fs::write(&voice_ref_path, voice_ref)
            .await
            .map_err(|e| {
                VoiceError::AudioProcessing(format!("Failed to write voice reference: {}", e))
            })?;

        let python_script = format!(
            r#"
import tortoise.api as tts
from tortoise.utils.audio import load_audio
import torch
import torchaudio

tts_model = tts.TextToSpeech()

voice_sample = load_audio("{}", 22050)

generated = tts_model.tts_with_preset(
    text="{}",
    voice_samples=[voice_sample],
    preset="fast"
)

torchaudio.save("{}", generated.squeeze(0).cpu(), 22050)
"#,
            voice_ref_path.display(),
            text,
            output_path.display()
        );

        let mut cmd = tokio::process::Command::new("python");
        cmd.arg("-c").arg(&python_script);

        let output = cmd.output().await.map_err(|e| {
            VoiceError::AudioProcessing(format!("TorToiSe synthesis failed: {}", e))
        })?;

        if !output.status.success() {
            return Err(VoiceError::AudioProcessing(format!(
                "TorToiSe failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        let result = tokio::fs::read(&output_path).await.map_err(|e| {
            VoiceError::AudioProcessing(format!("Failed to read converted audio: {}", e))
        })?;

        tokio::fs::remove_file(&voice_ref_path).await.ok();
        tokio::fs::remove_file(&output_path).await.ok();

        Ok(result)
    }
}

/// Concrete speech-to-speech builder implementation.
pub struct SpeechToSpeechBuilderImpl {
    source: Option<String>,
    audio_data: Option<Vec<u8>>,
    target_voice: Option<VoiceId>,
    model: Option<ModelId>,
    preserve_emotion: bool,
    preserve_style: bool,
    preserve_timing: bool,
    output_format: Option<AudioFormat>,
    stability: Option<f32>,
    similarity_boost: Option<f32>,
    voice_reference_path: Option<PathBuf>,
}

impl SpeechToSpeechBuilderImpl {
    /// Create a new speech-to-speech builder.
    pub fn new() -> Self {
        Self {
            source: None,
            audio_data: None,
            target_voice: None,
            model: None,
            preserve_emotion: true,
            preserve_style: true,
            preserve_timing: true,
            output_format: None,
            stability: None,
            similarity_boost: None,
            voice_reference_path: None,
        }
    }

    /// Set the voice reference file path for voice cloning.
    pub fn with_voice_reference_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.voice_reference_path = Some(path.into());
        self
    }
}

impl Default for SpeechToSpeechBuilderImpl {
    fn default() -> Self {
        Self::new()
    }
}

impl SpeechToSpeechBuilder for SpeechToSpeechBuilderImpl {
    type Session = SpeechToSpeechSessionImpl;

    fn with_audio_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    fn with_audio_data(mut self, data: Vec<u8>) -> Self {
        self.audio_data = Some(data);
        self
    }

    fn target_voice(mut self, voice_id: VoiceId) -> Self {
        self.target_voice = Some(voice_id);
        self
    }

    fn model(mut self, model: ModelId) -> Self {
        self.model = Some(model);
        self
    }

    fn preserve_emotion(mut self, preserve: bool) -> Self {
        self.preserve_emotion = preserve;
        self
    }

    fn preserve_style(mut self, preserve: bool) -> Self {
        self.preserve_style = preserve;
        self
    }

    fn preserve_timing(mut self, preserve: bool) -> Self {
        self.preserve_timing = preserve;
        self
    }

    fn output_format(mut self, format: AudioFormat) -> Self {
        self.output_format = Some(format);
        self
    }

    fn stability(mut self, stability: f32) -> Self {
        self.stability = Some(stability);
        self
    }

    fn similarity_boost(mut self, similarity: f32) -> Self {
        self.similarity_boost = Some(similarity);
        self
    }

    fn with_voice_reference_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.voice_reference_path = Some(path.into());
        self
    }

    fn convert<F, R>(self, matcher: F) -> impl Future<Output = R> + Send
    where
        F: FnOnce(Result<Self::Session, VoiceError>) -> R + Send + 'static,
    {
        async move {
            match self.validate_and_prepare().await {
                Ok((source_audio, target_voice_id, config)) => {
                    let session = SpeechToSpeechSessionImpl {
                        source_audio,
                        target_voice_id,
                        voice_reference_path: self.voice_reference_path.clone(),
                        conversion_config: config,
                    };
                    matcher(Ok(session))
                }
                Err(e) => matcher(Err(e)),
            }
        }
    }
}

impl SpeechToSpeechBuilderImpl {
    async fn validate_and_prepare(
        &self,
    ) -> Result<(Vec<u8>, VoiceId, ConversionConfig), VoiceError> {
        let source_audio = if let Some(ref audio_data) = self.audio_data {
            audio_data.clone()
        } else if let Some(ref source) = self.source {
            let source_path = std::path::PathBuf::from(source);
            if !source_path.exists() {
                return Err(VoiceError::Configuration(format!(
                    "Source audio file does not exist: {}",
                    source_path.display()
                )));
            }

            tokio::fs::read(&source_path).await.map_err(|e| {
                VoiceError::AudioProcessing(format!("Failed to load source audio file: {}", e))
            })?
        } else {
            return Err(VoiceError::Configuration(
                "Missing source audio".to_string(),
            ));
        };

        let target_voice_id = self
            .target_voice
            .clone()
            .ok_or_else(|| VoiceError::Configuration("Missing target voice".to_string()))?;

        let config = ConversionConfig {
            preserve_emotion: self.preserve_emotion,
            preserve_style: self.preserve_style,
            preserve_timing: self.preserve_timing,
            stability: self.stability.unwrap_or(0.75),
            similarity_boost: self.similarity_boost.unwrap_or(0.75),
            model: self.model.clone(),
        };

        Ok((source_audio, target_voice_id, config))
    }
}
