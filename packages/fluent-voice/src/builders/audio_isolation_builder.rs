//! Concrete audio isolation builder implementation.

use core::future::Future;
use fluent_voice_domain::audio_format::AudioFormat;
use fluent_voice_domain::VoiceError;
use futures_core::Stream;
use std::path::PathBuf;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

/// Builder trait for audio isolation functionality.
pub trait AudioIsolationBuilder: Sized + Send {
    /// The session type produced by this builder.
    type Session: AudioIsolationSession;

    /// Set the audio file source.
    fn with_file(self, source: impl Into<String>) -> Self;

    /// Set the audio data directly.
    fn with_audio_data(self, data: Vec<u8>) -> Self;

    /// Configure voice isolation.
    fn isolate_voices(self, isolate: bool) -> Self;

    /// Configure background removal.
    fn remove_background(self, remove: bool) -> Self;

    /// Configure noise reduction.
    fn reduce_noise(self, reduce: bool) -> Self;

    /// Set isolation strength.
    fn isolation_strength(self, strength: f32) -> Self;

    /// Set output audio format.
    fn output_format(self, format: AudioFormat) -> Self;

    /// Process the audio with a matcher closure.
    fn process<F, R>(self, matcher: F) -> impl Future<Output = R> + Send
    where
        F: FnOnce(Result<Self::Session, VoiceError>) -> R + Send + 'static;
}

/// Session trait for audio isolation operations.
pub trait AudioIsolationSession: Send {
    /// The audio stream type produced by this session.
    type AudioStream: Stream<Item = fluent_voice_domain::AudioChunk> + Send + Unpin;

    /// Convert this session into an audio stream.
    fn into_stream(self) -> Self::AudioStream;
}

/// Concrete audio isolation session implementation.
pub struct AudioIsolationSessionImpl {
    source_path: PathBuf,
    #[allow(dead_code)]
    isolation_config: IsolationConfig,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct IsolationConfig {
    isolate_vocals: bool,
    remove_background: bool,
    reduce_noise: bool,
    isolation_strength: f32,
}

impl AudioIsolationSession for AudioIsolationSessionImpl {
    type AudioStream = crate::audio_stream::AudioStream;

    fn into_stream(self) -> Self::AudioStream {
        let (tx, rx) = mpsc::unbounded_channel::<fluent_voice_domain::AudioChunk>();

        tokio::spawn(async move {
            match self.run_demucs_isolation().await {
                Ok(isolated_audio_bytes) => {
                    let chunk_size = 4096;
                    for (i, chunk) in isolated_audio_bytes.chunks(chunk_size).enumerate() {
                        let audio_chunk = fluent_voice_domain::AudioChunk::with_metadata(
                            chunk.to_vec(),
                            (chunk.len() / 2) as u64,
                            (i * chunk_size / 2) as u64,
                            Some(format!("isolated_chunk_{}", i)),
                            Some("Isolated vocals".to_string()),
                            Some(fluent_voice_domain::AudioFormat::Pcm24Khz),
                        );

                        if tx.send(audio_chunk).is_err() {
                            break;
                        }
                    }
                }
                Err(e) => {
                    let error_chunk = fluent_voice_domain::AudioChunk::with_metadata(
                        Vec::new(),
                        0,
                        0,
                        None,
                        Some(format!("[ISOLATION_ERROR] {}", e)),
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

impl AudioIsolationSessionImpl {
    async fn run_demucs_isolation(&self) -> Result<Vec<u8>, VoiceError> {
        let cache_dir = std::env::temp_dir().join("fluent_voice_isolation");
        tokio::fs::create_dir_all(&cache_dir).await.map_err(|e| {
            VoiceError::AudioProcessing(format!("Cache dir creation failed: {}", e))
        })?;

        let output_dir = cache_dir.join("separated");
        let mut cmd = tokio::process::Command::new("python");
        cmd.arg("-m")
            .arg("demucs.separate")
            .arg("--model")
            .arg("htdemucs_ft")
            .arg("--device")
            .arg("cuda")
            .arg("--two-stems")
            .arg("vocals")
            .arg("--out")
            .arg(&output_dir)
            .arg(&self.source_path);

        let output = cmd
            .output()
            .await
            .map_err(|e| VoiceError::AudioProcessing(format!("Demucs execution failed: {}", e)))?;

        if !output.status.success() {
            return Err(VoiceError::AudioProcessing(format!(
                "Demucs separation failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        let vocals_file =
            output_dir
                .join("htdemucs_ft")
                .join(self.source_path.file_stem().ok_or_else(|| {
                    VoiceError::AudioProcessing("Invalid source path".to_string())
                })?)
                .join("vocals.wav");

        tokio::fs::read(&vocals_file)
            .await
            .map_err(|e| VoiceError::AudioProcessing(format!("Failed to read vocals: {}", e)))
    }
}

/// Concrete audio isolation builder implementation.
pub struct AudioIsolationBuilderImpl {
    source: Option<String>,
    audio_data: Option<Vec<u8>>,
    isolate_voices: bool,
    remove_background: bool,
    reduce_noise: bool,
    isolation_strength: Option<f32>,
    output_format: Option<AudioFormat>,
}

impl AudioIsolationBuilderImpl {
    /// Create a new audio isolation builder.
    pub fn new() -> Self {
        Self {
            source: None,
            audio_data: None,
            isolate_voices: false,
            remove_background: false,
            reduce_noise: false,
            isolation_strength: None,
            output_format: None,
        }
    }
}

impl Default for AudioIsolationBuilderImpl {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioIsolationBuilder for AudioIsolationBuilderImpl {
    type Session = AudioIsolationSessionImpl;

    fn with_file(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    fn with_audio_data(mut self, data: Vec<u8>) -> Self {
        self.audio_data = Some(data);
        self
    }

    fn isolate_voices(mut self, isolate: bool) -> Self {
        self.isolate_voices = isolate;
        self
    }

    fn remove_background(mut self, remove: bool) -> Self {
        self.remove_background = remove;
        self
    }

    fn reduce_noise(mut self, reduce: bool) -> Self {
        self.reduce_noise = reduce;
        self
    }

    fn isolation_strength(mut self, strength: f32) -> Self {
        self.isolation_strength = Some(strength);
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
            match self.validate_and_prepare().await {
                Ok((source_path, config)) => {
                    let session = AudioIsolationSessionImpl {
                        source_path,
                        isolation_config: config,
                    };
                    matcher(Ok(session))
                }
                Err(e) => matcher(Err(e)),
            }
        }
    }
}

impl AudioIsolationBuilderImpl {
    async fn validate_and_prepare(&self) -> Result<(PathBuf, IsolationConfig), VoiceError> {
        let source_path = if let Some(ref source) = self.source {
            PathBuf::from(source)
        } else {
            return Err(VoiceError::Configuration(
                "Missing audio source".to_string(),
            ));
        };

        if !source_path.exists() {
            return Err(VoiceError::Configuration(format!(
                "Source file does not exist: {}",
                source_path.display()
            )));
        }

        let config = IsolationConfig {
            isolate_vocals: self.isolate_voices,
            remove_background: self.remove_background,
            reduce_noise: self.reduce_noise,
            isolation_strength: self.isolation_strength.unwrap_or(0.8),
        };

        Ok((source_path, config))
    }
}
