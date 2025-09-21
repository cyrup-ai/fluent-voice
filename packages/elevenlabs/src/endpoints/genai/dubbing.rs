//! # Sophisticated Dubbing API
//!
//! This module provides a type-safe, feature-rich interface to the ElevenLabs dubbing API.
//! The API supports dubbing audio and video files into different languages with advanced
//! configuration options.
//!
//! ## Features
//!
//! - **Type-safe source handling**: Enum-based approach prevents invalid configurations
//! - **Automatic validation**: File existence, format, and parameter validation
//! - **Rich metadata**: Automatic MIME type detection and file analysis
//! - **URL source support**: Publicly accessible URLs and pre-signed URLs supported
//! - **Comprehensive configuration**: All ElevenLabs dubbing parameters supported
//! - **Advanced error handling**: Domain-specific errors with actionable messages
//!
//! ## Usage Examples
//!
//! ### Basic File Dubbing
//! ```rust
//! use speakrs_elevenlabs::endpoints::genai::dubbing::*;
//!
//! let body = DubbingBody::from_file("video.mp4", "es")?
//!     .with_name("Spanish Dub")
//!     .with_quality_settings(true, false);
//!
//! let endpoint = DubAVideoOrAnAudioFile::new(body);
//! let response = client.hit(endpoint).await?;
//! ```
//!
//! ### URL-based Dubbing
//! ```rust
//! // URL must be publicly accessible or use pre-signed URLs from cloud storage
//! let body = DubbingBody::from_url("https://example.com/video.mp4", "fr")?
//!     .with_speaker_count(3)?
//!     .with_time_range(10.0, 60.0)?;
//!
//! let endpoint = DubAVideoOrAnAudioFile::new(body);
//! let response = client.hit(endpoint).await?;
//! ```
//!
//! ### Advanced Configuration
//! ```rust
//! let body = DubbingBody::from_file("podcast.mp3", "de")?
//!     .with_source_language("en")
//!     .with_target_accent("bavarian")
//!     .with_content_filtering(true, false)
//!     .with_advanced_options(true, false);
//!
//! let endpoint = DubAVideoOrAnAudioFile::new(body);
//! let response = client.hit(endpoint).await?;
//! ```

use super::*;
use crate::endpoints::genai::speech_to_text::{Word, WordType};
use crate::error::Error;

use serde::Deserialize;
use std::path::Path;
use std::string::ToString;
use strum::Display;

/// Maximum file size to cache in memory (100MB)
/// Files larger than this will fall back to direct file reading
/// with potential race condition risk but no memory issues
const MAX_CACHE_SIZE: u64 = 100 * 1024 * 1024;

/// Maximum file size for dubbing API (100MB)
/// Matches existing MAX_CACHE_SIZE for consistency and performance
const MAX_DUBBING_FILE_SIZE: u64 = 100 * 1024 * 1024;

/// Sophisticated dubbing source with rich metadata and validation
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DubbingSource {
    /// Local file with automatic MIME detection and validation
    File {
        path: String,
        mime_type: Option<String>,
        /// File size in bytes (populated during validation)
        size_bytes: Option<u64>,
        /// Validated file extension
        extension: Option<String>,
        /// Cached file data to prevent race conditions
        /// None for files larger than MAX_CACHE_SIZE or validation failures
        cached_data: Option<Vec<u8>>,
    },
    /// Remote URL (must be publicly accessible or use pre-signed URLs from cloud storage)
    Url {
        url: String,
        /// Expected content type (for validation)
        expected_content_type: Option<String>,
    },
    /// In-memory bytes with explicit metadata
    Bytes {
        data: Vec<u8>,
        filename: String,
        mime_type: String,
        /// Original source description for debugging
        source_description: Option<String>,
    },
}

/// Comprehensive metadata about a dubbing source
#[derive(Debug, Clone, PartialEq)]
pub struct SourceMetadata {
    pub source_type: SourceType,
    pub path: Option<String>,
    pub size_bytes: Option<u64>,
    pub mime_type: Option<String>,
    pub extension: Option<String>,
    pub is_cached: bool,
}

/// Type of dubbing source
#[derive(Debug, Clone, PartialEq)]
pub enum SourceType {
    File,
    Bytes,
    Url,
}

impl DubbingSource {
    /// Validate the source and populate metadata
    pub fn validate(&mut self) -> Result<()> {
        match self {
            DubbingSource::File {
                path,
                mime_type,
                size_bytes,
                extension,
                cached_data,
            } => {
                let file_path = Path::new(path);

                // ATOMIC OPERATION: Read file and validate in single step
                let file_data = std::fs::read(file_path).map_err(|e| {
                    Box::new(Error::FileValidationFailed {
                        path: path.clone(),
                        reason: format!("Failed to read file: {}", e),
                    })
                })?;

                // Populate size from actual data
                *size_bytes = Some(file_data.len() as u64);

                // Validate file size against API limits
                if file_data.len() as u64 > MAX_DUBBING_FILE_SIZE {
                    return Err(Box::new(Error::FileValidationFailed {
                        path: path.clone(),
                        reason: format!(
                            "File size {} bytes exceeds maximum allowed size {} bytes ({}MB). \
                             Consider compressing the file or using a smaller segment.",
                            file_data.len(),
                            MAX_DUBBING_FILE_SIZE,
                            MAX_DUBBING_FILE_SIZE / (1024 * 1024)
                        ),
                    }));
                }

                // Cache data if under threshold, otherwise None for memory efficiency
                if file_data.len() as u64 <= MAX_CACHE_SIZE {
                    *cached_data = Some(file_data);
                } else {
                    *cached_data = None;
                    // Large files will fall back to direct reading in TryFrom
                    // This accepts race condition risk for memory efficiency
                }

                // Continue with existing validation logic
                *extension = file_path
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .map(|s| s.to_lowercase());

                // Auto-detect MIME type if not provided
                if mime_type.is_none() {
                    *mime_type = Self::detect_mime_type_from_extension(extension.as_deref());
                }

                // Validate supported formats
                Self::validate_media_format(extension.as_deref())?;
            }
            DubbingSource::Url {
                url,
                expected_content_type,
                ..
            } => {
                // Validate URL format
                url::Url::parse(url).map_err(|e| {
                    Box::new(Error::UrlValidationFailed {
                        url: url.clone(),
                        reason: format!("Invalid URL format: {}", e),
                    })
                })?;

                // Validate expected content type if provided
                if let Some(content_type) = expected_content_type {
                    Self::validate_content_type(content_type)?;
                }
            }
            DubbingSource::Bytes {
                data, mime_type, ..
            } => {
                // Validate non-empty data
                if data.is_empty() {
                    return Err(Box::new(Error::InvalidDubbingSource(
                        "Byte data cannot be empty".to_string(),
                    )));
                }

                // Validate MIME type format
                Self::validate_content_type(mime_type)?;
            }
        }
        Ok(())
    }

    /// Detect MIME type from file extension
    fn detect_mime_type_from_extension(extension: Option<&str>) -> Option<String> {
        match extension {
            Some("mp4") => Some("video/mp4".to_string()),
            Some("mp3") => Some("audio/mp3".to_string()),
            Some("wav") => Some("audio/wav".to_string()),
            Some("m4a") => Some("audio/mp4".to_string()),
            Some("aac") => Some("audio/aac".to_string()),
            Some("flac") => Some("audio/flac".to_string()),
            Some("ogg") => Some("audio/ogg".to_string()),
            Some("webm") => Some("video/webm".to_string()),
            Some("avi") => Some("video/avi".to_string()),
            Some("mov") => Some("video/quicktime".to_string()),
            _ => None,
        }
    }

    /// Validate media format is supported by ElevenLabs
    fn validate_media_format(extension: Option<&str>) -> Result<()> {
        const SUPPORTED_AUDIO: &[&str] = &["mp3", "wav", "m4a", "aac", "flac", "ogg"];
        const SUPPORTED_VIDEO: &[&str] = &["mp4", "webm", "avi", "mov"];
        const SUPPORTED_ALL: &str = "mp3, wav, m4a, aac, flac, ogg, mp4, webm, avi, mov";

        match extension {
            Some(ext) if SUPPORTED_AUDIO.contains(&ext) || SUPPORTED_VIDEO.contains(&ext) => Ok(()),
            Some(ext) => Err(Box::new(Error::UnsupportedMediaFormat {
                extension: ext.to_string(),
                supported: SUPPORTED_ALL.to_string(),
            })),
            None => Err(Box::new(Error::InvalidDubbingSource(
                "Could not determine file format".to_string(),
            ))),
        }
    }

    /// Validate content type format
    fn validate_content_type(content_type: &str) -> Result<()> {
        if content_type.contains('/') && !content_type.is_empty() {
            Ok(())
        } else {
            Err(Box::new(Error::InvalidDubbingSource(format!(
                "Invalid content type format: {}",
                content_type
            ))))
        }
    }

    /// Get the file size in bytes for any source type
    /// Returns None for URL sources (size unknown until download)
    pub fn file_size(&self) -> Option<u64> {
        match self {
            DubbingSource::File { size_bytes, .. } => *size_bytes,
            DubbingSource::Bytes { data, .. } => Some(data.len() as u64),
            DubbingSource::Url { .. } => None, // Size unknown for remote URLs
        }
    }

    /// Get comprehensive metadata about the dubbing source
    pub fn metadata(&self) -> SourceMetadata {
        match self {
            DubbingSource::File {
                path,
                mime_type,
                size_bytes,
                extension,
                ..
            } => SourceMetadata {
                source_type: SourceType::File,
                path: Some(path.clone()),
                size_bytes: *size_bytes,
                mime_type: mime_type.clone(),
                extension: extension.clone(),
                is_cached: self.is_cached(),
            },
            DubbingSource::Bytes {
                data,
                filename,
                mime_type,
                source_description: _,
            } => {
                SourceMetadata {
                    source_type: SourceType::Bytes,
                    path: Some(filename.clone()),
                    size_bytes: Some(data.len() as u64),
                    mime_type: Some(mime_type.clone()),
                    extension: Self::extract_extension_from_filename(filename),
                    is_cached: true, // Bytes are always "cached"
                }
            }
            DubbingSource::Url {
                url,
                expected_content_type,
            } => {
                SourceMetadata {
                    source_type: SourceType::Url,
                    path: Some(url.clone()),
                    size_bytes: None, // Unknown until download
                    mime_type: expected_content_type.clone(),
                    extension: Self::extract_extension_from_url(url),
                    is_cached: false,
                }
            }
        }
    }

    /// Check if the source data is cached in memory
    pub fn is_cached(&self) -> bool {
        match self {
            DubbingSource::File { cached_data, .. } => cached_data.is_some(),
            DubbingSource::Bytes { .. } => true, // Always in memory
            DubbingSource::Url { .. } => false,  // Never cached
        }
    }

    /// Get human-readable size description
    pub fn size_description(&self) -> String {
        match self.file_size() {
            Some(bytes) => {
                if bytes < 1024 {
                    format!("{} bytes", bytes)
                } else if bytes < 1024 * 1024 {
                    format!("{:.1} KB", bytes as f64 / 1024.0)
                } else if bytes < 1024 * 1024 * 1024 {
                    format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
                } else {
                    format!("{:.1} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
                }
            }
            None => "Unknown size".to_string(),
        }
    }

    // Helper methods for metadata extraction
    fn extract_extension_from_filename(filename: &str) -> Option<String> {
        std::path::Path::new(filename)
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|s| s.to_lowercase())
    }

    fn extract_extension_from_url(url: &str) -> Option<String> {
        url::Url::parse(url).ok().and_then(|parsed| {
            std::path::Path::new(parsed.path())
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|s| s.to_lowercase())
        })
    }
}

/// Dubs provided audio or video file into given language.
///
/// # Example
///
/// ```no_run
/// use speakrs_elevenlabs::{ElevenLabsClient, Result};
/// use speakrs_elevenlabs::endpoints::genai::dubbing::*;
///
/// #[tokio::main]
/// async fn main() -> Result<()> {
///     let client = ElevenLabsClient::from_env()?;
///
///     // Use type-safe constructors for file or URL sources
///     let body = DubbingBody::from_url("https://example.com/video.mp4", "en")?
///         .with_quality_settings(true, false)
///         .with_speaker_count(2)?
///         .with_source_language("zh");
///
///     let endpoint = DubAVideoOrAnAudioFile::new(body);
///
///     let response = client.hit(endpoint).await?;
///
///     println!("{:?}", response);
///
///     Ok(())
/// }
/// ```
/// See [Dub a Video or Audio File API reference](https://elevenlabs.io/docs/api-reference/dubbing/dub-a-video-or-an-audio-file)
#[derive(Clone, Debug)]
pub struct DubAVideoOrAnAudioFile {
    body: DubbingBody,
}

impl DubAVideoOrAnAudioFile {
    pub fn new(body: DubbingBody) -> Self {
        DubAVideoOrAnAudioFile { body }
    }
}

impl ElevenLabsEndpoint for DubAVideoOrAnAudioFile {
    const PATH: &'static str = "v1/dubbing";

    const METHOD: Method = Method::POST;

    type ResponseBody = DubAVideoOrAnAudioFileResponse;

    async fn request_body(&self) -> Result<RequestBody> {
        TryInto::try_into(&self.body)
    }

    async fn response_body(self, resp: Response) -> Result<Self::ResponseBody> {
        Ok(resp.json().await?)
    }
}

/// Sophisticated dubbing request builder with comprehensive configuration
#[derive(Clone, Debug)]
pub struct DubbingBody {
    /// Required source (file, URL, or bytes)
    source: DubbingSource,
    /// Optional project name for organization
    name: Option<String>,
    /// Required target language for dubbing
    target_lang: String,
    /// Optional source language (auto-detected if not provided)
    source_lang: Option<String>,
    /// Optional target accent for regional variations
    target_accent: Option<String>,
    /// Optional number of speakers (auto-detected if not provided)
    num_speakers: Option<u32>,
    /// Optional watermark inclusion
    watermark: Option<bool>,
    /// Optional start time for partial dubbing (seconds)
    start_time: Option<f32>,
    /// Optional end time for partial dubbing (seconds)
    end_time: Option<f32>,
    /// Optional highest resolution flag for video
    highest_resolution: Option<bool>,
    /// Optional background audio removal
    drop_background_audio: Option<bool>,
    /// [BETA] Whether transcripts should have profanities censored with the words ‘[censored]’
    use_profanity_filter: Option<bool>,
    /// Optional dubbing studio mode
    dubbing_studio: Option<bool>,
    /// Optional voice cloning disable flag
    disable_voice_cloning: Option<bool>,
    /// Optional processing mode
    mode: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct DubAVideoOrAnAudioFileResponse {
    pub dubbing_id: String,
    pub expected_duration_sec: f32,
}

impl DubbingBody {
    /// Create from file path with validation
    pub fn from_file(path: impl Into<String>, target_lang: impl Into<String>) -> Result<Self> {
        let mut source = DubbingSource::File {
            path: path.into(),
            mime_type: None,
            size_bytes: None,
            extension: None,
            cached_data: None, // Will be populated during validation
        };
        source.validate()?;

        Ok(Self {
            source,
            target_lang: target_lang.into(),
            name: None,
            source_lang: None,
            target_accent: None,
            num_speakers: None,
            watermark: None,
            start_time: None,
            end_time: None,
            highest_resolution: None,
            drop_background_audio: None,
            use_profanity_filter: None,
            dubbing_studio: None,
            disable_voice_cloning: None,
            mode: None,
        })
    }

    /// Create from URL (must be publicly accessible or use pre-signed URLs from cloud storage)
    pub fn from_url(url: impl Into<String>, target_lang: impl Into<String>) -> Result<Self> {
        let mut source = DubbingSource::Url {
            url: url.into(),
            expected_content_type: None,
        };
        source.validate()?;

        Ok(Self {
            source,
            target_lang: target_lang.into(),
            name: None,
            source_lang: None,
            target_accent: None,
            num_speakers: None,
            watermark: None,
            start_time: None,
            end_time: None,
            highest_resolution: None,
            drop_background_audio: None,
            use_profanity_filter: None,
            dubbing_studio: None,
            disable_voice_cloning: None,
            mode: None,
        })
    }

    /// Create from in-memory bytes
    pub fn from_bytes(
        data: Vec<u8>,
        filename: impl Into<String>,
        mime_type: impl Into<String>,
        target_lang: impl Into<String>,
    ) -> Result<Self> {
        let mut source = DubbingSource::Bytes {
            data,
            filename: filename.into(),
            mime_type: mime_type.into(),
            source_description: None,
        };
        source.validate()?;

        Ok(Self {
            source,
            target_lang: target_lang.into(),
            name: None,
            source_lang: None,
            target_accent: None,
            num_speakers: None,
            watermark: None,
            start_time: None,
            end_time: None,
            highest_resolution: None,
            drop_background_audio: None,
            use_profanity_filter: None,
            dubbing_studio: None,
            disable_voice_cloning: None,
            mode: None,
        })
    }

    // Fluent builder methods with validation
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn with_source_language(mut self, source_lang: impl Into<String>) -> Self {
        self.source_lang = Some(source_lang.into());
        self
    }

    pub fn with_target_accent(mut self, accent: impl Into<String>) -> Self {
        self.target_accent = Some(accent.into());
        self
    }

    pub fn with_speaker_count(mut self, count: u32) -> Result<Self> {
        if count == 0 || count > 50 {
            return Err(Box::new(Error::InvalidSpeakerCount { count }));
        }
        self.num_speakers = Some(count);
        Ok(self)
    }

    pub fn with_time_range(mut self, start: f32, end: f32) -> Result<Self> {
        if start < 0.0 || end <= start {
            return Err(Box::new(Error::InvalidTimeRange { start, end }));
        }
        self.start_time = Some(start);
        self.end_time = Some(end);
        Ok(self)
    }

    pub fn with_quality_settings(
        mut self,
        highest_resolution: bool,
        drop_background: bool,
    ) -> Self {
        self.highest_resolution = Some(highest_resolution);
        self.drop_background_audio = Some(drop_background);
        self
    }

    pub fn with_content_filtering(mut self, profanity_filter: bool, watermark: bool) -> Self {
        self.use_profanity_filter = Some(profanity_filter);
        self.watermark = Some(watermark);
        self
    }

    pub fn with_advanced_options(mut self, studio_mode: bool, disable_cloning: bool) -> Self {
        self.dubbing_studio = Some(studio_mode);
        self.disable_voice_cloning = Some(disable_cloning);
        self
    }

    /// Get the source file size in bytes
    pub fn source_size(&self) -> Option<u64> {
        self.source.file_size()
    }

    /// Get comprehensive source metadata
    pub fn source_metadata(&self) -> SourceMetadata {
        self.source.metadata()
    }

    /// Get human-readable source size description
    pub fn source_size_description(&self) -> String {
        self.source.size_description()
    }

    /// Check if source data is cached in memory
    pub fn is_source_cached(&self) -> bool {
        self.source.is_cached()
    }
}

/// Returns metadata about a dubbing project, including whether it’s still in progress or not.
///
/// # Example
///
/// ```no_run
///use speakrs_elevenlabs::{ElevenLabsClient, Result};
///use speakrs_elevenlabs::endpoints::genai::dubbing::*;
///
///#[tokio::main]
///async fn main() -> Result<()> {
///    let client = ElevenLabsClient::from_env()?;
///
///    let endpoint = GetDubbing::new("some_id");
///
///    let response = client.hit(endpoint).await?;
///
///    println!("{:?}", response);
///
///    Ok(())
///}
/// ```
/// See [Get Dubbing API reference](https://elevenlabs.io/docs/api-reference/dubbing/get-dubbing-project-metadata)
#[derive(Clone, Debug)]
pub struct GetDubbing {
    dubbing_id: String,
}

impl GetDubbing {
    pub fn new(dubbing_id: impl Into<String>) -> Self {
        GetDubbing {
            dubbing_id: dubbing_id.into(),
        }
    }
}

impl ElevenLabsEndpoint for GetDubbing {
    const PATH: &'static str = "v1/dubbing/:dubbing_id";

    const METHOD: Method = Method::GET;

    type ResponseBody = GetDubbingResponse;

    fn path_params(&self) -> Vec<(&'static str, &str)> {
        vec![self.dubbing_id.and_param(PathParam::DubbingID)]
    }

    async fn response_body(self, resp: Response) -> Result<Self::ResponseBody> {
        Ok(resp.json().await?)
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct GetDubbingResponse {
    pub dubbing_id: String,
    pub name: String,
    pub status: String,
    pub target_languages: Vec<String>,
    pub media_metadata: MediaMetadata,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct MediaMetadata {
    pub content_type: String,
    pub duration: f32,
}

/// Returns dubbed file as a streamed file.
/// Videos will be returned in MP4 format and audio only dubs will be returned in MP3.
///
/// # Example
/// ```no_run
///
/// use speakrs_elevenlabs::{ElevenLabsClient, Result};
/// use speakrs_elevenlabs::endpoints::genai::dubbing::*;
/// use speakrs_elevenlabs::utils::save;
///
/// #[tokio::main]
/// async fn main() -> Result<()> {
///     let client = ElevenLabsClient::from_env()?;
///
///     let endpoint = GetDubbedAudio::new("some_id", "en");
///     let resp = client.hit(endpoint).await?;
///     save("dubbed.mp4", resp)?;
///
///     Ok(())
/// }
/// ```
/// See [Get Dubbed Audio API reference](https://elevenlabs.io/docs/api-reference/dubbing/get-dubbed-file)
#[derive(Clone, Debug)]
pub struct GetDubbedAudio {
    dubbing_id: String,
    language_code_id: String,
}

impl GetDubbedAudio {
    pub fn new(dubbing_id: impl Into<String>, language_code_id: impl Into<String>) -> Self {
        GetDubbedAudio {
            dubbing_id: dubbing_id.into(),
            language_code_id: language_code_id.into(),
        }
    }
}

impl ElevenLabsEndpoint for GetDubbedAudio {
    const PATH: &'static str = "v1/dubbing/:dubbing_id/audio/:language_code";

    const METHOD: Method = Method::GET;

    type ResponseBody = Bytes;

    fn path_params(&self) -> Vec<(&'static str, &str)> {
        vec![
            self.dubbing_id.and_param(PathParam::DubbingID),
            self.language_code_id.and_param(PathParam::LanguageCodeID),
        ]
    }

    async fn response_body(self, resp: Response) -> Result<Self::ResponseBody> {
        Ok(resp.bytes().await?)
    }
}

/// Deletes a dubbing project.
///
/// # Example
///
/// ```no_run
/// use speakrs_elevenlabs::{ElevenLabsClient, Result};
/// use speakrs_elevenlabs::endpoints::genai::dubbing::*;
///
/// #[tokio::main]
/// async fn main() -> Result<()> {
///     let client = ElevenLabsClient::from_env()?;
///
///     let endpoint = DeleteDubbing::new("some_id");
///     let resp = client.hit(endpoint).await?;
///     println!("{:?}", resp);
///
///     Ok(())
/// }
/// ```
/// See [Delete Dubbing API reference](https://elevenlabs.io/docs/api-reference/dubbing/delete-dubbing-project)
#[derive(Clone, Debug)]
pub struct DeleteDubbing {
    dubbing_id: String,
}

impl DeleteDubbing {
    pub fn new(dubbing_id: impl Into<String>) -> Self {
        DeleteDubbing {
            dubbing_id: dubbing_id.into(),
        }
    }
}

impl ElevenLabsEndpoint for DeleteDubbing {
    const PATH: &'static str = "v1/dubbing/:dubbing_id";

    const METHOD: Method = Method::DELETE;

    type ResponseBody = StatusResponseBody;

    fn path_params(&self) -> Vec<(&'static str, &str)> {
        vec![self.dubbing_id.and_param(PathParam::DubbingID)]
    }

    async fn response_body(self, resp: Response) -> Result<Self::ResponseBody> {
        Ok(resp.json().await?)
    }
}

/// Returns transcript for the dub as an SRT file.
///
/// # Example
/// ```no_run
/// use speakrs_elevenlabs::{ElevenLabsClient, Result};
/// use speakrs_elevenlabs::endpoints::genai::dubbing::*;
///
/// #[tokio::main]
/// async fn main() -> Result<()> {
///     let client = ElevenLabsClient::from_env()?;
///
///     let query = GetDubbedTranscriptQuery::default()
///         .with_format(TranscriptFormat::WebVtt);
///
///     let endpoint = GetDubbedTranscript::new("some_id", "en")
///         .with_query(query);
///
///     let resp = client.hit(endpoint).await?;
///     println!("{:?}", resp);
///
///     Ok(())
/// }
/// ```
/// See [Get Dubbed Transcript API reference](https://elevenlabs.io/docs/api-reference/dubbing/get-transcript-for-dub)
#[derive(Clone, Debug)]
pub struct GetDubbedTranscript {
    dubbing_id: String,
    language_code_id: String,
    query: Option<GetDubbedTranscriptQuery>,
}

impl GetDubbedTranscript {
    pub fn new(dubbing_id: impl Into<String>, language_code_id: impl Into<String>) -> Self {
        GetDubbedTranscript {
            dubbing_id: dubbing_id.into(),
            language_code_id: language_code_id.into(),
            query: None,
        }
    }

    pub fn with_query(mut self, query: GetDubbedTranscriptQuery) -> Self {
        self.query = Some(query);
        self
    }
}

#[derive(Clone, Debug, Default)]
pub struct GetDubbedTranscriptQuery {
    params: QueryValues,
}

impl GetDubbedTranscriptQuery {
    pub fn with_format(mut self, format: TranscriptFormat) -> Self {
        self.params.push(("format_type", format.to_string()));
        self
    }

    /// Get the format from query parameters, defaults to SRT
    pub fn get_format(&self) -> TranscriptFormat {
        self.params
            .iter()
            .find(|(key, _)| *key == "format_type")
            .and_then(|(_, value)| match value.as_str() {
                "webvtt" => Some(TranscriptFormat::WebVtt),
                "srt" => Some(TranscriptFormat::Srt),
                _ => None,
            })
            .unwrap_or(TranscriptFormat::Srt)
    }
}

/// Parse SRT format transcript into Word structures (REUSE existing patterns)
fn parse_srt_transcript(
    content: String,
    format: TranscriptFormat,
) -> Result<GetDubbedTranscriptResponse> {
    let items: Vec<srtparse::Item> =
        srtparse::from_str(&content).map_err(|e| Error::TranscriptParseError {
            format: "SRT".to_string(),
            reason: e.to_string(),
        })?;

    let words: Vec<Word> = items
        .into_iter()
        .map(|item| Word {
            text: item.text,
            r#type: WordType::Word, // REUSE existing enum
            start: Some(item.start_time.into_duration().as_secs_f32()),
            end: Some(item.end_time.into_duration().as_secs_f32()),
            speaker_id: None, // SRT doesn't have speaker info
            characters: None, // Not needed for transcript segments
        })
        .collect();

    let total_duration = words.last().and_then(|w| w.end);

    Ok(GetDubbedTranscriptResponse {
        format,
        raw_content: content,
        words,
        total_duration,
    })
}

/// Parse WebVTT format transcript into Word structures (REUSE existing patterns)
fn parse_webvtt_transcript(
    content: String,
    format: TranscriptFormat,
) -> Result<GetDubbedTranscriptResponse> {
    use std::str::FromStr;
    let webvtt = vtt::WebVtt::from_str(&content).map_err(|e| Error::TranscriptParseError {
        format: "WebVTT".to_string(),
        reason: e.to_string(),
    })?;

    let words: Vec<Word> = webvtt
        .cues
        .into_iter()
        .map(|cue| Word {
            text: cue.payload,
            r#type: WordType::Word, // REUSE existing enum
            start: Some(cue.start.as_duration().as_secs_f32()),
            end: Some(cue.end.as_duration().as_secs_f32()),
            speaker_id: cue.identifier, // WebVTT cue ID as speaker
            characters: None,           // Not needed for transcript segments
        })
        .collect();

    let total_duration = words.last().and_then(|w| w.end);

    Ok(GetDubbedTranscriptResponse {
        format,
        raw_content: content,
        words,
        total_duration,
    })
}

impl ElevenLabsEndpoint for GetDubbedTranscript {
    const PATH: &'static str = "v1/dubbing/:dubbing_id/transcript/:language_code";

    const METHOD: Method = Method::GET;

    type ResponseBody = GetDubbedTranscriptResponse; // CHANGED from String

    fn path_params(&self) -> Vec<(&'static str, &str)> {
        vec![
            self.dubbing_id.and_param(PathParam::DubbingID),
            self.language_code_id.and_param(PathParam::LanguageCodeID),
        ]
    }

    fn query_params(&self) -> Option<QueryValues> {
        self.query.as_ref().map(|q| q.params.clone())
    }

    async fn response_body(self, resp: Response) -> Result<Self::ResponseBody> {
        let content = resp.text().await?;
        let format = self
            .query
            .as_ref()
            .map(|q| q.get_format())
            .unwrap_or(TranscriptFormat::Srt);

        match format {
            TranscriptFormat::Srt => parse_srt_transcript(content, format),
            TranscriptFormat::WebVtt => parse_webvtt_transcript(content, format),
        }
    }
}

/// Transcript response that reuses existing Word structure
#[derive(Clone, Debug, Deserialize)]
pub struct GetDubbedTranscriptResponse {
    /// The format of the original transcript
    pub format: TranscriptFormat,
    /// Raw transcript content for debugging/fallback  
    pub raw_content: String,
    /// Parsed transcript segments using existing Word structure
    pub words: Vec<Word>,
    /// Total duration in seconds
    pub total_duration: Option<f32>,
}

// REUSE existing iterator pattern from speech_to_text.rs
impl IntoIterator for GetDubbedTranscriptResponse {
    type Item = Word;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.words.into_iter()
    }
}

impl<'a> IntoIterator for &'a GetDubbedTranscriptResponse {
    type Item = &'a Word;
    type IntoIter = std::slice::Iter<'a, Word>;

    fn into_iter(self) -> Self::IntoIter {
        self.words.iter()
    }
}

impl GetDubbedTranscriptResponse {
    /// Get iterator over words (REUSE existing pattern)
    pub fn words(&self) -> impl Iterator<Item = &Word> {
        self.words.iter()
    }

    /// Find words containing text (USEFUL utility method)
    pub fn find_words_containing(&self, text: &str) -> Vec<&Word> {
        self.words
            .iter()
            .filter(|word| word.text.contains(text))
            .collect()
    }

    /// Get words in time range (USEFUL utility method)
    pub fn words_in_range(&self, start: f32, end: f32) -> Vec<&Word> {
        self.words
            .iter()
            .filter(|word| {
                if let (Some(word_start), Some(word_end)) = (word.start, word.end) {
                    word_start >= start && word_end <= end
                } else {
                    false
                }
            })
            .collect()
    }

    /// Convert back to SRT format (REUSE existing utilities)
    pub fn to_srt(&self) -> String {
        let mut srt = String::new();
        for (i, word) in self.words.iter().enumerate() {
            if let (Some(start), Some(end)) = (word.start, word.end) {
                let start_time = crate::timestamp_export::seconds_to_srt_time(start);
                let end_time = crate::timestamp_export::seconds_to_srt_time(end);
                srt.push_str(&format!(
                    "{}\n{} --> {}\n{}\n\n",
                    i + 1,
                    start_time,
                    end_time,
                    word.text
                ));
            }
        }
        srt
    }

    /// Convert back to WebVTT format (REUSE existing utilities)  
    pub fn to_vtt(&self) -> String {
        let mut vtt = String::from("WEBVTT\n\n");
        for word in &self.words {
            if let (Some(start), Some(end)) = (word.start, word.end) {
                let start_time = crate::timestamp_export::seconds_to_vtt_time(start);
                let end_time = crate::timestamp_export::seconds_to_vtt_time(end);
                vtt.push_str(&format!(
                    "{} --> {}\n{}\n\n",
                    start_time, end_time, word.text
                ));
            }
        }
        vtt
    }
}

#[derive(Clone, Debug, Display, Deserialize)]
#[strum(serialize_all = "lowercase")]
pub enum TranscriptFormat {
    Srt,
    WebVtt,
}

impl TryFrom<&DubbingBody> for RequestBody {
    type Error = Box<dyn std::error::Error + Send + Sync>;

    fn try_from(body: &DubbingBody) -> Result<Self> {
        let mut form = Form::new();

        // Handle source based on enum variant
        match &body.source {
            DubbingSource::File {
                path,
                mime_type,
                cached_data,
                size_bytes,
                ..
            } => {
                let file_bytes = match cached_data {
                    Some(data) => {
                        // Use cached data - eliminates race condition
                        data.clone()
                    }
                    None => {
                        // Fallback for large files - original behavior with race condition risk
                        let file_path = Path::new(path);
                        std::fs::read(file_path).map_err(|e| {
                            format!(
                                "Failed to read large file '{}' (size: {} bytes): {}. \
                                 File may have been deleted after validation.",
                                path,
                                size_bytes.unwrap_or(0),
                                e
                            )
                        })?
                    }
                };

                let mut part = Part::bytes(file_bytes);

                // Set filename
                if let Some(filename) = Path::new(path).file_name().and_then(|n| n.to_str()) {
                    part = part.file_name(filename.to_string());
                }

                // Set MIME type with sophisticated detection
                let mime_type = mime_type
                    .as_deref()
                    .or_else(|| {
                        Path::new(path)
                            .extension()
                            .and_then(|ext| ext.to_str())
                            .and_then(|ext| match ext.to_lowercase().as_str() {
                                "mp4" => Some("video/mp4"),
                                "mp3" => Some("audio/mp3"),
                                "wav" => Some("audio/wav"),
                                "m4a" => Some("audio/mp4"),
                                "aac" => Some("audio/aac"),
                                "flac" => Some("audio/flac"),
                                "ogg" => Some("audio/ogg"),
                                "webm" => Some("video/webm"),
                                "avi" => Some("video/avi"),
                                "mov" => Some("video/quicktime"),
                                _ => None,
                            })
                    })
                    .ok_or_else(|| format!("Could not determine MIME type for file: {}", path))?;

                part = part.mime_str(mime_type)?;
                form = form.part("file", part);
            }

            DubbingSource::Url { url, .. } => {
                form = form.text("source_url", url.clone());
            }

            DubbingSource::Bytes {
                data,
                filename,
                mime_type,
                ..
            } => {
                let mut part = Part::bytes(data.clone());
                part = part.file_name(filename.clone());
                part = part.mime_str(mime_type)?;
                form = form.part("file", part);
            }
        }

        // Add required target language
        form = form.text("target_lang", body.target_lang.clone());

        // Add optional parameters with validation
        if let Some(name) = &body.name {
            form = form.text("name", name.clone());
        }

        if let Some(source_lang) = &body.source_lang {
            form = form.text("source_lang", source_lang.clone());
        }

        if let Some(target_accent) = &body.target_accent {
            form = form.text("target_accent", target_accent.clone());
        }

        if let Some(num_speakers) = body.num_speakers {
            form = form.text("num_speakers", num_speakers.to_string());
        }

        if let Some(watermark) = body.watermark {
            form = form.text("watermark", watermark.to_string());
        }

        if let Some(start_time) = body.start_time {
            form = form.text("start_time", start_time.to_string());
        }

        if let Some(end_time) = body.end_time {
            form = form.text("end_time", end_time.to_string());
        }

        if let Some(highest_resolution) = body.highest_resolution {
            form = form.text("highest_resolution", highest_resolution.to_string());
        }

        if let Some(drop_background_audio) = body.drop_background_audio {
            form = form.text("drop_background_audio", drop_background_audio.to_string());
        }

        if let Some(use_profanity_filter) = body.use_profanity_filter {
            form = form.text("use_profanity_filter", use_profanity_filter.to_string());
        }

        if let Some(dubbing_studio) = body.dubbing_studio {
            form = form.text("dubbing_studio", dubbing_studio.to_string());
        }

        if let Some(disable_voice_cloning) = body.disable_voice_cloning {
            form = form.text("disable_voice_cloning", disable_voice_cloning.to_string());
        }

        if let Some(mode) = &body.mode {
            form = form.text("mode", mode.clone());
        }

        Ok(RequestBody::Multipart(form))
    }
}
