#![allow(dead_code)]

use serde_json::Value;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("reqwest error: {0}")]
    ReqwestError(#[from] reqwest::Error),
    #[error("serde error: {0}")]
    SerdeError(#[from] serde_json::Error),
    #[error("http error: {0}")]
    HttpError(Value),
    #[error("file extension not found")]
    FileExtensionNotFound,
    #[error("file extension not valid utf8")]
    FileExtensionNotValidUTF8,
    #[error("file extension not supported")]
    FileExtensionNotSupported,
    #[error("path not valid utf8")]
    PathNotValidUTF8,
    #[error("voice not found")]
    VoiceNotFound,
    #[error("generated voice id header not found")]
    GeneratedVoiceIDHeaderNotFound,
    #[error("Invalid dubbing source: {0}")]
    InvalidDubbingSource(String),
    #[error("Unsupported media format: {extension}. Supported formats: {supported}")]
    UnsupportedMediaFormat {
        extension: String,
        supported: String,
    },
    #[error("Invalid time range: start={start}, end={end}")]
    InvalidTimeRange { start: f32, end: f32 },
    #[error("Invalid speaker count: {count}. Must be between 1 and 50")]
    InvalidSpeakerCount { count: u32 },
    #[error("File validation failed: {path} - {reason}")]
    FileValidationFailed { path: String, reason: String },
    #[error("URL validation failed: {url} - {reason}")]
    UrlValidationFailed { url: String, reason: String },
    #[error("Authentication required for URL: {url}")]
    AuthenticationRequired { url: String },
    #[error("Audio format detection failed: {reason}. Tried methods: {methods_tried}")]
    FormatDetectionFailed {
        reason: String,
        methods_tried: String,
    },
    #[error("Audio decoding failed for format {format}: {details}")]
    AudioDecodingFailed { format: String, details: String },
    #[error(
        "Unsupported audio parameters: sample_rate={sample_rate}, channels={channels}, bit_depth={bit_depth}"
    )]
    UnsupportedAudioParameters {
        sample_rate: u32,
        channels: u16,
        bit_depth: u16,
    },
    #[error("Transcript parsing failed for {format}: {reason}")]
    TranscriptParseError {
        format: String,
        reason: String,
    },
}

#[derive(Error, Debug)]
pub enum WebSocketError {
    #[error("NonNormalCloseCode: {0}")]
    NonNormalCloseCode(String),
    #[error("ClosedWithoutCloseFrame")]
    ClosedWithoutCloseFrame,
    #[error("UnexpectedMessageType")]
    UnexpectedMessageType,
}
