use crate::endpoints::genai::voice_changer::VoiceChangerQuery;
use crate::error::Error;
use crate::shared::query_params::OutputFormat;
use http::HeaderMap;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum DetectionLayer {
    QueryParameter(OutputFormat),
    ContentTypeHeader(String),
    MagicNumber(FormatSignature),
    RodioAutomatic,
    SymphoniaAdvanced,
    ContextualInference,
}

#[derive(Debug, Clone)]
pub enum ConfidenceLevel {
    High,   // 90-100% confidence
    Medium, // 70-89% confidence
    Low,    // 50-69% confidence
}

#[derive(Debug, Clone)]
pub struct FormatSignature {
    pub signature: Vec<u8>,
    pub format_name: String,
    pub output_format: Option<OutputFormat>,
}

#[derive(Debug, Clone)]
pub struct AudioMetadata {
    pub sample_rate: Option<u32>,
    pub channels: Option<u16>,
    pub bit_depth: Option<u16>,
    pub duration: Option<f64>,
}

#[derive(Debug, Clone)]
pub enum DetectedFormat {
    Known {
        output_format: OutputFormat,
        sample_rate: u32,
        channels: u16,
        bit_depth: u16,
    },
    Unknown {
        mime_type: Option<String>,
        estimated_params: AudioParams,
    },
    Advanced {
        format_name: String,
        metadata: AudioMetadata,
    },
}

#[derive(Debug, Clone)]
pub struct AudioParams {
    pub sample_rate: u32,
    pub channels: u16,
    pub bit_depth: u16,
}

impl Default for AudioParams {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            channels: 2,
            bit_depth: 16,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FormatDetectionResult {
    pub detected_format: DetectedFormat,
    pub confidence: ConfidenceLevel,
    pub detection_method: DetectionLayer,
    pub metadata: Option<AudioMetadata>,
    pub fallback_available: bool,
}

pub struct MagicNumberDatabase {
    signatures: Vec<FormatSignature>,
}

impl Default for MagicNumberDatabase {
    fn default() -> Self {
        let mut signatures = Vec::new();

        // MP3 signatures
        signatures.push(FormatSignature {
            signature: b"ID3".to_vec(),
            format_name: "MP3".to_string(),
            output_format: Some(OutputFormat::Mp3_44100Hz128kbps),
        });
        signatures.push(FormatSignature {
            signature: vec![0xFF, 0xFB],
            format_name: "MP3".to_string(),
            output_format: Some(OutputFormat::Mp3_44100Hz128kbps),
        });
        signatures.push(FormatSignature {
            signature: vec![0xFF, 0xF3],
            format_name: "MP3".to_string(),
            output_format: Some(OutputFormat::Mp3_44100Hz128kbps),
        });
        signatures.push(FormatSignature {
            signature: vec![0xFF, 0xF2],
            format_name: "MP3".to_string(),
            output_format: Some(OutputFormat::Mp3_44100Hz128kbps),
        });

        // WAV signature
        signatures.push(FormatSignature {
            signature: b"RIFF".to_vec(),
            format_name: "WAV".to_string(),
            output_format: Some(OutputFormat::Pcm44100Hz),
        });

        // FLAC signature
        signatures.push(FormatSignature {
            signature: b"fLaC".to_vec(),
            format_name: "FLAC".to_string(),
            output_format: Some(OutputFormat::Pcm44100Hz),
        });

        // OGG signature
        signatures.push(FormatSignature {
            signature: b"OggS".to_vec(),
            format_name: "OGG".to_string(),
            output_format: Some(OutputFormat::Pcm44100Hz),
        });

        // AAC signatures
        signatures.push(FormatSignature {
            signature: vec![0xFF, 0xF1],
            format_name: "AAC".to_string(),
            output_format: Some(OutputFormat::Mp3_44100Hz128kbps),
        });
        signatures.push(FormatSignature {
            signature: vec![0xFF, 0xF9],
            format_name: "AAC".to_string(),
            output_format: Some(OutputFormat::Mp3_44100Hz128kbps),
        });

        Self { signatures }
    }
}

impl MagicNumberDatabase {
    pub fn detect_format(&self, data: &[u8]) -> Option<FormatSignature> {
        for signature in &self.signatures {
            if data.len() >= signature.signature.len() && data.starts_with(&signature.signature) {
                return Some(signature.clone());
            }
        }
        None
    }
}

pub struct ContentTypeMappings {
    mappings: HashMap<String, OutputFormat>,
}

impl Default for ContentTypeMappings {
    fn default() -> Self {
        let mut mappings = HashMap::new();

        // Standard MIME types
        mappings.insert("audio/mpeg".to_string(), OutputFormat::Mp3_44100Hz128kbps);
        mappings.insert("audio/mp3".to_string(), OutputFormat::Mp3_44100Hz128kbps);
        mappings.insert("audio/wav".to_string(), OutputFormat::Pcm44100Hz);
        mappings.insert("audio/wave".to_string(), OutputFormat::Pcm44100Hz);
        mappings.insert("audio/x-wav".to_string(), OutputFormat::Pcm44100Hz);
        mappings.insert("audio/flac".to_string(), OutputFormat::Pcm44100Hz);
        mappings.insert("audio/ogg".to_string(), OutputFormat::Pcm44100Hz);
        mappings.insert("audio/vorbis".to_string(), OutputFormat::Pcm44100Hz);
        mappings.insert("audio/aac".to_string(), OutputFormat::Mp3_44100Hz128kbps);
        mappings.insert("audio/mp4".to_string(), OutputFormat::Mp3_44100Hz128kbps);
        mappings.insert("audio/basic".to_string(), OutputFormat::MuLaw8000Hz);

        Self { mappings }
    }
}

impl ContentTypeMappings {
    pub fn get_format(&self, content_type: &str) -> Option<OutputFormat> {
        // Extract the main MIME type, ignoring parameters
        let main_type = content_type.split(';').next()?.trim().to_lowercase();
        self.mappings.get(&main_type).cloned()
    }
}

pub struct AudioFormatDetector {
    rodio_enabled: bool,
    symphonia_enabled: bool,
    magic_number_db: MagicNumberDatabase,
    content_type_mappings: ContentTypeMappings,
}

impl AudioFormatDetector {
    pub fn new() -> Self {
        Self {
            rodio_enabled: true,
            symphonia_enabled: cfg!(feature = "advanced_audio"),
            magic_number_db: MagicNumberDatabase::default(),
            content_type_mappings: ContentTypeMappings::default(),
        }
    }

    pub fn with_rodio_enabled(mut self, enabled: bool) -> Self {
        self.rodio_enabled = enabled;
        self
    }

    pub fn with_symphonia_enabled(mut self, enabled: bool) -> Self {
        self.symphonia_enabled = enabled;
        self
    }

    pub fn detect_format_enhanced(
        &self,
        headers: &HeaderMap,
        data: &[u8],
        query: &Option<VoiceChangerQuery>,
    ) -> Result<FormatDetectionResult, Error> {
        // Layer 1: Query parameter hints
        if let Some(query) = query {
            // Extract output_format from query parameters
            for (key, value) in &query.params {
                if *key == "output_format" {
                    if let Ok(format) = value.parse::<OutputFormat>() {
                        return Ok(FormatDetectionResult {
                            detected_format: DetectedFormat::Known {
                                output_format: format.clone(),
                                sample_rate: self.get_default_sample_rate(&format),
                                channels: 2,
                                bit_depth: 16,
                            },
                            confidence: ConfidenceLevel::High,
                            detection_method: DetectionLayer::QueryParameter(format.clone()),
                            metadata: Some(AudioMetadata {
                                sample_rate: Some(self.get_default_sample_rate(&format)),
                                channels: Some(2),
                                bit_depth: Some(16),
                                duration: None,
                            }),
                            fallback_available: true,
                        });
                    }
                }
            }
        }

        // Layer 2: Content-Type header analysis
        if let Some(content_type) = headers.get("content-type") {
            if let Ok(content_type_str) = content_type.to_str() {
                if let Some(format) = self.content_type_mappings.get_format(content_type_str) {
                    return Ok(FormatDetectionResult {
                        detected_format: DetectedFormat::Known {
                            output_format: format.clone(),
                            sample_rate: self.get_default_sample_rate(&format),
                            channels: 2,
                            bit_depth: 16,
                        },
                        confidence: ConfidenceLevel::High,
                        detection_method: DetectionLayer::ContentTypeHeader(
                            content_type_str.to_string(),
                        ),
                        metadata: Some(AudioMetadata {
                            sample_rate: Some(self.get_default_sample_rate(&format)),
                            channels: Some(2),
                            bit_depth: Some(16),
                            duration: None,
                        }),
                        fallback_available: true,
                    });
                }
            }
        }

        // Layer 3: Magic number detection
        if let Some(signature) = self.magic_number_db.detect_format(data) {
            if let Some(format) = &signature.output_format {
                let signature_clone = signature.clone();
                return Ok(FormatDetectionResult {
                    detected_format: DetectedFormat::Known {
                        output_format: format.clone(),
                        sample_rate: self.get_default_sample_rate(format),
                        channels: 2,
                        bit_depth: 16,
                    },
                    confidence: ConfidenceLevel::Medium,
                    detection_method: DetectionLayer::MagicNumber(signature_clone),
                    metadata: Some(AudioMetadata {
                        sample_rate: Some(self.get_default_sample_rate(format)),
                        channels: Some(2),
                        bit_depth: Some(16),
                        duration: None,
                    }),
                    fallback_available: true,
                });
            }
        }

        // Layer 4: Rodio automatic detection
        if self.rodio_enabled {
            if let Ok(detection_result) = self.detect_with_rodio(data) {
                return Ok(detection_result);
            }
        }

        // Layer 5: Symphonia advanced detection (feature-gated)
        #[cfg(feature = "advanced_audio")]
        if self.symphonia_enabled {
            if let Ok(detection_result) = self.detect_with_symphonia(data) {
                return Ok(detection_result);
            }
        }

        Err(Error::UnsupportedMediaFormat {
            extension: "unknown".to_string(),
            supported: self.get_supported_formats_list(),
        })
    }

    fn detect_with_rodio(&self, data: &[u8]) -> Result<FormatDetectionResult, Error> {
        use rodio::Decoder;
        use std::io::Cursor;

        let data_owned = data.to_vec();
        let cursor = Cursor::new(data_owned);
        match Decoder::try_from(cursor) {
            Ok(_decoder) => {
                // Get format info from the first few samples
                let sample_rate = 44100; // Default, Rodio doesn't expose this directly
                let channels = 2; // Default stereo

                Ok(FormatDetectionResult {
                    detected_format: DetectedFormat::Unknown {
                        mime_type: None,
                        estimated_params: AudioParams {
                            sample_rate,
                            channels,
                            bit_depth: 16, // Rodio typically outputs 16-bit
                        },
                    },
                    confidence: ConfidenceLevel::Medium,
                    detection_method: DetectionLayer::RodioAutomatic,
                    metadata: Some(AudioMetadata {
                        sample_rate: Some(sample_rate),
                        channels: Some(channels),
                        bit_depth: Some(16),
                        duration: None,
                    }),
                    fallback_available: false,
                })
            }
            Err(_) => Err(Error::UnsupportedMediaFormat {
                extension: "rodio-failed".to_string(),
                supported: "MP3, FLAC, Vorbis, WAV".to_string(),
            }),
        }
    }

    #[cfg(feature = "advanced_audio")]
    fn detect_with_symphonia(&self, data: &[u8]) -> Result<FormatDetectionResult, Error> {
        use std::io::Cursor;
        use symphonia::core::io::MediaSourceStream;
        use symphonia::core::probe::Hint;

        let cursor = Cursor::new(data);
        let mss = MediaSourceStream::new(Box::new(cursor), Default::default());

        let hint = Hint::new();

        match symphonia::default::get_probe().format(
            &hint,
            mss,
            &Default::default(),
            &Default::default(),
        ) {
            Ok(probed) => {
                let format = probed.format;
                if let Some(track) = format.tracks().iter().next() {
                    let codec_params = &track.codec_params;

                    Ok(FormatDetectionResult {
                        detected_format: DetectedFormat::Advanced {
                            format_name: format!("{:?}", codec_params.codec),
                            metadata: AudioMetadata {
                                sample_rate: codec_params.sample_rate,
                                channels: codec_params.channels.map(|c| c.count() as u16),
                                bit_depth: codec_params.bits_per_sample.map(|b| b as u16),
                                duration: None,
                            },
                        },
                        confidence: ConfidenceLevel::High,
                        detection_method: DetectionLayer::SymphoniaAdvanced,
                        metadata: Some(AudioMetadata {
                            sample_rate: codec_params.sample_rate,
                            channels: codec_params.channels.map(|c| c.count() as u16),
                            bit_depth: codec_params.bits_per_sample.map(|b| b as u16),
                            duration: None,
                        }),
                        fallback_available: true,
                    })
                } else {
                    Err(Error::UnsupportedMediaFormat {
                        extension: "no-audio-track".to_string(),
                        supported: "Files with audio tracks".to_string(),
                    })
                }
            }
            Err(_) => Err(Error::UnsupportedMediaFormat {
                extension: "symphonia-failed".to_string(),
                supported: "Symphonia-supported formats".to_string(),
            }),
        }
    }

    fn get_default_sample_rate(&self, format: &OutputFormat) -> u32 {
        match format {
            OutputFormat::Pcm8000Hz | OutputFormat::MuLaw8000Hz => 8000,
            OutputFormat::Pcm16000Hz => 16000,
            OutputFormat::Mp3_22050Hz32kbps | OutputFormat::Pcm22050Hz => 22050,
            OutputFormat::Pcm24000Hz => 24000,
            OutputFormat::Mp3_44100Hz32kbps
            | OutputFormat::Mp3_44100Hz64kbps
            | OutputFormat::Mp3_44100Hz96kbps
            | OutputFormat::Mp3_44100Hz128kbps
            | OutputFormat::Mp3_44100Hz192kbps
            | OutputFormat::Pcm44100Hz => 44100,
            OutputFormat::Opus48000Hz32kbps
            | OutputFormat::Opus48000Hz64kbps
            | OutputFormat::Opus48000Hz96kbps
            | OutputFormat::Opus48000Hz128kbps
            | OutputFormat::Opus48000Hz192kbps => 48000,
        }
    }

    fn get_supported_formats_list(&self) -> String {
        let mut formats = vec!["MP3", "WAV", "FLAC", "OGG", "AAC"];

        if self.rodio_enabled {
            formats.push("Rodio-supported");
        }

        #[cfg(feature = "advanced_audio")]
        if self.symphonia_enabled {
            formats.push("Symphonia-advanced");
        }

        formats.join(", ")
    }

    /// Detect μ-law format with Twilio-specific context awareness
    pub fn detect_twilio_mulaw(
        &self,
        headers: &HeaderMap,
        _data: &[u8],
    ) -> Option<FormatDetectionResult> {
        // Priority 1: Explicit μ-law MIME types
        if let Some(content_type) = headers.get("content-type") {
            if let Ok(content_type_str) = content_type.to_str() {
                let ct_lower = content_type_str.to_lowercase();
                if ct_lower.contains("audio/basic")
                    || ct_lower.contains("audio/x-mulaw")
                    || ct_lower.contains("audio/ulaw")
                {
                    return Some(self.create_mulaw_detection_result(
                        ConfidenceLevel::High,
                        DetectionLayer::ContentTypeHeader(content_type_str.to_string()),
                    ));
                }
            }
        }

        // Priority 2: Twilio WebSocket context indicators
        if self.is_twilio_websocket_context(headers) {
            return Some(self.create_mulaw_detection_result(
                ConfidenceLevel::Medium,
                DetectionLayer::ContextualInference,
            ));
        }

        // Priority 3: Twilio HTTP context indicators
        if self.is_twilio_http_context(headers) {
            return Some(self.create_mulaw_detection_result(
                ConfidenceLevel::Low,
                DetectionLayer::ContextualInference,
            ));
        }

        None
    }

    fn is_twilio_websocket_context(&self, headers: &HeaderMap) -> bool {
        // Check for Twilio WebSocket signatures
        headers.get("x-twilio-signature").is_some()
            || headers.get("upgrade").map_or(false, |h| {
                h.to_str().unwrap_or("").eq_ignore_ascii_case("websocket")
            }) && headers.get("user-agent").map_or(false, |ua| {
                let ua_str = ua.to_str().unwrap_or("");
                ua_str.contains("TwilioProxy") || ua_str.contains("Twilio")
            })
    }

    fn is_twilio_http_context(&self, headers: &HeaderMap) -> bool {
        // Check for Twilio HTTP request indicators
        headers.get("x-twilio-signature").is_some()
            || headers.get("user-agent").map_or(false, |ua| {
                ua.to_str().unwrap_or("").contains("TwilioProxy")
            })
    }

    fn create_mulaw_detection_result(
        &self,
        confidence: ConfidenceLevel,
        detection_method: DetectionLayer,
    ) -> FormatDetectionResult {
        FormatDetectionResult {
            detected_format: DetectedFormat::Known {
                output_format: OutputFormat::MuLaw8000Hz,
                sample_rate: 8000,
                channels: 1,
                bit_depth: 8,
            },
            confidence,
            detection_method,
            metadata: Some(AudioMetadata {
                sample_rate: Some(8000),
                channels: Some(1),
                bit_depth: Some(8),
                duration: None,
            }),
            fallback_available: true,
        }
    }
}

impl Default for AudioFormatDetector {
    fn default() -> Self {
        Self::new()
    }
}
