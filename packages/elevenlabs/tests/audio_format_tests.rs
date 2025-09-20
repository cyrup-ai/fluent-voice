use fluent_voice_elevenlabs::audio_decoders::{AudioFormatDecoder, create_decoder};
use fluent_voice_elevenlabs::audio_format_detection::{
    AudioFormatDetector, DetectionLayer, ConfidenceLevel, DetectedFormat, 
    FormatDetectionResult, AudioMetadata, AudioParams
};
use fluent_voice_elevenlabs::endpoints::genai::voice_changer::VoiceChangerQuery;
use fluent_voice_elevenlabs::shared::query_params::OutputFormat;
use fluent_voice_elevenlabs::error::Error;
use http::HeaderMap;
use std::time::Instant;

#[cfg(test)]
mod audio_format_detection_tests {
    use super::*;

    /// Test Layer 1: Query Parameter Detection
    #[test]
    fn test_query_parameter_detection() {
        let detector = AudioFormatDetector::new();
        let mut query = VoiceChangerQuery::default();
        query = query.with_output_format(OutputFormat::Mp3_44100Hz128kbps);
        
        let headers = HeaderMap::new();
        let data = b"dummy_audio_data";
        
        let result = detector.detect_format_enhanced(&headers, data, &Some(query));
        
        assert!(result.is_ok());
        let detection_result = result.unwrap();
        
        match detection_result.detection_method {
            DetectionLayer::QueryParameter(format) => {
                assert_eq!(format, OutputFormat::Mp3_44100Hz128kbps);
            }
            _ => panic!("Expected QueryParameter detection method"),
        }
        
        assert_eq!(detection_result.confidence, ConfidenceLevel::High);
        assert!(detection_result.fallback_available);
    }

    /// Test Layer 2: Content-Type Header Detection
    #[test]
    fn test_content_type_detection() {
        let detector = AudioFormatDetector::new();
        let mut headers = HeaderMap::new();
        headers.insert("content-type", "audio/mpeg".parse().unwrap());
        
        let data = b"dummy_audio_data";
        
        let result = detector.detect_format_enhanced(&headers, data, &None);
        
        assert!(result.is_ok());
        let detection_result = result.unwrap();
        
        match detection_result.detection_method {
            DetectionLayer::ContentTypeHeader(content_type) => {
                assert_eq!(content_type, "audio/mpeg");
            }
            _ => panic!("Expected ContentTypeHeader detection method"),
        }
        
        assert_eq!(detection_result.confidence, ConfidenceLevel::High);
    }

    /// Test Layer 3: Magic Number Detection - MP3
    #[test]
    fn test_mp3_magic_number_detection() {
        let detector = AudioFormatDetector::new();
        let headers = HeaderMap::new();
        
        // MP3 with ID3 header
        let mp3_data = b"ID3\x03\x00\x00\x00dummy_mp3_data";
        
        let result = detector.detect_format_enhanced(&headers, mp3_data, &None);
        
        assert!(result.is_ok());
        let detection_result = result.unwrap();
        
        match detection_result.detection_method {
            DetectionLayer::MagicNumber(signature) => {
                assert_eq!(signature.format_name, "MP3");
            }
            _ => panic!("Expected MagicNumber detection method"),
        }
        
        assert_eq!(detection_result.confidence, ConfidenceLevel::Medium);
    }

    /// Test Layer 3: Magic Number Detection - WAV
    #[test]
    fn test_wav_magic_number_detection() {
        let detector = AudioFormatDetector::new();
        let headers = HeaderMap::new();
        
        // WAV with RIFF header
        let wav_data = b"RIFF\x24\x08\x00\x00WAVEdummy_wav_data";
        
        let result = detector.detect_format_enhanced(&headers, wav_data, &None);
        
        assert!(result.is_ok());
        let detection_result = result.unwrap();
        
        match detection_result.detection_method {
            DetectionLayer::MagicNumber(signature) => {
                assert_eq!(signature.format_name, "WAV");
            }
            _ => panic!("Expected MagicNumber detection method"),
        }
    }

    /// Test Layer 3: Magic Number Detection - FLAC
    #[test]
    fn test_flac_magic_number_detection() {
        let detector = AudioFormatDetector::new();
        let headers = HeaderMap::new();
        
        // FLAC signature
        let flac_data = b"fLaCdummy_flac_data";
        
        let result = detector.detect_format_enhanced(&headers, flac_data, &None);
        
        assert!(result.is_ok());
        let detection_result = result.unwrap();
        
        match detection_result.detection_method {
            DetectionLayer::MagicNumber(signature) => {
                assert_eq!(signature.format_name, "FLAC");
            }
            _ => panic!("Expected MagicNumber detection method"),
        }
    }

    /// Test Layer 3: Magic Number Detection - OGG
    #[test]
    fn test_ogg_magic_number_detection() {
        let detector = AudioFormatDetector::new();
        let headers = HeaderMap::new();
        
        // OGG signature
        let ogg_data = b"OggSdummy_ogg_data";
        
        let result = detector.detect_format_enhanced(&headers, ogg_data, &None);
        
        assert!(result.is_ok());
        let detection_result = result.unwrap();
        
        match detection_result.detection_method {
            DetectionLayer::MagicNumber(signature) => {
                assert_eq!(signature.format_name, "OGG");
            }
            _ => panic!("Expected MagicNumber detection method"),
        }
    }

    /// Test Layer 4: Rodio Detection Fallback
    #[test]
    fn test_rodio_detection_fallback() {
        let detector = AudioFormatDetector::new().with_rodio_enabled(true);
        let headers = HeaderMap::new();
        
        // Data that doesn't match magic numbers but might be valid audio
        let unknown_data = b"unknown_audio_format_data";
        
        let result = detector.detect_format_enhanced(&headers, unknown_data, &None);
        
        // This should either succeed with Rodio detection or fail gracefully
        match result {
            Ok(detection_result) => {
                match detection_result.detection_method {
                    DetectionLayer::RodioAutomatic => {
                        assert_eq!(detection_result.confidence, ConfidenceLevel::Medium);
                    }
                    _ => {
                        // If not Rodio, it should be an error fallback
                    }
                }
            }
            Err(Error::UnsupportedMediaFormat { .. }) => {
                // Expected for invalid data
            }
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    /// Test Detection Priority Order
    #[test]
    fn test_detection_priority_order() {
        let detector = AudioFormatDetector::new();
        
        // Create query with MP3 format
        let mut query = VoiceChangerQuery::default();
        query = query.with_output_format(OutputFormat::Mp3_44100Hz64kbps);
        
        // Create headers with different format
        let mut headers = HeaderMap::new();
        headers.insert("content-type", "audio/wav".parse().unwrap());
        
        // Create data with FLAC magic number
        let data = b"fLaCdummy_flac_data";
        
        let result = detector.detect_format_enhanced(&headers, data, &Some(query));
        
        assert!(result.is_ok());
        let detection_result = result.unwrap();
        
        // Query parameter should have highest priority
        match detection_result.detection_method {
            DetectionLayer::QueryParameter(format) => {
                assert_eq!(format, OutputFormat::Mp3_44100Hz64kbps);
            }
            _ => panic!("Query parameter should have highest priority"),
        }
    }

    /// Test Multiple Content-Type Formats
    #[test]
    fn test_multiple_content_types() {
        let detector = AudioFormatDetector::new();
        let data = b"dummy_data";
        
        let test_cases = vec![
            ("audio/mpeg", true),
            ("audio/mp3", true),
            ("audio/wav", true),
            ("audio/wave", true),
            ("audio/x-wav", true),
            ("audio/flac", true),
            ("audio/ogg", true),
            ("audio/vorbis", true),
            ("audio/aac", true),
            ("audio/mp4", true),
            ("audio/basic", true),
            ("text/plain", false),
            ("application/json", false),
        ];
        
        for (content_type, should_succeed) in test_cases {
            let mut headers = HeaderMap::new();
            headers.insert("content-type", content_type.parse().unwrap());
            
            let result = detector.detect_format_enhanced(&headers, data, &None);
            
            if should_succeed {
                assert!(result.is_ok(), "Should succeed for content-type: {}", content_type);
            } else {
                // Should fall through to other detection methods or fail
                match result {
                    Ok(_) => {
                        // Might succeed with other detection methods
                    }
                    Err(Error::UnsupportedMediaFormat { .. }) => {
                        // Expected for unsupported content types
                    }
                    Err(e) => panic!("Unexpected error for {}: {:?}", content_type, e),
                }
            }
        }
    }

    /// Test Edge Case: Empty Data
    #[test]
    fn test_empty_data() {
        let detector = AudioFormatDetector::new();
        let headers = HeaderMap::new();
        let empty_data = b"";
        
        let result = detector.detect_format_enhanced(&headers, empty_data, &None);
        
        assert!(result.is_err());
        match result.unwrap_err() {
            Error::UnsupportedMediaFormat { .. } => {
                // Expected
            }
            e => panic!("Unexpected error for empty data: {:?}", e),
        }
    }

    /// Test Edge Case: Very Small Data
    #[test]
    fn test_very_small_data() {
        let detector = AudioFormatDetector::new();
        let headers = HeaderMap::new();
        let small_data = b"ab";
        
        let result = detector.detect_format_enhanced(&headers, small_data, &None);
        
        // Should handle gracefully
        match result {
            Ok(_) => {
                // Might succeed if detected by some method
            }
            Err(Error::UnsupportedMediaFormat { .. }) => {
                // Expected for insufficient data
            }
            Err(e) => panic!("Unexpected error for small data: {:?}", e),
        }
    }

    /// Test Edge Case: Invalid Headers
    #[test]
    fn test_invalid_headers() {
        let detector = AudioFormatDetector::new();
        let mut headers = HeaderMap::new();
        headers.insert("content-type", "invalid/format".parse().unwrap());
        
        let data = b"some_audio_data";
        
        let result = detector.detect_format_enhanced(&headers, data, &None);
        
        // Should fall through to other detection methods
        match result {
            Ok(_) => {
                // Might succeed with magic number or Rodio detection
            }
            Err(Error::UnsupportedMediaFormat { .. }) => {
                // Expected if no other methods succeed
            }
            Err(e) => panic!("Unexpected error for invalid headers: {:?}", e),
        }
    }

    /// Test Configuration Options
    #[test]
    fn test_detector_configuration() {
        // Test with Rodio disabled
        let detector_no_rodio = AudioFormatDetector::new().with_rodio_enabled(false);
        
        // Test with Symphonia disabled
        let detector_no_symphonia = AudioFormatDetector::new().with_symphonia_enabled(false);
        
        // Test with both enabled
        let detector_full = AudioFormatDetector::new()
            .with_rodio_enabled(true)
            .with_symphonia_enabled(true);
        
        let headers = HeaderMap::new();
        let data = b"unknown_format_data";
        
        // All should handle the request gracefully
        for detector in [detector_no_rodio, detector_no_symphonia, detector_full] {
            let result = detector.detect_format_enhanced(&headers, data, &None);
            
            match result {
                Ok(_) => {
                    // Success is possible
                }
                Err(Error::UnsupportedMediaFormat { .. }) => {
                    // Expected for unknown format
                }
                Err(e) => panic!("Unexpected error: {:?}", e),
            }
        }
    }

    /// Test Error Handling for Invalid Query Parameters
    #[test]
    fn test_invalid_query_parameters() {
        let detector = AudioFormatDetector::new();
        let headers = HeaderMap::new();
        let data = b"dummy_data";
        
        // Create query with invalid format string (this would be caught at parse time)
        let mut query = VoiceChangerQuery::default();
        // Manually add invalid format to params
        query.params.push(("output_format", "invalid_format".to_string()));
        
        let result = detector.detect_format_enhanced(&headers, data, &Some(query));
        
        // Should fall through to other detection methods since parsing fails
        match result {
            Ok(_) => {
                // Might succeed with other methods
            }
            Err(Error::UnsupportedMediaFormat { .. }) => {
                // Expected if no other methods work
            }
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    /// Performance Benchmark Test
    #[test]
    fn test_detection_performance() {
        let detector = AudioFormatDetector::new();
        let headers = HeaderMap::new();
        
        // Test with MP3 magic number
        let mp3_data = b"ID3\x03\x00\x00\x00dummy_mp3_data_for_performance_test";
        
        let start_time = Instant::now();
        let iterations = 1000;
        
        for _ in 0..iterations {
            let result = detector.detect_format_enhanced(&headers, mp3_data, &None);
            assert!(result.is_ok());
        }
        
        let elapsed = start_time.elapsed();
        let avg_time = elapsed / iterations;
        
        // Detection should be fast (less than 1ms per detection on average)
        assert!(avg_time.as_millis() < 1, 
            "Detection too slow: {}ms average", avg_time.as_millis());
        
        println!("Performance: {} detections in {:?} (avg: {:?})", 
            iterations, elapsed, avg_time);
    }

    /// Test Metadata Extraction
    #[test]
    fn test_metadata_extraction() {
        let detector = AudioFormatDetector::new();
        let mut query = VoiceChangerQuery::default();
        query = query.with_output_format(OutputFormat::Pcm44100Hz);
        
        let headers = HeaderMap::new();
        let data = b"dummy_data";
        
        let result = detector.detect_format_enhanced(&headers, data, &Some(query));
        
        assert!(result.is_ok());
        let detection_result = result.unwrap();
        
        // Check metadata
        assert!(detection_result.metadata.is_some());
        let metadata = detection_result.metadata.unwrap();
        
        assert_eq!(metadata.sample_rate, Some(44100));
        assert_eq!(metadata.channels, Some(2));
        assert_eq!(metadata.bit_depth, Some(16));
    }

    /// Test All OutputFormat Variants
    #[test]
    fn test_all_output_formats() {
        let detector = AudioFormatDetector::new();
        let headers = HeaderMap::new();
        let data = b"dummy_data";
        
        let formats = vec![
            OutputFormat::Mp3_22050Hz32kbps,
            OutputFormat::Mp3_44100Hz32kbps,
            OutputFormat::Mp3_44100Hz64kbps,
            OutputFormat::Mp3_44100Hz96kbps,
            OutputFormat::Mp3_44100Hz128kbps,
            OutputFormat::Mp3_44100Hz192kbps,
            OutputFormat::Pcm8000Hz,
            OutputFormat::Pcm16000Hz,
            OutputFormat::Pcm22050Hz,
            OutputFormat::Pcm24000Hz,
            OutputFormat::Pcm44100Hz,
            OutputFormat::MuLaw8000Hz,
            OutputFormat::Opus48000Hz32kbps,
            OutputFormat::Opus48000Hz64kbps,
            OutputFormat::Opus48000Hz96kbps,
            OutputFormat::Opus48000Hz128kbps,
            OutputFormat::Opus48000Hz192kbps,
        ];
        
        for format in formats {
            let mut query = VoiceChangerQuery::default();
            query = query.with_output_format(format.clone());
            
            let result = detector.detect_format_enhanced(&headers, data, &Some(query));
            
            assert!(result.is_ok(), "Failed for format: {:?}", format);
            
            let detection_result = result.unwrap();
            match detection_result.detected_format {
                DetectedFormat::Known { output_format, .. } => {
                    assert_eq!(output_format, format);
                }
                _ => panic!("Expected Known format for: {:?}", format),
            }
        }
    }
}

#[cfg(test)]
mod audio_decoder_tests {
    use super::*;

    /// Test Decoder Creation
    #[test]
    fn test_decoder_creation() {
        let known_format = DetectedFormat::Known {
            output_format: OutputFormat::Mp3_44100Hz128kbps,
            sample_rate: 44100,
            channels: 2,
            bit_depth: 16,
        };
        
        let decoder = create_decoder(&known_format);
        let format_info = decoder.get_format_info();
        
        assert_eq!(format_info.name, "MP3");
        assert_eq!(format_info.sample_rate, 44100);
    }

    /// Test Unknown Format Decoder
    #[test]
    fn test_unknown_format_decoder() {
        let unknown_format = DetectedFormat::Unknown {
            mime_type: Some("audio/unknown".to_string()),
            estimated_params: AudioParams {
                sample_rate: 48000,
                channels: 1,
                bit_depth: 24,
            },
        };
        
        let decoder = create_decoder(&unknown_format);
        let format_info = decoder.get_format_info();
        
        assert_eq!(format_info.name, "Universal");
        assert_eq!(format_info.sample_rate, 48000);
        assert_eq!(format_info.channels, 1);
        assert_eq!(format_info.bit_depth, 24);
    }

    /// Test Advanced Format Decoder
    #[test]
    fn test_advanced_format_decoder() {
        let advanced_format = DetectedFormat::Advanced {
            format_name: "Advanced_Codec".to_string(),
            metadata: AudioMetadata {
                sample_rate: Some(96000),
                channels: Some(6),
                bit_depth: Some(32),
                duration: Some(120.5),
            },
        };
        
        let decoder = create_decoder(&advanced_format);
        let format_info = decoder.get_format_info();
        
        assert_eq!(format_info.name, "Advanced_Codec");
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    /// Integration Test: Full Detection and Decoding Pipeline
    #[test]
    fn test_full_pipeline() {
        let detector = AudioFormatDetector::new();
        
        // Test with query parameter
        let mut query = VoiceChangerQuery::default();
        query = query.with_output_format(OutputFormat::Pcm44100Hz);
        
        let headers = HeaderMap::new();
        let data = b"dummy_pcm_data_for_integration_test";
        
        // Step 1: Detect format
        let detection_result = detector.detect_format_enhanced(&headers, data, &Some(query));
        assert!(detection_result.is_ok());
        
        let format_result = detection_result.unwrap();
        
        // Step 2: Create decoder
        let decoder = create_decoder(&format_result.detected_format);
        
        // Step 3: Verify decoder properties
        let format_info = decoder.get_format_info();
        assert_eq!(format_info.name, "PCM");
        assert_eq!(format_info.sample_rate, 44100);
        
        // Step 4: Test streaming support
        let supports_streaming = decoder.supports_streaming();
        assert!(supports_streaming); // PCM should support streaming
    }

    /// Integration Test: Error Recovery
    #[test]
    fn test_error_recovery_pipeline() {
        let detector = AudioFormatDetector::new();
        let headers = HeaderMap::new();
        let invalid_data = b"completely_invalid_audio_data";
        
        let result = detector.detect_format_enhanced(&headers, invalid_data, &None);
        
        match result {
            Ok(detection_result) => {
                // If detection succeeds, decoder should handle gracefully
                let decoder = create_decoder(&detection_result.detected_format);
                let decode_result = decoder.decode_to_pcm(invalid_data);
                
                // Decoding might fail, which is expected for invalid data
                match decode_result {
                    Ok(_) => {
                        // Unexpected success, but not necessarily wrong
                    }
                    Err(_) => {
                        // Expected for invalid data
                    }
                }
            }
            Err(Error::UnsupportedMediaFormat { supported, .. }) => {
                // Expected error with helpful message
                assert!(!supported.is_empty());
            }
            Err(e) => panic!("Unexpected error type: {:?}", e),
        }
    }
}

/// Feature Flag Tests
#[cfg(test)]
mod feature_flag_tests {
    use super::*;

    /// Test with advanced_audio feature disabled (default)
    #[test]
    fn test_without_advanced_audio_feature() {
        let detector = AudioFormatDetector::new().with_symphonia_enabled(true);
        
        // This should work even without the feature flag
        let headers = HeaderMap::new();
        let data = b"test_data";
        
        let result = detector.detect_format_enhanced(&headers, data, &None);
        
        // Should handle gracefully regardless of feature flag
        match result {
            Ok(_) => {
                // Success is possible
            }
            Err(Error::UnsupportedMediaFormat { .. }) => {
                // Expected for unknown format
            }
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    /// Test Advanced Format Decoder Creation
    #[test]
    fn test_advanced_decoder_creation() {
        let advanced_format = DetectedFormat::Advanced {
            format_name: "TestCodec".to_string(),
            metadata: AudioMetadata {
                sample_rate: Some(48000),
                channels: Some(2),
                bit_depth: Some(16),
                duration: None,
            },
        };
        
        let decoder = create_decoder(&advanced_format);
        
        // Should create appropriate decoder based on feature availability
        let format_info = decoder.get_format_info();
        
        #[cfg(feature = "advanced_audio")]
        {
            assert_eq!(format_info.name, "TestCodec");
        }
        
        #[cfg(not(feature = "advanced_audio"))]
        {
            // Should fall back to universal decoder
            assert_eq!(format_info.name, "Universal");
        }
    }
}
