//! Export utilities for timestamp data to various subtitle formats
//!
//! This module provides functionality to export timestamp metadata to
//! industry-standard subtitle formats like SRT and WebVTT.

use crate::timestamp_metadata::TimestampMetadata;

impl TimestampMetadata {
    /// Export to SRT subtitle format
    pub fn to_srt(&self) -> Result<String, crate::engine::FluentVoiceError> {
        if let Some(word_alignments) = &self.word_alignments {
            let mut srt = String::new();
            for (i, word) in word_alignments.iter().enumerate() {
                let start_time = seconds_to_srt_time(word.start_seconds);
                let end_time = seconds_to_srt_time(word.end_seconds);
                srt.push_str(&format!(
                    "{}\n{} --> {}\n{}\n\n",
                    i + 1,
                    start_time,
                    end_time,
                    word.word
                ));
            }
            Ok(srt)
        } else {
            Err(crate::engine::FluentVoiceError::ConfigError(
                "No word alignments available for SRT export".into(),
            ))
        }
    }

    /// Export to WebVTT format
    pub fn to_vtt(&self) -> Result<String, crate::FluentVoiceError> {
        if let Some(word_alignments) = &self.word_alignments {
            let mut vtt = String::from("WEBVTT\n\n");
            for word in word_alignments {
                let start_time = seconds_to_vtt_time(word.start_seconds);
                let end_time = seconds_to_vtt_time(word.end_seconds);
                vtt.push_str(&format!(
                    "{} --> {}\n{}\n\n",
                    start_time, end_time, word.word
                ));
            }
            Ok(vtt)
        } else {
            Err(crate::FluentVoiceError::ConfigError(
                "No word alignments available for VTT export".into(),
            ))
        }
    }
}

/// Convert seconds to SRT time format (HH:MM:SS,mmm)
pub fn seconds_to_srt_time(seconds: f32) -> String {
    let total_ms = (seconds * 1000.0) as u64;
    let hours = total_ms / 3600000;
    let minutes = (total_ms % 3600000) / 60000;
    let secs = (total_ms % 60000) / 1000;
    let ms = total_ms % 1000;
    format!("{:02}:{:02}:{:02},{:03}", hours, minutes, secs, ms)
}

/// Convert seconds to WebVTT time format (HH:MM:SS.mmm)
pub fn seconds_to_vtt_time(seconds: f32) -> String {
    let total_ms = (seconds * 1000.0) as u64;
    let hours = total_ms / 3600000;
    let minutes = (total_ms % 3600000) / 60000;
    let secs = (total_ms % 60000) / 1000;
    let ms = total_ms % 1000;
    format!("{:02}:{:02}:{:02}.{:03}", hours, minutes, secs, ms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seconds_to_srt_time() {
        assert_eq!(seconds_to_srt_time(0.0), "00:00:00,000");
        assert_eq!(seconds_to_srt_time(61.5), "00:01:01,500");
        assert_eq!(seconds_to_srt_time(3661.123), "01:01:01,123");
    }

    #[test]
    fn test_seconds_to_vtt_time() {
        assert_eq!(seconds_to_vtt_time(0.0), "00:00:00.000");
        assert_eq!(seconds_to_vtt_time(61.5), "00:01:01.500");
        assert_eq!(seconds_to_vtt_time(3661.123), "01:01:01.123");
    }
}
