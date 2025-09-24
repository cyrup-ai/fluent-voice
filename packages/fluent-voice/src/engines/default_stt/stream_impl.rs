//! Stream Implementation for STT Conversation
//!
//! Contains the main processing loop for audio stream handling including
//! wake word detection, VAD processing, and transcription.

use super::audio_processor::IndependentListenerResults;
use fluent_voice_domain::{TranscriptionSegmentImpl, VoiceError};

/// Core audio processing loop implementation
pub async fn process_audio_loop(
    mut conversation: super::conversation::DefaultSTTConversation,
) -> impl futures_core::Stream<Item = Result<TranscriptionSegmentImpl, VoiceError>> {
    use async_stream::stream;

    stream! {
        // Stream state variables
        let mut wake_word_detected = false;
        let mut audio_buffer: Vec<f32> = Vec::with_capacity(32000); // 2 seconds at 16kHz
        let speech_start_time = std::time::Instant::now();
        let mut last_speech_time = std::time::Instant::now();
        let wake_threshold = conversation.wake_word_config.sensitivity;

        // Main processing loop
        loop {
            // Check if stream manager is still active
            if !conversation.stream_manager.is_active() {
                tracing::warn!("Stream manager is no longer active, ending conversation");
                break;
            }

            // Read from crossbeam channel
            let audio_chunk = match conversation.audio_receiver.try_recv() {
                Ok(chunk) => chunk,
                Err(crossbeam_channel::TryRecvError::Empty) => {
                    // No data available, sleep briefly and continue
                    tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                    continue;
                }
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    // Channel disconnected, audio stream manager has stopped
                    tracing::info!("Audio channel disconnected, ending conversation");
                    break;
                }
            };

            if audio_chunk.is_empty() {
                continue;
            }

            // ENHANCED: Process all independent listeners simultaneously
            let listener_results = conversation.audio_processor.process_independent_listeners(&audio_chunk);

            // Coordinate callback dispatch based on listener results
            match listener_results {
                IndependentListenerResults { wake_detection: Some(detection), .. } if detection.score > wake_threshold => {
                    wake_word_detected = true;

                    // Dispatch to existing wake_handler callback
                    if let Some(ref mut handler) = conversation.wake_handler {
                        handler.0(detection.name.clone());
                    }

                    // Create wake word segment using existing chunk_processor callback
                    let wake_segment = TranscriptionSegmentImpl::new(
                        format!("[WAKE WORD: {}]", detection.name),
                        0, 500, None
                    );

                    if let Some(ref mut processor) = conversation.chunk_processor {
                        let processed_segment = processor.0(Ok(wake_segment));
                        yield Ok(processed_segment);
                    }
                },

                IndependentListenerResults { vad_probability, .. } if wake_word_detected && vad_probability > conversation.vad_config.sensitivity => {
                    last_speech_time = std::time::Instant::now();

                    // Dispatch to existing turn_handler callback
                    if let Some(ref mut handler) = conversation.turn_handler {
                        handler.0(None, "Speech detected".to_string());
                    }
                },

                IndependentListenerResults { transcription_context: Some(context), .. } if wake_word_detected => {
                    // Transcription listener ready - process with pre-converted audio data
                    let transcription_result = conversation.audio_processor.transcribe_with_context(context).await;

                    match transcription_result {
                        Ok(text) if !text.trim().is_empty() => {
                            let end_ms = speech_start_time.elapsed().as_millis() as u32;
                            let start_ms = end_ms.saturating_sub(500);
                            let segment = TranscriptionSegmentImpl::new(text.clone(), start_ms, end_ms, None);

                            // Process prediction if callback is configured
                            if let Some(ref mut processor) = conversation.prediction_processor {
                                processor.0(text.clone(), text);
                            }

                            // Dispatch to existing chunk_processor callback
                            if let Some(ref mut processor) = conversation.chunk_processor {
                                let processed_segment = processor.0(Ok(segment));
                                yield Ok(processed_segment);
                            }
                        },
                        Err(e) => {
                            // Dispatch error to existing chunk_processor callback
                            if let Some(ref mut processor) = conversation.chunk_processor {
                                let error_segment = processor.0(Err(e));
                                yield Ok(error_segment);
                            }
                        },
                        _ => {
                            // Empty transcription - continue monitoring
                        }
                    }
                },

                _ => {
                    // No listeners fired or not in active state - continue monitoring
                }
            }

            // Timeout reset logic (keep existing)
            let silence_duration = last_speech_time.elapsed().as_millis() as u32;
            if wake_word_detected && silence_duration > conversation.vad_config.max_silence_duration {
                wake_word_detected = false;
                audio_buffer.clear();
                tracing::debug!("Reset conversation after {}ms of silence", silence_duration);
            }
        }
    }
}
