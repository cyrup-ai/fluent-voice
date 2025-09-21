//! Stream Implementation for STT Conversation
//!
//! Contains the main processing loop for audio stream handling including
//! wake word detection, VAD processing, and transcription.

use fluent_voice_domain::{TranscriptionSegment, TranscriptionSegmentImpl, VoiceError};

/// Core audio processing loop implementation
pub async fn process_audio_loop(
    mut conversation: super::conversation::DefaultSTTConversation,
) -> impl futures_core::Stream<Item = Result<TranscriptionSegmentImpl, VoiceError>> {
    use async_stream::stream;

    stream! {
        // Stream state variables
        let mut wake_word_detected = false;
        let mut audio_buffer = Vec::with_capacity(32000); // 2 seconds at 16kHz
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

            // Step 1: Wake word detection (always active until detected)
            if !wake_word_detected {
                let detection_result = conversation.audio_processor.process_audio_chunk(&audio_chunk);

                if let Some(detection) = detection_result {
                    if detection.score > wake_threshold {
                        wake_word_detected = true;

                        // Call the wake_handler with the actual koffee detection
                        if let Some(ref mut handler) = conversation.wake_handler {
                            handler.0(detection.name.clone());
                        }

                        let segment = TranscriptionSegmentImpl::new(
                            format!("[WAKE WORD: {}]", detection.name),
                            0,
                            500,
                            None,
                        );

                        // CRITICAL: Use the chunk processor to transform the segment
                        let segment_impl = TranscriptionSegmentImpl::new(
                            segment.text().to_string(),
                            segment.start_ms(),
                            segment.end_ms(),
                            segment.speaker_id().map(|s| s.to_string()),
                        );
                        let processed_segment = if let Some(ref mut processor) = conversation.chunk_processor {
                            processor.0(Ok(segment_impl))
                        } else {
                            segment_impl  // Fallback if no processor
                        };

                        // Yield the processed TranscriptionSegmentImpl directly
                        yield Ok(processed_segment);
                        continue;
                    }
                }
                continue;
            }

            // Step 2: VAD processing (only after wake word)
            audio_buffer.extend_from_slice(&audio_chunk);

            // Process in chunks
            if audio_buffer.len() >= conversation.audio_processor.frame_size { // 100ms at 16kHz
                let chunk_to_process = audio_buffer.drain(..conversation.audio_processor.frame_size).collect::<Vec<_>>();

                // Voice Activity Detection using AudioProcessor
                let speech_probability = conversation.audio_processor.process_vad(&chunk_to_process);

                let speech_probability = match speech_probability {
                    Ok(prob) => prob,
                    Err(e) => {
                        let error = VoiceError::ProcessingError(format!("VAD error: {}", e));

                        // Call the error_handler with the actual error
                        if let Some(ref mut handler) = conversation.error_handler {
                            let _error_message = handler.0(error.clone());
                            // Error message logged in handler, not used further here
                        }

                        yield Err(error);
                        continue;
                    }
                };

                let is_speech = speech_probability > conversation.vad_config.sensitivity; // Use configured VAD sensitivity

                if is_speech {
                    last_speech_time = std::time::Instant::now(); // Update last speech time
                    // Step 3: Whisper transcription on speech segments
                    // Use VAD config for minimum speech duration (converted from ms to samples at 16kHz)
                    let min_speech_samples = (conversation.vad_config.min_speech_duration * 16) as usize;
                    if audio_buffer.len() >= min_speech_samples {
                        let speech_data = audio_buffer.clone();

                        // Use AudioProcessor for in-memory transcription (no temp files needed)
                        let transcription_result = {
                            // Use AudioProcessor for in-memory transcription (no temp files)
                            conversation.audio_processor.transcribe_audio(&speech_data).await.map_err(|e| {
                                VoiceError::ProcessingError(format!("AudioProcessor transcription failed: {}", e))
                            }).map(|text| {
                                // Create transcript compatible with existing code
                                struct InMemoryTranscript { text: String }
                                impl InMemoryTranscript {
                                    fn as_text(&self) -> &str { &self.text }
                                }
                                InMemoryTranscript { text }
                            })
                        };

                        match transcription_result {
                            Ok(transcript) => {
                                let transcription = transcript.as_text();
                                if !transcription.trim().is_empty() {
                                    let end_ms = speech_start_time.elapsed().as_millis() as u32;
                                    let start_ms = end_ms.saturating_sub(500);

                                    let segment = TranscriptionSegmentImpl::new(
                                        transcription.to_string(),
                                        start_ms,
                                        end_ms,
                                        None,
                                    );

                                    // Process prediction if callback is configured
                                    if let Some(ref mut processor) = conversation.prediction_processor {
                                        // Call prediction processor with raw and processed transcript
                                        processor.0(transcription.to_string(), transcription.to_string());
                                    }

                                    // CRITICAL: Use the chunk processor to transform the segment
                                    let segment_impl = TranscriptionSegmentImpl::new(
                                        segment.text().to_string(),
                                        segment.start_ms(),
                                        segment.end_ms(),
                                        segment.speaker_id().map(|s| s.to_string()),
                                    );

                                    if let Some(ref mut processor) = conversation.chunk_processor {
                                        let processed_segment = processor.0(Ok(segment_impl.clone()));
                                        yield Ok(processed_segment);
                                    } else {
                                        yield Ok(segment_impl);
                                    }
                                }
                            },
                            Err(e) => {
                                let error = VoiceError::ProcessingError(format!("Transcription failed: {}", e));

                                // CRITICAL: Use the chunk processor to handle the error
                                if let Some(ref mut processor) = conversation.chunk_processor {
                                    let processed_segment = processor.0(Err(error));
                                    // Yield the processed TranscriptionSegmentImpl directly
                                    yield Ok(processed_segment);
                                } else {
                                    yield Err(error);  // Fallback if no processor
                                }
                            }
                        }

                        // Clear buffer after transcription
                        audio_buffer.clear();
                    }
                } else if !audio_buffer.is_empty() {
                    // End of speech - process accumulated audio
                    let speech_data = audio_buffer.clone();
                    if speech_data.len() >= 3200 { // At least 200ms of speech

                        // Call the turn_handler with the actual VAD detection
                        if let Some(ref mut handler) = conversation.turn_handler {
                            handler.0(None, "Speech turn detected".to_string());
                        }

                        // Final transcription of remaining speech
                        // Check minimum speech duration based on VAD config
                        let min_speech_samples = (conversation.vad_config.min_speech_duration * 16) as usize;
                        if speech_data.len() < min_speech_samples {
                            // Too short, skip transcription
                            audio_buffer.clear();
                            wake_word_detected = false;
                            continue;
                        }

                        let transcription_result = {
                            // Use AudioProcessor for in-memory transcription (no temp files)
                            conversation.audio_processor.transcribe_audio(&speech_data).await.map_err(|e| {
                                VoiceError::ProcessingError(format!("AudioProcessor transcription failed: {}", e))
                            }).map(|text| {
                                // Create transcript compatible with existing code
                                struct InMemoryTranscript { text: String }
                                impl InMemoryTranscript {
                                    fn as_text(&self) -> &str { &self.text }
                                }
                                InMemoryTranscript { text }
                            })
                        };

                        match transcription_result {
                            Ok(transcript) => {
                                let transcription = transcript.as_text();
                                if !transcription.trim().is_empty() {
                                    let end_ms = speech_start_time.elapsed().as_millis() as u32;
                                    let start_ms = end_ms.saturating_sub((speech_data.len() as u32 * 1000) / 16000);

                                    let segment = TranscriptionSegmentImpl::new(
                                        transcription.to_string(),
                                        start_ms,
                                        end_ms,
                                        None,
                                    );

                                    // Process prediction if callback is configured
                                    if let Some(ref mut processor) = conversation.prediction_processor {
                                        // Call prediction processor with raw and processed transcript
                                        processor.0(transcription.to_string(), transcription.to_string());
                                    }

                                    // CRITICAL: Use the chunk processor to transform the segment
                                    let segment_impl = TranscriptionSegmentImpl::new(
                                        segment.text().to_string(),
                                        segment.start_ms(),
                                        segment.end_ms(),
                                        segment.speaker_id().map(|s| s.to_string()),
                                    );

                                    if let Some(ref mut processor) = conversation.chunk_processor {
                                        let processed_segment = processor.0(Ok(segment_impl.clone()));
                                        yield Ok(processed_segment);
                                    } else {
                                        yield Ok(segment_impl);
                                    }
                                }
                            },
                            Err(e) => {
                                let error_msg = format!("Final transcription failed: {}", e);
                                yield Err(VoiceError::ProcessingError(error_msg));
                            }
                        }
                    }

                    // Reset for next utterance
                    audio_buffer.clear();
                    wake_word_detected = false;
                }

                // Timeout reset using VAD config max_silence_duration
                let silence_duration = last_speech_time.elapsed().as_millis() as u32;
                if wake_word_detected && silence_duration > conversation.vad_config.max_silence_duration {
                    wake_word_detected = false;
                    audio_buffer.clear();
                    tracing::debug!("Reset conversation after {}ms of silence (max: {}ms)",
                                  silence_duration, conversation.vad_config.max_silence_duration);
                }
            }
        }
    }
}
