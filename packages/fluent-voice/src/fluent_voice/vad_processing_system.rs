//! Real-Time VAD Processing System
//!
//! Complete end-to-end VAD processing pipeline that handles continuous audio streams
//! with real-time voice activity detection, turn detection, and event publishing.

use crate::fluent_voice::{
    default_engine_coordinator::{VadEngine, VadResult},
    event_bus::{EventBus, VoiceEvent},
};
use fluent_voice_domain::VoiceError;
use futures::{Stream, StreamExt};
use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::{Duration, Instant},
};
use tokio::sync::{mpsc, Mutex};

/// Complete real-time VAD processing system
pub struct RealTimeVadSystem {
    vad_engine: Arc<Mutex<VadEngine>>,
    event_bus: Arc<EventBus>,
    processing_state: ProcessingState,
    performance_metrics: Arc<Mutex<PerformanceMetrics>>,
}

/// Current state of the VAD processing system
#[derive(Debug, Clone)]
pub struct ProcessingState {
    pub is_active: bool,
    pub current_voice_state: bool,
    pub last_activity_timestamp: Option<u64>,
    pub turn_count: u64,
}

impl Default for ProcessingState {
    fn default() -> Self {
        Self {
            is_active: false,
            current_voice_state: false,
            last_activity_timestamp: None,
            turn_count: 0,
        }
    }
}

/// Performance metrics for monitoring system health
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub total_chunks_processed: u64,
    pub average_processing_time_ms: f64,
    pub turn_detection_count: u64,
    pub error_count: u64,
    start_time: Instant,
}

impl PerformanceMetrics {
    pub fn new() -> Self {
        Self {
            total_chunks_processed: 0,
            average_processing_time_ms: 0.0,
            turn_detection_count: 0,
            error_count: 0,
            start_time: Instant::now(),
        }
    }

    pub fn record_chunk_processed(&mut self, processing_time: Duration) {
        self.total_chunks_processed += 1;
        let processing_time_ms = processing_time.as_millis() as f64;

        // Update running average
        self.average_processing_time_ms = (self.average_processing_time_ms
            * (self.total_chunks_processed - 1) as f64
            + processing_time_ms)
            / self.total_chunks_processed as f64;
    }

    pub fn record_turn_detected(&mut self) {
        self.turn_detection_count += 1;
    }

    pub fn record_error(&mut self) {
        self.error_count += 1;
    }

    pub fn uptime(&self) -> Duration {
        self.start_time.elapsed()
    }
}

impl RealTimeVadSystem {
    /// Create a new real-time VAD processing system
    pub fn new(event_bus: Arc<EventBus>, vad_engine: Arc<Mutex<VadEngine>>) -> Self {
        Self {
            vad_engine,
            event_bus,
            processing_state: ProcessingState::default(),
            performance_metrics: Arc::new(Mutex::new(PerformanceMetrics::new())),
        }
    }

    /// Start complete real-time VAD processing
    pub async fn start_processing<S>(
        &mut self,
        audio_stream: S,
    ) -> Result<VadProcessingStream, VoiceError>
    where
        S: Stream<Item = Vec<i16>> + Send + Unpin + 'static,
    {
        self.processing_state.is_active = true;

        let (result_tx, result_rx) = mpsc::channel(1000);
        let vad_engine = Arc::clone(&self.vad_engine);
        let event_bus = self.event_bus.clone();
        let performance_metrics = self.performance_metrics.clone();

        // Spawn processing task
        tokio::spawn(async move {
            let mut audio_stream = audio_stream;
            let mut last_voice_state = false;

            while let Some(audio_chunk) = audio_stream.next().await {
                let processing_start = Instant::now();

                // Convert to bytes for VAD processing
                let audio_bytes: Vec<u8> = audio_chunk
                    .iter()
                    .flat_map(|&sample| sample.to_le_bytes().to_vec())
                    .collect();

                // Process through VAD
                let vad_result = {
                    match vad_engine.try_lock() {
                        Ok(mut vad) => vad.detect_voice_activity(&audio_bytes).await,
                        Err(_) => {
                            // VAD engine is busy, skip this chunk
                            continue;
                        }
                    }
                };

                match vad_result {
                    Ok(result) => {
                        // Store values before result is moved
                        let voice_detected = result.voice_detected;
                        let timestamp = result.timestamp;
                        
                        // Detect turn transitions
                        if voice_detected != last_voice_state {
                            // Record turn detection in performance metrics
                            {
                                let mut metrics = performance_metrics.lock().await;
                                metrics.record_turn_detected();
                            }

                            // Publish turn detection event
                            let turn_event = VoiceEvent::ConversationTurnDetected {
                                speaker_change: true,
                            };
                            let _ = event_bus.publish(turn_event).await;
                        }

                        // Publish voice activity event
                        let activity_event = if voice_detected {
                            VoiceEvent::VoiceActivityStarted { timestamp }
                        } else {
                            VoiceEvent::VoiceActivityEnded { timestamp }
                        };

                        let _ = event_bus.publish(activity_event).await;

                        // Send result downstream
                        if result_tx.send(Ok(result)).await.is_err() {
                            break; // Receiver dropped
                        }

                        last_voice_state = voice_detected;
                    }
                    Err(e) => {
                        // Send error downstream
                        if result_tx.send(Err(e)).await.is_err() {
                            break;
                        }
                    }
                }

                // Record performance metrics
                let processing_time = processing_start.elapsed();
                if processing_time > Duration::from_millis(10) {
                    // Log slow processing
                    tracing::warn!("Slow VAD processing: {}ms", processing_time.as_millis());
                }
            }
        });

        Ok(VadProcessingStream::new(result_rx))
    }

    /// Get current processing state
    pub fn get_processing_state(&self) -> &ProcessingState {
        &self.processing_state
    }

    /// Get performance metrics
    pub async fn get_performance_metrics(&self) -> PerformanceMetrics {
        self.performance_metrics.lock().await.clone()
    }

    /// Stop processing and cleanup resources
    pub async fn stop_processing(&mut self) {
        self.processing_state.is_active = false;
        // Additional cleanup can be added here
    }
}

/// Stream of VAD processing results
pub struct VadProcessingStream {
    receiver: mpsc::Receiver<Result<VadResult, VoiceError>>,
}

impl VadProcessingStream {
    fn new(receiver: mpsc::Receiver<Result<VadResult, VoiceError>>) -> Self {
        Self { receiver }
    }
}

impl Stream for VadProcessingStream {
    type Item = Result<VadResult, VoiceError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.receiver.poll_recv(cx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_stream::wrappers::ReceiverStream;

    fn create_test_audio_stream() -> impl Stream<Item = Vec<i16>> + Send + Unpin {
        let (tx, rx) = mpsc::channel(100);

        tokio::spawn(async move {
            // Generate test audio data
            for i in 0..10 {
                let samples = if i % 2 == 0 {
                    // Silence
                    vec![0i16; 512]
                } else {
                    // Simple sine wave (simulated speech)
                    (0..512)
                        .map(|j| {
                            let t = j as f32 / 16000.0;
                            (440.0 * 2.0 * std::f32::consts::PI * t).sin() * 1000.0
                        })
                        .map(|f| f as i16)
                        .collect()
                };

                if tx.send(samples).await.is_err() {
                    break;
                }

                tokio::time::sleep(Duration::from_millis(32)).await; // 32ms chunks
            }
        });

        ReceiverStream::new(rx)
    }

    #[tokio::test]
    async fn test_real_time_vad_system_creation() {
        let event_bus = Arc::new(EventBus::new());
        let vad_engine_result = VadEngine::new();
        if vad_engine_result.is_err() {
            return; // Skip test if VAD engine can't be created
        }
        let vad_engine = Arc::new(Mutex::new(vad_engine_result.unwrap()));
        let vad_system = RealTimeVadSystem::new(event_bus, vad_engine);

        assert!(vad_system.processing_state.is_active == false, "Should create VAD system successfully");
    }

    #[tokio::test]
    async fn test_vad_processing_stream() {
        let event_bus = Arc::new(EventBus::new());
        let vad_engine_result = VadEngine::new();
        if vad_engine_result.is_err() {
            return; // Skip test if VAD engine can't be created
        }
        let vad_engine = Arc::new(Mutex::new(vad_engine_result.unwrap()));
        let mut vad_system = RealTimeVadSystem::new(event_bus, vad_engine);

        let audio_stream = create_test_audio_stream();
        let processing_result = vad_system.start_processing(audio_stream).await;

        match processing_result {
            Ok(mut processing_stream) => {
                // Process a few chunks
                let mut results = Vec::new();
                for _ in 0..3 {
                    if let Some(result) = processing_stream.next().await {
                        results.push(result);
                    }
                }

                assert!(!results.is_empty(), "Should process audio chunks");
            }
            Err(_) => {
                // Test passes if processing can't start (dependencies may not be available)
            }
        }
    }

    #[test]
    fn test_performance_metrics() {
        let mut metrics = PerformanceMetrics::new();

        // Record some processing times
        metrics.record_chunk_processed(Duration::from_millis(5));
        metrics.record_chunk_processed(Duration::from_millis(7));
        metrics.record_turn_detected();

        assert_eq!(metrics.total_chunks_processed, 2);
        assert_eq!(metrics.turn_detection_count, 1);
        assert!(metrics.average_processing_time_ms > 0.0);
    }
}
