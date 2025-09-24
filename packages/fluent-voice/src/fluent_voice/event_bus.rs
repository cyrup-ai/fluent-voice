//! Event bus for cross-engine communication in the default engine coordination system.

use fluent_voice_domain::VoiceError;
use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};

/// Type alias for event handler function
pub type EventHandler = Box<
    dyn Fn(VoiceEvent) -> Pin<Box<dyn Future<Output = Result<(), VoiceError>> + Send + 'static>>
        + Send
        + Sync
        + 'static,
>;

/// Type alias for boxed future returned by event handlers
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Event bus for cross-engine communication
pub struct EventBus {
    subscribers: Arc<RwLock<HashMap<EventType, Vec<EventHandler>>>>,
    event_queue: Arc<Mutex<VecDeque<VoiceEvent>>>,
}

/// Types of events that can be published on the event bus
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventType {
    WakeWordDetected,
    VoiceActivityStarted,
    VoiceActivityEnded,
    SpeechTranscribed,
    SynthesisStarted,
    SynthesisCompleted,
    ConversationTurnDetected,
    ErrorOccurred,
}

/// Events that can be published and subscribed to on the event bus
#[derive(Debug, Clone)]
pub enum VoiceEvent {
    WakeWordDetected {
        confidence: f32,
        timestamp: u64,
    },
    VoiceActivityStarted {
        timestamp: u64,
    },
    VoiceActivityEnded {
        timestamp: u64,
    },
    SpeechTranscribed {
        text: String,
        confidence: f32,
        timestamp: u64,
    },
    SynthesisStarted {
        text: String,
        speaker_id: String,
    },
    SynthesisCompleted {
        audio_data: Vec<u8>,
        duration_ms: u64,
    },
    ConversationTurnDetected {
        speaker_change: bool,
    },
    ErrorOccurred {
        engine: EngineType,
        error: String,
    },
}

/// Types of engines that can report errors
#[derive(Debug, Clone)]
pub enum EngineType {
    Tts,
    Stt,
    Vad,
    WakeWord,
}

impl VoiceEvent {
    /// Get the event type for this event
    pub fn event_type(&self) -> EventType {
        match self {
            VoiceEvent::WakeWordDetected { .. } => EventType::WakeWordDetected,
            VoiceEvent::VoiceActivityStarted { .. } => EventType::VoiceActivityStarted,
            VoiceEvent::VoiceActivityEnded { .. } => EventType::VoiceActivityEnded,
            VoiceEvent::SpeechTranscribed { .. } => EventType::SpeechTranscribed,
            VoiceEvent::SynthesisStarted { .. } => EventType::SynthesisStarted,
            VoiceEvent::SynthesisCompleted { .. } => EventType::SynthesisCompleted,
            VoiceEvent::ConversationTurnDetected { .. } => EventType::ConversationTurnDetected,
            VoiceEvent::ErrorOccurred { .. } => EventType::ErrorOccurred,
        }
    }
}

impl EventBus {
    /// Create a new event bus
    pub fn new() -> Self {
        Self {
            subscribers: Arc::new(RwLock::new(HashMap::new())),
            event_queue: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    /// Publish an event to all subscribers
    pub async fn publish(&self, event: VoiceEvent) -> Result<(), VoiceError> {
        // Add event to queue for processing
        {
            let mut queue = self.event_queue.lock().await;
            queue.push_back(event.clone());
        }

        // Publish event to all subscribers
        let subscribers = self.subscribers.read().await;
        if let Some(handlers) = subscribers.get(&event.event_type()) {
            for handler in handlers {
                if let Err(e) = handler(event.clone()).await {
                    // Log error but continue processing other handlers
                    tracing::error!("Event handler failed: {}", e);
                }
            }
        }
        Ok(())
    }

    /// Subscribe to events of a specific type with a handler function
    pub async fn subscribe<F>(&self, event_type: EventType, handler: F)
    where
        F: Fn(VoiceEvent) -> BoxFuture<'static, Result<(), VoiceError>> + Send + Sync + 'static,
    {
        let boxed_handler: EventHandler = Box::new(move |event| Box::pin(handler(event)));

        let mut subscribers = self.subscribers.write().await;
        subscribers
            .entry(event_type)
            .or_insert_with(Vec::new)
            .push(boxed_handler);
    }

    /// Get the current number of pending events in the queue
    pub async fn pending_event_count(&self) -> usize {
        let queue = self.event_queue.lock().await;
        queue.len()
    }

    /// Clear all pending events from the queue
    pub async fn clear_event_queue(&self) -> Result<(), VoiceError> {
        let mut queue = self.event_queue.lock().await;
        queue.clear();
        Ok(())
    }

    /// Get the next event from the queue without removing it
    pub async fn peek_next_event(&self) -> Option<VoiceEvent> {
        let queue = self.event_queue.lock().await;
        queue.front().cloned()
    }

    /// Remove and return the next event from the queue
    pub async fn pop_next_event(&self) -> Option<VoiceEvent> {
        let mut queue = self.event_queue.lock().await;
        queue.pop_front()
    }

    /// Check if there are any subscribers for a given event type
    pub async fn has_subscribers(&self, event_type: EventType) -> bool {
        let subscribers = self.subscribers.read().await;
        subscribers
            .get(&event_type)
            .is_some_and(|handlers| !handlers.is_empty())
    }

    /// Get the number of subscribers for a given event type
    pub async fn subscriber_count(&self, event_type: EventType) -> usize {
        let subscribers = self.subscribers.read().await;
        subscribers
            .get(&event_type)
            .map_or(0, |handlers| handlers.len())
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[tokio::test]
    async fn test_event_bus_creation() {
        let event_bus = EventBus::new();
        assert_eq!(event_bus.pending_event_count().await, 0);
        assert!(!event_bus.has_subscribers(EventType::WakeWordDetected).await);
    }

    #[tokio::test]
    async fn test_event_publishing_and_subscription() {
        let event_bus = EventBus::new();
        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = call_count.clone();

        // Subscribe to wake word events
        event_bus
            .subscribe(EventType::WakeWordDetected, move |_event| {
                let call_count = call_count_clone.clone();
                Box::pin(async move {
                    call_count.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                })
            })
            .await;

        // Publish a wake word event
        let event = VoiceEvent::WakeWordDetected {
            confidence: 0.95,
            timestamp: 12345,
        };

        event_bus
            .publish(event)
            .await
            .expect("Failed to publish event");

        // Small delay to allow async handler to execute
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        assert_eq!(call_count.load(Ordering::SeqCst), 1);
        assert_eq!(
            event_bus
                .subscriber_count(EventType::WakeWordDetected)
                .await,
            1
        );
    }

    #[tokio::test]
    async fn test_event_queue_operations() {
        let event_bus = EventBus::new();

        let event1 = VoiceEvent::VoiceActivityStarted { timestamp: 100 };
        let event2 = VoiceEvent::VoiceActivityEnded { timestamp: 200 };

        event_bus
            .publish(event1.clone())
            .await
            .expect("Failed to publish event1");
        event_bus
            .publish(event2.clone())
            .await
            .expect("Failed to publish event2");

        assert_eq!(event_bus.pending_event_count().await, 2);

        let peeked = event_bus.peek_next_event().await;
        assert!(peeked.is_some());
        assert_eq!(event_bus.pending_event_count().await, 2); // Should still be 2

        let popped = event_bus.pop_next_event().await;
        assert!(popped.is_some());
        assert_eq!(event_bus.pending_event_count().await, 1); // Should now be 1

        event_bus
            .clear_event_queue()
            .await
            .expect("Failed to clear queue");
        assert_eq!(event_bus.pending_event_count().await, 0);
    }
}
