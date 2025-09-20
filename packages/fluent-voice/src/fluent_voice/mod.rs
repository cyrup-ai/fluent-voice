//! Unified entry point for Text-to-Speech and Speech-to-Text operations.

pub mod default_implementation;
pub mod default_tts_builder;
pub mod default_tts_conversation;
pub mod entry_points;
pub mod extension_traits;
pub mod fluent_voice_trait;
pub mod session_tracking;
pub mod synthesis_parameters;

// Engine coordination modules
pub mod conversation_manager;
pub mod coordinated_voice_stream;
pub mod default_engine_coordinator;
pub mod default_engine_provider;
pub mod event_bus;

// VAD integration systems
pub mod coordinated_vad_system;
pub mod vad_conversation_system;
pub mod vad_processing_system;

// Re-export main types for backward compatibility
pub use default_implementation::FluentVoiceImpl;
pub use default_tts_builder::DefaultTtsBuilder;
pub use default_tts_conversation::DefaultTtsConversation;
pub use entry_points::{SttEntry, SttPostChunkEntry, TtsEntry};
pub use fluent_voice_trait::FluentVoice;
pub use synthesis_parameters::{SessionStatus, SynthesisParameters, SynthesisSession};

// Re-export coordination types for public API
pub use conversation_manager::{
    ConversationManager, ConversationResult, ConversationState, ConversationStream,
    ConversationTurn, TurnDetectionConfig, TurnDetectionEngine,
};
pub use coordinated_voice_stream::{CoordinatedVoiceStream, PipelineResult, PipelineState};
pub use default_engine_coordinator::{CoordinationState, DefaultEngineCoordinator, ProcessingMode};
pub use default_engine_provider::{
    DefaultEngineImplementation, DefaultEngineProvider, SttEngine, TtsEngine, VadEngine,
    WakeWordEngine,
};
pub use event_bus::{EngineType, EventBus, EventType, VoiceEvent};

// Re-export VAD integration systems for public API
pub use coordinated_vad_system::{
    CoordinatedProcessingStream, CoordinatedVadSystem, CoordinationResult,
    CoordinationState as VadCoordinationState, ResourceUsage,
};
pub use vad_conversation_system::{
    ConversationStats, ConversationStream as VadConversationStream, DialogueController,
    SpeakerTracker, TurnProcessor, VadConversationSystem,
};
pub use vad_processing_system::{
    PerformanceMetrics, ProcessingState, RealTimeVadSystem, VadProcessingStream,
};
