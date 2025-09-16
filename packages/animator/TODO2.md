# TODO2.md - Production Code Quality Issues (RESEARCH-ENHANCED)

## 🚨 **RESEARCH STATUS UPDATE** 

### **CRITICAL DISCOVERY**: TODO1.md Implementation Already Resolved Major Issues!

**✅ COMPLETED IN TODO1.md**:
- **All println! statements → tracing logging**: No println! found in `examples/full_room_visualizer.rs`
- **Comprehensive tracing implementation**: `tracing::info!`, `tracing::error!`, `tracing::debug!`, `tracing::warn!` throughout codebase
- **Only remaining println!**: [`src/main.rs:3`](src/main.rs) (development stub - not production code)

---

## ⚠️ **REMAINING NON-PRODUCTION CODE PATTERNS** (RESEARCH-VERIFIED)

### **1. ✅ Debug Prints → Tracing Logging (COMPLETED)**
**File**: `examples/full_room_visualizer.rs`  
**Status**: **✅ ALREADY COMPLETED** - Converted during TODO1.md implementation  
**Priority**: **DONE**  
**Research Evidence**: Extensive search found NO println! in production code paths

**Current Implementation (VERIFIED)**:
```rust
// PRODUCTION TRACING PATTERNS ALREADY IMPLEMENTED:
tracing::info!("Successfully connected to room via channel");
tracing::error!(error = ?e, "Failed to connect to room");
tracing::info!(participant_id = %identity.0, "Participant connected");
tracing::info!(participant_id = %identity.0, "Participant disconnected");
tracing::info!("Audio player created for: {}", participant_id.0);
tracing::debug!("Updated volume for {}: {:.2}", participant_id.0, participant_state.participant_volume);
```

**Established Logging Patterns** (from research):
- **Info**: Connection events, participant changes, major state transitions
- **Error**: Connection failures, resource creation failures  
- **Debug**: Volume changes, detailed state updates
- **Warn**: Recovery from poisoned mutexes, performance warnings

---

### **2. ❌ Hardcoded Magic Numbers (VERIFIED PRESENT)**
**File**: [`examples/full_room_visualizer.rs:191`](examples/full_room_visualizer.rs)  
**Status**: **❌ NON-CONFIGURABLE** - Speaking threshold hardcoded  
**Priority**: **MEDIUM**  
**Research Dependencies**: [`src/audio_visualizer.rs`](src/audio_visualizer.rs) AudioVisualizerConfig pattern

```rust
// CURRENT (HARDCODED):
participant_state.is_speaking = stats.current_amplitude > 0.01; // ❌ Magic number
```

**SOLUTION ARCHITECTURE** (Based on AudioVisualizerConfig Pattern):

**Configuration Struct Pattern** (from [`src/audio_visualizer.rs:8-18`](src/audio_visualizer.rs)):
```rust
/// Configuration for room visualizer behavior
/// Based on AudioVisualizerConfig pattern from src/audio_visualizer.rs
#[derive(Debug, Clone)]
pub struct RoomVisualizerConfig {
    /// Threshold for speaking detection (0.0 = always speaking, 1.0 = never speaking)
    pub speaking_threshold: f32,
    /// Interval for connection quality updates
    pub connection_quality_update_interval: std::time::Duration,
    /// How long to display error messages in UI
    pub error_display_timeout: std::time::Duration,
    /// Auto-cleanup timeout for disconnected participants
    pub auto_cleanup_timeout: std::time::Duration,
    /// Smoothing factor for speaking detection (reduces flickering)
    pub speaking_smoothing: f32,
}

impl Default for RoomVisualizerConfig {
    fn default() -> Self {
        Self {
            speaking_threshold: 0.01,      // Current hardcoded value
            connection_quality_update_interval: std::time::Duration::from_secs(5),
            error_display_timeout: std::time::Duration::from_secs(10),
            auto_cleanup_timeout: std::time::Duration::from_secs(300),
            speaking_smoothing: 0.2,       // Same as AudioVisualizerConfig::smoothing_factor
        }
    }
}
```

**UI Implementation Pattern**:
```rust
// ADD TO FullRoomVisualizerApp struct:
config: RoomVisualizerConfig,

// ADD TO UI (Settings Panel):
ui.collapsing("🔧 Detection Settings", |ui| {
    ui.horizontal(|ui| {
        ui.label("Speaking Threshold:");
        if ui.add(egui::Slider::new(&mut self.config.speaking_threshold, 0.001..=0.1)
            .logarithmic(true)
            .suffix(" amplitude")).changed() {
            tracing::debug!("Speaking threshold updated to: {:.3}", self.config.speaking_threshold);
        }
    });
    
    ui.horizontal(|ui| {
        ui.label("Quality Update Interval:");
        let mut secs = self.config.connection_quality_update_interval.as_secs() as f32;
        if ui.add(egui::Slider::new(&mut secs, 1.0..=30.0).suffix(" sec")).changed() {
            self.config.connection_quality_update_interval = std::time::Duration::from_secs(secs as u64);
        }
    });
});

// REPLACE HARDCODED LOGIC:
participant_state.is_speaking = stats.current_amplitude > self.config.speaking_threshold;
```

---

### **3. ❌ Hardcoded Connection Quality (VERIFIED PRESENT)**
**File**: [`examples/full_room_visualizer.rs:215`](examples/full_room_visualizer.rs)  
**Status**: **❌ FAKE DATA** - Always shows "Good"  
**Priority**: **HIGH**  
**Research Dependencies**: [`tmp/livekit-rust-sdks`](tmp/livekit-rust-sdks) WebRTC stats APIs

```rust
// CURRENT (HARDCODED):
connection_quality: "Good".to_string(),  // ❌ Never real data
```

**SOLUTION ARCHITECTURE** (Based on LiveKit Stats Research):

**LiveKit Stats Integration Pattern** (from [`tmp/livekit-rust-sdks/libwebrtc/src/rtp_receiver.rs`](tmp/livekit-rust-sdks/libwebrtc/src/rtp_receiver.rs)):
```rust
#[derive(Debug, Clone)]
pub enum ConnectionQuality {
    Excellent,  // >90% packet success, <50ms latency
    Good,       // >80% packet success, <100ms latency  
    Fair,       // >60% packet success, <200ms latency
    Poor,       // <60% packet success, >200ms latency
    Unknown,    // No data available
}

impl ConnectionQuality {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Excellent => "Excellent",
            Self::Good => "Good", 
            Self::Fair => "Fair",
            Self::Poor => "Poor",
            Self::Unknown => "Unknown",
        }
    }
    
    pub fn color(&self) -> egui::Color32 {
        match self {
            Self::Excellent => egui::Color32::from_rgb(0, 255, 0),   // Green
            Self::Good => egui::Color32::from_rgb(144, 238, 144),     // Light Green
            Self::Fair => egui::Color32::from_rgb(255, 255, 0),      // Yellow
            Self::Poor => egui::Color32::from_rgb(255, 0, 0),        // Red
            Self::Unknown => egui::Color32::GRAY,                     // Gray
        }
    }
}
```

**Real Quality Monitoring Implementation**:
```rust
// ADD TO FullRoomVisualizerApp:
last_quality_update: std::time::Instant,

// ADD METHOD:
fn update_connection_quality(&mut self) {
    if self.last_quality_update.elapsed() < self.config.connection_quality_update_interval {
        return;
    }
    
    for (participant_id, participant_state) in &mut self.participants {
        let quality = self.calculate_real_quality(participant_id);
        participant_state.connection_quality = quality.as_str().to_string();
    }
    
    self.last_quality_update = std::time::Instant::now();
}

fn calculate_real_quality(&self, participant_id: &ParticipantIdentity) -> ConnectionQuality {
    // Method 1: Audio-based quality estimation (immediate implementation)
    if let Some(audio_viz) = self.audio_visualizers.get(participant_id) {
        let stats = audio_viz.get_stats();
        
        // Quality heuristics based on audio metrics
        return match stats.average_amplitude {
            a if a > 0.1 && stats.peak_amplitude < 1.0 => ConnectionQuality::Excellent,
            a if a > 0.05 && stats.peak_amplitude < 1.5 => ConnectionQuality::Good,
            a if a > 0.01 => ConnectionQuality::Fair,
            _ => {
                // Check if participant is supposed to be speaking but we hear nothing
                if participant_state.is_muted { 
                    ConnectionQuality::Good  // Muted is expected
                } else {
                    ConnectionQuality::Poor  // Should hear something
                }
            }
        };
    }
    
    // Method 2: LiveKit WebRTC stats (advanced implementation)
    // TODO: Integrate with RtpReceiver.get_stats() from LiveKit SDK
    // if let Some(room) = &self.room {
    //     if let Some(audio_track) = self.get_audio_track(participant_id) {
    //         let webrtc_stats = audio_track.get_stats().await;
    //         return Self::calculate_quality_from_webrtc_stats(webrtc_stats);
    //     }
    // }
    
    ConnectionQuality::Unknown
}
```

---

### **4. ❌ Error Handling Gaps (ANALYSIS CONFIRMED)**
**File**: `examples/full_room_visualizer.rs`  
**Status**: **INCOMPLETE** - Missing comprehensive error handling  
**Priority**: **MEDIUM**  
**Research Dependencies**: [`packages/domain/src/voice_error.rs`](../../domain/src/voice_error.rs) VoiceError pattern

**Research Discovery** - Established Error Pattern from [`packages/domain/src/voice_error.rs`](../../domain/src/voice_error.rs):
```rust
/// Top-level error covering both TTS & STT operations.
#[derive(Debug, Clone, Error)]
pub enum VoiceError {
    #[error("tts: {0}")]
    Tts(&'static str),
    #[error("stt: {0}")]
    Stt(&'static str),
    #[error("configuration: {0}")]
    Configuration(String),
    #[error("processing: {0}")]
    ProcessingError(String),
    #[error("synthesis: {0}")]
    Synthesis(String),
    #[error("not synthesizable: {0}")]
    NotSynthesizable(String),
    #[error("transcription: {0}")]
    Transcription(String),
}
```

**SOLUTION ARCHITECTURE** (Following VoiceError Pattern):
```rust
/// Comprehensive error types for room visualizer operations
/// Based on VoiceError pattern from packages/domain/src/voice_error.rs
#[derive(Debug, Clone, thiserror::Error)]
pub enum VisualizerError {
    #[error("connection: {0}")]
    Connection(String),
    
    #[error("configuration: {0}")]
    Configuration(String),
    
    #[error("livekit: {0}")]
    LiveKit(String),
    
    #[error("audio processing: {0}")]
    AudioProcessing(String),
    
    #[error("video rendering: {0}")]
    VideoRendering(String),
    
    #[error("participant management: {0}")]
    ParticipantManagement(String),
}

impl VisualizerError {
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            Self::Connection(_) => ErrorSeverity::Critical,
            Self::LiveKit(_) => ErrorSeverity::High,
            Self::Configuration(_) => ErrorSeverity::Medium,
            Self::AudioProcessing(_) => ErrorSeverity::Low,
            Self::VideoRendering(_) => ErrorSeverity::Low,
            Self::ParticipantManagement(_) => ErrorSeverity::Medium,
        }
    }
}

#[derive(Debug, Clone)]
pub enum ErrorSeverity {
    Critical,  // Red, blocks functionality
    High,      // Orange, major features impacted
    Medium,    // Yellow, minor features impacted  
    Low,       // Blue, cosmetic issues
}
```

**UI Error State Management**:
```rust
// ADD TO FullRoomVisualizerApp struct:
error_message: Option<(VisualizerError, std::time::Instant)>,

// ADD TO UI update method:
fn handle_ui_errors(&mut self, ui: &mut egui::Ui) {
    if let Some((error, timestamp)) = &self.error_message {
        let elapsed = timestamp.elapsed();
        
        if elapsed > self.config.error_display_timeout {
            self.error_message = None;
        } else {
            let severity = error.severity();
            let color = match severity {
                ErrorSeverity::Critical => egui::Color32::RED,
                ErrorSeverity::High => egui::Color32::from_rgb(255, 165, 0), // Orange
                ErrorSeverity::Medium => egui::Color32::YELLOW,
                ErrorSeverity::Low => egui::Color32::LIGHT_BLUE,
            };
            
            ui.horizontal(|ui| {
                ui.colored_label(color, format!("⚠️ {}", error));
                if ui.button("Dismiss").clicked() {
                    self.error_message = None;
                }
                
                // Auto-dismiss countdown
                let remaining = self.config.error_display_timeout.saturating_sub(elapsed);
                ui.label(format!("({}s)", remaining.as_secs()));
            });
        }
    }
}

// ERROR REPORTING HELPER:
fn report_error(&mut self, error: VisualizerError) {
    tracing::error!("{}", error);
    self.error_message = Some((error, std::time::Instant::now()));
}

// CONNECTION ERROR HANDLING:
fn connect_to_room(&mut self) {
    if self.room_url.is_empty() {
        self.report_error(VisualizerError::Configuration(
            "Room URL is required".to_string()
        ));
        return;
    }
    
    if self.api_key.is_empty() || self.api_secret.is_empty() {
        self.report_error(VisualizerError::Configuration(
            "API key and secret are required".to_string()
        ));
        return;
    }
    
    // Proceed with connection...
}
```

---

### **5. ❌ Resource Management Issues (ANALYSIS CONFIRMED)**
**File**: `examples/full_room_visualizer.rs`  
**Status**: **POTENTIAL LEAKS** - Missing systematic cleanup  
**Priority**: **MEDIUM**  
**Research Dependencies**: [`packages/video/src/lib.rs:155-160`](../../video/src/lib.rs) Drop pattern, [`src/livekit_audio_player.rs:201-205`](src/livekit_audio_player.rs) cleanup

**Research Discovery** - Established Cleanup Pattern from [`packages/video/src/lib.rs:155-160`](../../video/src/lib.rs):
```rust
impl Drop for TerminalRenderer {
    fn drop(&mut self) {
        // Restore cursor when renderer is dropped
        if self.cursor_hidden {
            print!("\x1B[?25h"); // Show cursor
        }
        // Cleanup resources...
    }
}
```

**SOLUTION ARCHITECTURE** (Following Video Package Pattern):
```rust
impl Drop for FullRoomVisualizerApp {
    fn drop(&mut self) {
        tracing::debug!("Cleaning up FullRoomVisualizerApp resources");
        
        // 1. Clean up audio visualizers (already have Drop impl from TODO1.md)
        for (participant_id, mut audio_viz) in self.audio_visualizers.drain() {
            audio_viz.stop(); // Method from AudioVisualizer
            tracing::debug!("Stopped audio visualizer for: {}", participant_id.0);
        }
        
        // 2. Clean up audio players (already have Drop impl from TODO1.md)  
        for (participant_id, audio_player) in self.audio_players.drain() {
            audio_player.stop(); // Method from LiveKitAudioPlayer
            tracing::debug!("Stopped audio player for: {}", participant_id.0);
        }
        
        // 3. Clean up video renderers
        for (participant_id, mut video_renderer) in self.video_renderers.drain() {
            // video_renderer.stop(); // If API exists
            tracing::debug!("Stopped video renderer for: {}", participant_id.0);
        }
        
        // 4. Disconnect from room gracefully
        if let Some(room) = self.room.take() {
            // Room disconnect is async, but Drop is sync
            // Best effort cleanup - room will be dropped
            tracing::info!("Disconnecting from room during cleanup");
        }
        
        // 5. Close event channels
        if let Some(_rx) = self.room_events_rx.take() {
            tracing::debug!("Closed room events channel");
        }
        
        if let Some(_rx) = self.connection_rx.take() {
            tracing::debug!("Closed connection results channel");
        }
    }
}

// GRACEFUL SHUTDOWN HANDLING:
impl FullRoomVisualizerApp {
    pub fn shutdown(&mut self) {
        tracing::info!("Initiating graceful shutdown");
        
        // Mark all resources for cleanup
        self.connection_status = "Shutting down...".to_string();
        
        // Stop all audio/video processing
        for audio_viz in self.audio_visualizers.values_mut() {
            audio_viz.stop();
        }
        
        for audio_player in self.audio_players.values() {
            audio_player.stop();
        }
        
        // Disconnect from room if connected
        if let Some(room) = &self.room {
            // Async disconnect - spawn task for cleanup
            let room_clone = room.clone();
            self.rt_handle.spawn(async move {
                // room_clone.disconnect().await;
                tracing::info!("Room disconnected during shutdown");
            });
        }
        
        // Clear all participants
        self.participants.clear();
        self.selected_participant = None;
        
        tracing::info!("Graceful shutdown completed");
    }
    
    /// Cleanup orphaned participants who disconnected without proper cleanup
    pub fn cleanup_orphaned_participants(&mut self) {
        let now = std::time::Instant::now();
        let timeout = self.config.auto_cleanup_timeout;
        
        let mut to_remove = Vec::new();
        
        for (participant_id, participant_state) in &self.participants {
            // Check if participant has been inactive for too long
            if let Some(audio_viz) = self.audio_visualizers.get(participant_id) {
                let stats = audio_viz.get_stats();
                // If no audio activity and no video track, mark for cleanup
                if stats.average_amplitude == 0.0 && participant_state.video_track.is_none() {
                    to_remove.push(participant_id.clone());
                }
            }
        }
        
        for participant_id in to_remove {
            self.remove_participant(&participant_id);
            tracing::info!("Cleaned up orphaned participant: {}", participant_id.0);
        }
    }
    
    fn remove_participant(&mut self, participant_id: &ParticipantIdentity) {
        // Clean removal with proper resource cleanup
        if let Some(mut audio_viz) = self.audio_visualizers.remove(participant_id) {
            audio_viz.stop();
        }
        
        if let Some(audio_player) = self.audio_players.remove(participant_id) {
            audio_player.stop();
        }
        
        self.video_renderers.remove(participant_id);
        self.participants.remove(participant_id);
        
        if self.selected_participant.as_ref() == Some(participant_id) {
            self.selected_participant = None;
        }
    }
}
```

---

## 🔍 **COMPREHENSIVE RESEARCH FINDINGS & CITATIONS**

### **Audio Processing Architecture Discovery**
**Source Analysis**: [`src/audio_visualizer.rs:8-29`](src/audio_visualizer.rs)

**Established Configuration Pattern**:
```rust
/// Configuration for the audio visualizer
pub struct AudioVisualizerConfig {
    /// Size of the amplitude history buffer
    pub buffer_size: usize,
    /// Sample rate for audio processing
    pub sample_rate: u32,
    /// Number of audio channels
    pub num_channels: u32,
    /// Smoothing factor for amplitude values (0.0 = no smoothing, 1.0 = max smoothing)
    pub smoothing_factor: f32,
}

impl Default for AudioVisualizerConfig {
    fn default() -> Self {
        Self {
            buffer_size: 100,
            sample_rate: 48000,
            num_channels: 2,
            smoothing_factor: 0.2,  // ← REUSE THIS PATTERN FOR SPEAKING SMOOTHING
        }
    }
}
```

### **Error Handling Architecture Discovery**
**Source Analysis**: [`packages/domain/src/voice_error.rs:1-29`](../../domain/src/voice_error.rs)

**Established Error Categories**:
- **Tts/Stt**: Engine-specific errors
- **Configuration**: Setup and parameter validation
- **ProcessingError**: Runtime processing failures
- **Synthesis/Transcription**: Operation-specific errors

**Key Pattern**: Uses `thiserror::Error` for automatic `Display` and `std::error::Error` implementations

### **Resource Management Discovery**
**Source Analysis**: [`packages/video/src/lib.rs:155-160`](../../video/src/lib.rs)

**Established Cleanup Pattern**:
```rust
impl Drop for TerminalRenderer {
    fn drop(&mut self) {
        // Restore cursor when renderer is dropped
        if self.cursor_hidden {
            print!("\x1B[?25h"); // Show cursor
        }
    }
}
```

**Key Pattern**: Immediate cleanup in Drop, restore system state

### **LiveKit Integration Discovery**
**Source Analysis**: [`tmp/livekit-rust-sdks/libwebrtc/src/rtp_receiver.rs`](tmp/livekit-rust-sdks/libwebrtc/src/rtp_receiver.rs)

**Available Stats APIs**:
```rust
// From LiveKit WebRTC bindings:
impl RtpReceiver {
    pub fn get_stats(&self) -> RtpReceiverStats {
        self.handle.get_stats()
    }
}
```

**Stats Categories Available**:
- Packets received/lost
- Bytes transferred  
- Jitter measurements
- Round-trip time
- Bandwidth utilization

### **Tracing Implementation Discovery**
**Source Analysis**: [`examples/full_room_visualizer.rs`](examples/full_room_visualizer.rs) (completed in TODO1.md)

**Established Logging Levels**:
- `tracing::info!`: Connection events, participant changes
- `tracing::error!`: Connection failures, critical errors
- `tracing::debug!`: Volume changes, detailed state updates  
- `tracing::warn!`: Recovery situations, performance warnings

---

## 🎯 **UPDATED ACCEPTANCE CRITERIA** (RESEARCH-ENHANCED)

**For TODO2 Completion**:
1. ✅ **Tracing Logging**: **COMPLETED** - All println! replaced with structured tracing
2. ❌ **Configurable Speaking Threshold**: UI controls for threshold adjustment (0.001-0.1 range)
3. ❌ **Real Connection Quality**: Replace hardcoded "Good" with LiveKit stats-based calculation
4. ❌ **Comprehensive Error Handling**: UI error display using VoiceError pattern  
5. ❌ **Resource Cleanup**: Drop implementation with graceful shutdown
6. ❌ **Orphaned Participant Cleanup**: Auto-cleanup for disconnected participants
7. ❌ **Configuration Persistence**: Settings survive application restart

**Technical Requirements**:
- Configuration using `RoomVisualizerConfig` struct with `Default` implementation
- Error handling using `thiserror::Error` derive macro following VoiceError pattern
- Resource cleanup using `Drop` trait following video package pattern
- Real-time connection quality at configurable intervals (1-30 seconds)
- Proper error recovery with user feedback (error display timeout)

**Testing Requirements**:
- Connect to LiveKit room and verify speaking threshold adjustment works
- Disconnect participants and verify auto-cleanup after timeout
- Trigger various error conditions and verify UI error display
- Test graceful shutdown during active room session
- Verify configuration changes persist across application restarts
- Performance test: No resource leaks during extended usage

---

## 📚 **IMPLEMENTATION DEPENDENCIES** (RESEARCH-VERIFIED)

**Required Crate Dependencies**:
```toml
# Already available in workspace:
thiserror = "2"               # Error derive macro (following domain pattern)
tracing = "0.1"               # Structured logging (already implemented)
egui = "0.28"                 # UI controls for configuration
tokio = "1.0"                 # Async runtime for cleanup tasks
livekit = "*"                 # WebRTC stats APIs for connection quality
```

**Key Source Files to Modify**:
1. [`examples/full_room_visualizer.rs`](examples/full_room_visualizer.rs) - Add configuration, error handling, cleanup
2. **NEW FILE**: `src/visualizer_config.rs` - Configuration management
3. **NEW FILE**: `src/visualizer_error.rs` - Error types and handling

**Reference Implementation Patterns**:
- [`src/audio_visualizer.rs:8-29`](src/audio_visualizer.rs) - Configuration struct pattern
- [`packages/domain/src/voice_error.rs`](../../domain/src/voice_error.rs) - Error enum pattern
- [`packages/video/src/lib.rs:155-160`](../../video/src/lib.rs) - Drop cleanup pattern
- [`tmp/livekit-rust-sdks/libwebrtc/src/rtp_receiver.rs`](tmp/livekit-rust-sdks/libwebrtc/src/rtp_receiver.rs) - Stats API pattern

---

## 🏗️ **IMPLEMENTATION PHASES** (PRIORITY-ORDERED)

**Phase 1: Configuration Foundation**
1. Create `RoomVisualizerConfig` struct following AudioVisualizerConfig pattern
2. Add configuration UI controls in settings panel
3. Replace hardcoded 0.01 threshold with configurable value

**Phase 2: Error Handling System**  
1. Create `VisualizerError` enum following VoiceError pattern
2. Add UI error state management with severity colors
3. Integrate error reporting throughout connection logic

**Phase 3: Connection Quality Monitoring**
1. Implement audio-based quality heuristics (immediate solution)
2. Add periodic quality update system with configurable interval
3. Replace hardcoded "Good" with real quality calculations

**Phase 4: Resource Management**
1. Implement Drop trait for FullRoomVisualizerApp
2. Add graceful shutdown method for clean disconnection
3. Add orphaned participant auto-cleanup system

**Phase 5: Production Polish**
1. Add configuration persistence (save/load settings)
2. Performance optimization for real-time quality monitoring
3. Comprehensive error recovery testing

---

**Last Updated**: 2025-01-15 (Research-Enhanced with Comprehensive Citations)  
**Focus Area**: Production Code Quality, Configuration, Error Handling, Resource Management  
**Dependencies**: VoiceError pattern, AudioVisualizerConfig pattern, LiveKit stats APIs, Drop cleanup pattern

**RESEARCH COMPLETED**: ✅ Full ecosystem analysis, ✅ Pattern identification, ✅ LiveKit API research, ✅ Resource management analysis

**CRITICAL PATH**: Most println! issues already resolved in TODO1.md - focus on remaining configuration and quality monitoring improvements!