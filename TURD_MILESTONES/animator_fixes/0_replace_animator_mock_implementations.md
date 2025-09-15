# Replace Animator Mock Implementations

## Description
Replace mock implementations in animator examples with real LiveKit integration to provide production-quality examples for users.

## Current Violations
- `packages/animator/src/examples/amplitude_waveform_example.rs:12,117,128` - Uses `create_mock_audio_track()`
- `packages/animator/src/examples/video_rendering_example.rs:13,128,135` - Uses `create_mock_video_track()`

## Discovered Existing Infrastructure

### LiveKit Package Already Provides Complete Implementation
The codebase already has a fully functional LiveKit integration in [`packages/livekit/`](../../packages/livekit/src/):

1. **Room Connection** ([`livekit_client.rs:41-65`](../../packages/livekit/src/livekit_client.rs)):
   ```rust
   pub async fn connect(
       url: String,
       token: String,
   ) -> Result<(Self, mpsc::UnboundedReceiver<RoomEvent>)>
   ```

2. **Audio/Video Track Types** ([`livekit_client.rs:11-14`](../../packages/livekit/src/livekit_client.rs)):
   ```rust
   pub struct RemoteVideoTrack(pub livekit::track::RemoteVideoTrack);
   pub struct RemoteAudioTrack(pub livekit::track::RemoteAudioTrack);
   ```

3. **Playback Module** ([`playback.rs`](../../packages/livekit/src/playback.rs)):
   - Lock-free audio stack with atomic operations
   - Video frame handling with platform-specific optimizations
   - Pre-allocated frame buffer pools for zero allocation

### AudioVisualizer Integration Points
The [`AudioVisualizer`](../../packages/animator/src/audio_visualizer.rs) expects `RtcAudioTrack` which comes from `livekit::webrtc::prelude::*`. This is the underlying WebRTC type that our wrapper types contain.

## Technical Resolution

### Step 1: Update Dependencies
Add livekit package dependency to animator's Cargo.toml:
```toml
[dependencies]
livekit-wrapper = { path = "../livekit" }
```

### Step 2: Replace Mock Audio Track Implementation

**File:** `packages/animator/src/examples/amplitude_waveform_example.rs`

```rust
use livekit_wrapper::{Room, RemoteAudioTrack, RoomEvent};
use futures::StreamExt;

// Replace mock function with real LiveKit connection
async fn create_livekit_audio_track(
    room_url: &str,
    access_token: &str,
) -> Result<livekit::webrtc::prelude::RtcAudioTrack, Box<dyn std::error::Error>> {
    // Connect to LiveKit room
    let (room, mut events) = Room::connect(
        room_url.to_string(),
        access_token.to_string(),
    ).await?;
    
    // Wait for first audio track from any participant
    while let Some(event) = events.next().await {
        match event {
            RoomEvent::TrackSubscribed { 
                track: livekit_wrapper::RemoteTrack::Audio(audio_track),
                publication,
                participant,
            } => {
                tracing::info!(
                    participant = %participant.identity().0,
                    track_sid = %audio_track.sid(),
                    "Connected to remote audio track"
                );
                
                // Enable the track
                publication.set_enabled(true);
                
                // Return the underlying RTC track for AudioVisualizer
                return Ok(audio_track.0.clone());
            }
            RoomEvent::Connected { participants_with_tracks } => {
                // Check existing participants for audio tracks
                for (participant, publications) in participants_with_tracks {
                    for publication in publications {
                        if publication.is_audio() {
                            if let Some(livekit_wrapper::RemoteTrack::Audio(track)) = publication.track() {
                                tracing::info!(
                                    participant = %participant.identity().0,
                                    "Found existing audio track"
                                );
                                publication.set_enabled(true);
                                return Ok(track.0.clone());
                            }
                        }
                    }
                }
            }
            _ => continue,
        }
    }
    
    Err("No audio track found in room".into())
}

// Update the example runner to handle async and connection params
pub fn run_amplitude_waveform_example() -> Result<(), Box<dyn std::error::Error>> {
    // Get connection details from environment or config
    let room_url = std::env::var("LIVEKIT_URL")
        .unwrap_or_else(|_| "wss://your-livekit-server.com".to_string());
    let access_token = std::env::var("LIVEKIT_TOKEN")
        .ok_or("LIVEKIT_TOKEN environment variable required")?;
    
    // Create tokio runtime for async operations
    let runtime = Runtime::new()?;
    
    // Connect to LiveKit and get audio track
    let audio_track = runtime.block_on(async {
        create_livekit_audio_track(&room_url, &access_token).await
    })?;
    
    // Rest of the code remains the same...
    let visualizer = AudioVisualizer::new(runtime.handle(), audio_track);
    // ...
}
```

### Step 3: Replace Mock Video Track Implementation

**File:** `packages/animator/src/examples/video_rendering_example.rs`

```rust
use livekit_wrapper::{Room, RemoteVideoTrack, RoomEvent, play_remote_video_track};
use futures::StreamExt;

async fn create_livekit_video_track(
    room_url: &str,
    access_token: &str,
) -> Result<livekit::webrtc::prelude::RtcVideoTrack, Box<dyn std::error::Error>> {
    // Connect to LiveKit room
    let (room, mut events) = Room::connect(
        room_url.to_string(),
        access_token.to_string(),
    ).await?;
    
    // Wait for first video track
    while let Some(event) = events.next().await {
        match event {
            RoomEvent::TrackSubscribed {
                track: livekit_wrapper::RemoteTrack::Video(video_track),
                publication,
                participant,
            } => {
                tracing::info!(
                    participant = %participant.identity().0,
                    track_sid = %video_track.sid(),
                    "Connected to remote video track"
                );
                
                // Enable the track
                publication.set_enabled(true);
                
                // Return the underlying RTC track
                return Ok(video_track.0.clone());
            }
            RoomEvent::Connected { participants_with_tracks } => {
                // Check for existing video tracks
                for (participant, publications) in participants_with_tracks {
                    for publication in publications {
                        if !publication.is_audio() {
                            if let Some(livekit_wrapper::RemoteTrack::Video(track)) = publication.track() {
                                tracing::info!(
                                    participant = %participant.identity().0,
                                    "Found existing video track"
                                );
                                publication.set_enabled(true);
                                return Ok(track.0.clone());
                            }
                        }
                    }
                }
            }
            _ => continue,
        }
    }
    
    Err("No video track found in room".into())
}
```

### Step 4: Enhanced Example with Full Room Management

Create a new comprehensive example that showcases full LiveKit integration:

**File:** `packages/animator/src/examples/livekit_room_visualizer.rs`

```rust
use livekit_wrapper::{Room, RoomEvent, RemoteTrack, ConnectionState};
use std::collections::HashMap;
use tokio::sync::mpsc;

pub struct LiveKitRoomVisualizer {
    room: Room,
    audio_visualizers: HashMap<String, AudioVisualizer>,
    video_renderers: HashMap<String, VideoRenderer>,
    event_receiver: mpsc::UnboundedReceiver<RoomEvent>,
}

impl LiveKitRoomVisualizer {
    pub async fn new(
        room_url: &str,
        access_token: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let (room, event_receiver) = Room::connect(
            room_url.to_string(),
            access_token.to_string(),
        ).await?;
        
        Ok(Self {
            room,
            audio_visualizers: HashMap::new(),
            video_renderers: HashMap::new(),
            event_receiver,
        })
    }
    
    pub async fn process_events(&mut self, rt_handle: &tokio::runtime::Handle) {
        while let Some(event) = self.event_receiver.recv().await {
            match event {
                RoomEvent::TrackSubscribed { track, participant, .. } => {
                    let participant_id = participant.identity().0.clone();
                    
                    match track {
                        RemoteTrack::Audio(audio) => {
                            let visualizer = AudioVisualizer::new(
                                rt_handle,
                                audio.0.clone()
                            );
                            self.audio_visualizers.insert(participant_id, visualizer);
                        }
                        RemoteTrack::Video(video) => {
                            // Create video renderer for the track
                            // Implementation depends on rendering backend
                        }
                    }
                }
                RoomEvent::TrackUnsubscribed { track, participant, .. } => {
                    let participant_id = participant.identity().0.clone();
                    match track {
                        RemoteTrack::Audio(_) => {
                            self.audio_visualizers.remove(&participant_id);
                        }
                        RemoteTrack::Video(_) => {
                            self.video_renderers.remove(&participant_id);
                        }
                    }
                }
                RoomEvent::Disconnected { reason } => {
                    tracing::warn!("Disconnected from room: {}", reason);
                    break;
                }
                _ => {}
            }
        }
    }
}
```

## Implementation Patterns from Existing Code

### Pattern 1: Lock-Free Audio Processing
From [`packages/livekit/src/playback.rs:54-90`](../../packages/livekit/src/playback.rs):
- Uses atomic operations and channels instead of mutexes
- Pre-allocated frame buffer pools for zero allocation
- Crossbeam queues for lock-free communication

### Pattern 2: Platform-Specific Video Handling
From [`packages/livekit/src/playback.rs:20-35`](../../packages/livekit/src/playback.rs):
- macOS-specific CoreVideo pixel format constants
- Platform-optimized video frame handling

### Pattern 3: Event-Driven Architecture
From [`packages/livekit/src/livekit_client.rs:48-57`](../../packages/livekit/src/livekit_client.rs):
- Unbounded channels for event distribution
- Tokio spawned tasks for event processing
- Clean separation of concerns

## Testing Strategy

### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_room_connection() {
        // Test with mock LiveKit server
        let mock_server = MockLiveKitServer::new();
        let (room, _) = Room::connect(
            mock_server.url(),
            mock_server.generate_token(),
        ).await.unwrap();
        
        assert_eq!(room.connection_state(), ConnectionState::Connected);
    }
}
```

### Integration Tests
1. Test with LiveKit Cloud sandbox environment
2. Verify audio amplitude extraction accuracy
3. Validate video frame rendering pipeline
4. Test reconnection and error handling

## Configuration Management

### Environment Variables
```bash
# Required for running examples
export LIVEKIT_URL="wss://your-domain.livekit.cloud"
export LIVEKIT_TOKEN="your-access-token"
export LIVEKIT_API_KEY="your-api-key"
export LIVEKIT_API_SECRET="your-api-secret"
```

### Configuration File Support
```toml
# livekit.toml
[connection]
url = "wss://your-domain.livekit.cloud"
auto_reconnect = true
timeout_seconds = 30

[audio]
sample_rate = 48000
channels = 2
echo_cancellation = true

[video]
max_resolution = "1920x1080"
target_fps = 30
```

## Error Handling Improvements

### Comprehensive Error Types
```rust
#[derive(Debug, thiserror::Error)]
pub enum LiveKitExampleError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),
    
    #[error("No tracks available in room")]
    NoTracksAvailable,
    
    #[error("Track subscription failed: {0}")]
    SubscriptionFailed(String),
    
    #[error("Authentication failed: {0}")]
    AuthenticationFailed(String),
}
```

## Documentation Requirements

### User Guide
1. **Setup Instructions**
   - LiveKit server setup (cloud or self-hosted)
   - Token generation guide
   - Network requirements

2. **API Documentation**
   - Connection parameters
   - Event handling patterns
   - Resource cleanup

3. **Examples**
   - Basic audio visualization
   - Video rendering with controls
   - Multi-participant room handling
   - Recording and playback

## Success Criteria
- [x] Identify existing LiveKit implementation in codebase
- [x] Map RTC types to wrapper types
- [ ] Remove all `create_mock_audio_track()` usage
- [ ] Remove all `create_mock_video_track()` usage  
- [ ] Implement real LiveKit room connections
- [ ] Add proper error handling for network issues
- [ ] Include authentication and connection management
- [ ] Provide working examples that connect to real LiveKit rooms
- [ ] Add comprehensive documentation for setup
- [ ] Add integration tests with LiveKit sandbox
- [ ] Add configuration file support
- [ ] Implement graceful degradation for offline mode

## Dependencies
- Milestone 0: Async Architecture Compliance - ✅ Already using tokio
- Milestone 1: Configuration Management - Need to add config file support
- Milestone 2: Audio Processing Enhancement - ✅ Lock-free architecture in place

## Architecture Impact
MEDIUM - While originally assessed as LOW, this actually improves architecture by:
- Demonstrating proper async patterns
- Showcasing lock-free audio processing
- Providing production-ready examples
- Establishing patterns for other integrations

## References
- LiveKit Rust SDK: [`./tmp/livekit-rust-sdks/`](../../tmp/livekit-rust-sdks/)
- Existing Implementation: [`packages/livekit/src/`](../../packages/livekit/src/)
- AudioVisualizer: [`packages/animator/src/audio_visualizer.rs`](../../packages/animator/src/audio_visualizer.rs)
- Current Examples: [`packages/animator/src/examples/`](../../packages/animator/src/examples/)

## Implementation Notes

### IMPORTANT DISCOVERIES:
1. **The livekit package already exists** with full Room connection, audio/video track handling
2. **RtcAudioTrack and RtcVideoTrack** are from `livekit::webrtc::prelude::*`, not custom types
3. **Lock-free architecture** is already implemented in the playback module
4. **Event-driven patterns** are established and should be followed

### MINIMAL CHANGES NEEDED:
The task is simpler than originally described - we just need to:
1. Import the existing livekit package
2. Replace mock track creation with real Room connections
3. Add configuration/environment variable support
4. Update documentation with setup instructions

No need to implement LiveKit from scratch - it's already done!