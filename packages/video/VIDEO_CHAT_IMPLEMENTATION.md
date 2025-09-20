# Video Chat Implementation Task

## Objective
Implement functional video chat by connecting existing video infrastructure components in the `video_chat.rs` binary. This is NOT a greenfield implementation - all core video functionality already exists and needs orchestration.

## Existing Infrastructure Analysis

### ✅ Complete Video Pipeline Already Built

**VideoSource** (`src/video_source.rs`):
- Multi-platform video sources: camera, screen capture, file playback
- Async streaming with configurable resolution, FPS, frame formats
- Platform-specific implementations (macOS AVFoundation, generic fallback)
- Thread-safe frame access with proper error handling

**VideoTrack** (`src/track.rs`):
- Thread-safe video track management with 60fps frame processing
- Async frame update loops with configurable frame processors
- Stream interface for reactive video frame access
- Built-in play/stop lifecycle management

**VideoFrame + NativeVideoFrame** (`src/video_frame.rs`, `src/native_video.rs`):
- Platform-optimized video frame handling with zero-copy operations
- Video rotation support (0°, 90°, 180°, 270°)
- RGBA conversion utilities for display and processing
- macOS Core Video buffer integration

**VideoTrackView** (`src/lib.rs:165-285`):
- Display component supporting both terminal ASCII art and window rendering
- Remote LiveKit track integration via `new_from_remote()`
- Real-time video frame to ASCII art conversion
- Terminal cursor management and frame buffer rendering

### ✅ Complete CLI Framework

**CLI Arguments** (`src/cli_args.rs`):
- Stream mode: URL, user-id, meeting-id for video calls
- Info mode: Camera discovery, format/resolution listing
- Video device selection (index, name, UUID)
- Resolution configuration (default 1280x720)
- FPS, bitrate, VP9 encoding controls
- Frame format support (NV12, YUYV)
- Debug features (test patterns, offline testing)

### ✅ LiveKit Integration Ready

**Dependencies** (`Cargo.toml:125-128`):
```toml
libwebrtc = { git = "https://github.com/zed-industries/livekit-rust-sdks" }
livekit = { git = "https://github.com/zed-industries/livekit-rust-sdks", features = ["__rustls-tls"] }
```

**Remote Track Support**: `VideoTrackView::new_from_remote()` already handles LiveKit remote tracks

## Current State

**❌ Missing Implementation**: `src/bin/video_chat.rs` contains only placeholder:
```rust
fn main() {
    println!("Video chat functionality - placeholder implementation");
    // TODO: Implement video chat functionality
}
```

## Implementation Strategy: "Three Wires" Approach

This is pure integration work - connecting existing production-ready components:

### Wire 1: CLI → VideoSource Setup
- Parse `cli_args::Opt` using existing structures
- Create `VideoSource` from CLI parameters:
  - `VideoSource::from_camera()` for camera capture
  - `VideoSource::from_screen()` for screen sharing  
  - Apply `VideoSourceOptions` with resolution, FPS from CLI

### Wire 2: VideoSource → LiveKit Publishing
- Wrap `VideoSource` in `VideoTrack::new()`
- Connect `VideoTrack::get_frame_stream()` to LiveKit room publishing
- Establish LiveKit room connection using CLI URL, user_id, meeting_id
- Configure VP9 encoding with CLI bitrate/quality settings

### Wire 3: LiveKit Remote Tracks → Display
- Handle remote participant tracks via `VideoTrackView::new_from_remote()`
- Use `TerminalRenderer` for ASCII art display of remote video
- Support multiple remote participants with separate VideoTrackView instances

## Detailed Implementation Plan

### Phase 1: CLI Integration (30 lines)
```rust
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    use clap::Parser;
    use fluent_video::cli_args::Opt;
    
    let opt = Opt::parse();
    match opt.mode {
        Mode::Stream(stream_args) => handle_stream_mode(stream_args).await,
        Mode::Info(info_args) => handle_info_mode(info_args).await,
    }
}
```

### Phase 2: Video Source Setup (50 lines)
```rust
async fn handle_stream_mode(args: Stream) -> anyhow::Result<()> {
    // Parse resolution string "1280x720" → (1280, 720)
    let (width, height) = parse_resolution(&args.resolution)?;
    
    // Create video source options from CLI
    let options = VideoSourceOptions {
        width: Some(width),
        height: Some(height), 
        fps: Some(args.fps),
    };
    
    // Initialize video source based on CLI settings
    let source = if args.send_test_pattern {
        VideoSource::from_test_pattern(options)?
    } else {
        VideoSource::from_camera(options)?
    };
    
    let track = VideoTrack::new(source);
    track.play()?;
}
```

### Phase 3: LiveKit Room Connection (80 lines)  
```rust
use livekit::{Room, RoomOptions, VideoTrack as LiveKitTrack};

async fn setup_livekit_room(args: &Stream, track: VideoTrack) -> anyhow::Result<()> {
    // Connect to LiveKit room
    let room_options = RoomOptions::default();
    let room = Room::connect(&args.url.to_string(), &room_options).await?;
    
    // Publish local video track
    let local_participant = room.local_participant();
    let livekit_track = LiveKitTrack::from_frame_stream(track.get_frame_stream());
    local_participant.publish_track(livekit_track).await?;
    
    // Handle remote participants
    let mut remote_views = Vec::new();
    room.on_participant_connected(|participant| {
        for track in participant.video_tracks() {
            let view = VideoTrackView::new_from_remote(track)?;
            remote_views.push(view);
        }
    });
}
```

### Phase 4: Display Management (60 lines)
```rust
async fn render_video_chat(local_track: VideoTrack, remote_views: Vec<VideoTrackView>) -> anyhow::Result<()> {
    // Create local preview
    let mut local_view = VideoTrackView::new(local_track);
    local_view.initialize_renderer(&window)?;
    
    // Render loop
    loop {
        // Update local preview
        local_view.update()?;
        
        // Update remote participant views
        for view in &mut remote_views {
            view.update()?;
        }
        
        tokio::time::sleep(Duration::from_millis(16)).await; // 60fps
    }
}
```

### Phase 5: Info Mode Implementation (40 lines)
```rust
async fn handle_info_mode(args: Info) -> anyhow::Result<()> {
    if args.list_cameras {
        // Use existing VideoSource camera discovery
        let cameras = VideoSource::list_cameras()?;
        for (i, camera) in cameras.iter().enumerate() {
            println!("{}: {}", i, camera.name);
        }
    }
    
    if let Some(device) = args.list_formats {
        // List supported formats for device
        let formats = VideoSource::list_formats_for_device(device)?;
        for format in formats {
            println!("{}", format);
        }
    }
}
```

## Dependencies & Integration Points

### Existing APIs to Leverage:
- `VideoSource::from_camera(VideoSourceOptions)` - Camera capture
- `VideoTrack::new(VideoSource)` - Track management  
- `VideoTrack::get_frame_stream()` - Async frame streaming
- `VideoTrackView::new_from_remote(RemoteVideoTrack)` - Remote display
- `TerminalRenderer` - ASCII art video display
- `cli_args::Opt` - Complete CLI parsing

### LiveKit Integration Requirements:
- Room connection and authentication
- Local track publishing with VP9 encoding
- Remote participant track handling
- Connection lifecycle management

### Error Handling Chain:
- `VideoSource` errors → `anyhow::Result`
- `LiveKit` errors → `livekit::Error` → `anyhow::Result`
- Terminal rendering errors → `anyhow::Result`

## Testing Strategy

### Debug Features (Already Implemented):
- `--debug-send-test-pattern` - Test without camera
- `--debug-offline-streaming-test` - Local encoding validation
- `--debug-keylog` - TLS debugging for LiveKit connection

### Validation Steps:
1. Camera discovery via Info mode
2. Local video capture and encoding
3. LiveKit room connection and publishing
4. Remote participant video display
5. Resource cleanup and error handling

## Acceptance Criteria

### ✅ Functional Requirements:
- [ ] Parse CLI arguments correctly
- [ ] Initialize video source from camera/screen/test pattern
- [ ] Connect to LiveKit room with user credentials
- [ ] Publish local video track with configured encoding
- [ ] Display remote participant video feeds
- [ ] Support multiple concurrent participants
- [ ] Handle camera device selection by index/name/UUID

### ✅ Quality Requirements:
- [ ] Proper async/await coordination
- [ ] Resource cleanup (camera, LiveKit connection, terminal)
- [ ] Error handling with user-friendly messages
- [ ] 60fps local rendering, 30fps network streaming
- [ ] VP9 encoding with configurable quality/speed tradeoff

### ✅ Platform Requirements:
- [ ] macOS AVFoundation camera integration
- [ ] Cross-platform fallback support
- [ ] Terminal ASCII art rendering on all platforms
- [ ] LiveKit WebRTC compatibility

## Implementation Size Estimate

**Total: ~260 lines of orchestration code**
- CLI integration: ~30 lines
- Video source setup: ~50 lines  
- LiveKit room management: ~80 lines
- Display rendering: ~60 lines
- Info mode: ~40 lines

This is pure integration work leveraging 100% existing infrastructure. No new video processing, encoding, or rendering code required.