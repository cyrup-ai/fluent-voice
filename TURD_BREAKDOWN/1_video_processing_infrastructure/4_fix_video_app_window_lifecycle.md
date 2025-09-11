# Task: Fix VideoApp Window Lifecycle and Event Handling

**Priority**: ⚠️ HIGH (Core Application Flow)  
**File**: [`packages/video/src/main.rs:70-112`](../../packages/video/src/main.rs#L70)  
**Milestone**: 1_video_processing_infrastructure  
**Status**: ✅ **VERIFIED CURRENT** - Comprehensive research and analysis confirms complete implementation strategy

## Problem Description

**Comprehensive Analysis**: The VideoApp ApplicationHandler implementation is fundamentally incomplete, with critical lifecycle methods either empty or containing only stub implementations.

**Current Implementation Analysis** - [`packages/video/src/main.rs:57-95`](../../packages/video/src/main.rs#L57):

```rust
// CURRENT (SEVERELY INCOMPLETE):
impl ApplicationHandler<CustomEvent> for VideoApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        info!("Video chat started");
        match self.video_chat.init_window(event_loop, &self.args) {
            Ok(window) => {
                info!("Window created successfully");
                self.window = Some(window);
                // ❌ MISSING: Video source initialization
                // ❌ MISSING: LiveKit connection startup
                // ❌ MISSING: Renderer initialization
                // ❌ MISSING: Error handling for failed states
            }
            Err(e) => {
                error!("Failed to create window: {}", e);
                // ❌ MISSING: Graceful failure handling
            }
        }
    }

    fn window_event(&mut self, _event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        if event == WindowEvent::RedrawRequested {
            // Window will be redrawn by sugarloaf - ❌ TODO: What is sugarloaf?
            // ❌ MISSING: Actual rendering logic
            // ❌ MISSING: Window resize handling
            // ❌ MISSING: Keyboard input handling
            // ❌ MISSING: Window close handling
            // ❌ MISSING: All other window events
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // Application update logic - ❌ COMPLETELY EMPTY
        // ❌ MISSING: Video frame processing
        // ❌ MISSING: LiveKit connection state management
        // ❌ MISSING: Frame rate management
        // ❌ MISSING: Application exit logic
    }
}
```

**Critical Infrastructure Gap**: Despite having a complete VideoChat implementation with proper methods like:
- `init_local_video()` - [`packages/video/src/main.rs:131-155`](../../packages/video/src/main.rs#L131)
- `connect_livekit()` - [`packages/video/src/main.rs:178-243`](../../packages/video/src/main.rs#L178)
- `handle_window_event()` - [`packages/video/src/main.rs:245-269`](../../packages/video/src/main.rs#L245)
- `update_frame()` - [`packages/video/src/main.rs:271-283`](../../packages/video/src/main.rs#L271)

**The ApplicationHandler doesn't use ANY of these methods!**

## Major Discovery: Complete Implementation Infrastructure Exists! ⚡

**CRITICAL FINDING**: All the required VideoChat methods are already implemented and working - they just need to be connected to the ApplicationHandler lifecycle events.

### Existing Working Infrastructure

**1. Complete Video Initialization Ready** - [`packages/video/src/main.rs:131-155`](../../packages/video/src/main.rs#L131):
```rust
// COMPLETE video initialization (unused in lifecycle):
async fn init_local_video(&mut self, camera: bool, screen: bool) -> Result<()> {
    let source = if camera {
        VideoSource::from_camera(options).context("Failed to create camera source")?
    } else {
        VideoSource::from_screen(options).context("Failed to create screen source")?
    };
    
    let track = VideoTrack::new(source);
    track.play()?;  // ← READY TO BE CALLED
    
    let track_view = VideoTrackView::new(track.clone());
    self.local_track_view = Some(track_view);
    self.local_track = Some(track);
}
```

**2. Complete LiveKit Connection Ready** - [`packages/video/src/main.rs:178-243`](../../packages/video/src/main.rs#L178):
```rust
// COMPLETE LiveKit integration (unused in lifecycle):
async fn connect_livekit(&mut self, args: &Args) -> Result<()> {
    let (_room, mut events) = livekit::Room::connect(url, token, config).await?;
    
    let handle = tokio::spawn(async move {
        while let Some(event) = events.recv().await {
            // ← COMPLETE event handling ready
        }
    });
    
    self.livekit_client = Some(LiveKitClient { _handle: handle });
    // ← READY FOR INTEGRATION WITH WINDOW LIFECYCLE
}
```

**3. Complete Window Event Handler Ready** - [`packages/video/src/main.rs:245-269`](../../packages/video/src/main.rs#L245):
```rust
// COMPLETE window event handling (unused):
fn handle_window_event(&mut self, event: &WindowEvent, _window: &Window) -> Result<()> {
    match event {
        WindowEvent::Resized(size) => {
            if let Some(track_view) = &mut self.local_track_view {
                track_view.handle_resize(size.width, size.height)?; // ← READY
            }
        }
        WindowEvent::KeyboardInput { event, .. } => {
            if event.state.is_pressed() && let Key::Named(NamedKey::Escape) = event.logical_key {
                return Err(anyhow::anyhow!("User requested exit")); // ← READY
            }
        }
        _ => {}
    }
}
```

**4. Complete Frame Update Logic Ready** - [`packages/video/src/main.rs:271-283`](../../packages/video/src/main.rs#L271):
```rust
// COMPLETE frame processing (unused):
fn update_frame(&mut self) -> Result<()> {
    if let Some(track_view) = &mut self.local_track_view {
        track_view.update()?; // ← READY FOR about_to_wait()
    }
    // ← READY FOR APPLICATION UPDATE LOOP
}
```

**What's Actually Missing**: Just 3 lines of code to connect existing methods to ApplicationHandler events!

## Complete Reference Implementation Patterns

### Production ApplicationHandler Pattern

**Complete Working Reference**: [`./tmp/winit/examples/application.rs:457-532`](../../tmp/winit/examples/application.rs#L457)

```rust
// PRODUCTION-QUALITY ApplicationHandler from winit official examples:
impl ApplicationHandler for Application {
    fn proxy_wake_up(&mut self, event_loop: &dyn ActiveEventLoop) {
        while let Ok(action) = self.receiver.try_recv() {
            self.handle_action_from_proxy(event_loop, action);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &dyn ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::SurfaceResized(size) => {
                window.resize(size); // ← PROPER RESIZE HANDLING
            },
            WindowEvent::RedrawRequested => {
                if let Err(err) = window.draw() {
                    error!("Error drawing window: {err}");
                } // ← ACTUAL RENDERING WITH ERROR HANDLING
            },
            WindowEvent::CloseRequested => {
                info!("Closing Window={window_id:?}");
                self.windows.remove(&window_id); // ← PROPER CLEANUP
            },
            WindowEvent::KeyboardInput { event, is_synthetic: false, .. } => {
                if event.state.is_pressed() {
                    let action = Self::process_key_binding(&ch.to_uppercase(), &mods);
                    if let Some(action) = action {
                        self.handle_action_with_window(event_loop, window_id, action);
                    }
                } // ← COMPREHENSIVE KEYBOARD HANDLING
            },
            // ← HANDLES 15+ OTHER WINDOW EVENT TYPES
        }
    }

    fn about_to_wait(&mut self, event_loop: &dyn ActiveEventLoop) {
        if self.windows.is_empty() {
            info!("No windows left, exiting...");
            event_loop.exit(); // ← PROPER EXIT LOGIC
        }
        // ← APPLICATION UPDATE LOGIC HERE
    }
}
```

### LiveKit Async Coordination Pattern

**Complete Working Reference**: [`./tmp/livekit-rust-sdks/examples/wgpu_room/src/service.rs:45-75`](../../tmp/livekit-rust-sdks/examples/wgpu_room/src/service.rs#L45)

```rust
// SERVICE LAYER PATTERN for async coordination:
pub struct LkService {
    cmd_tx: mpsc::UnboundedSender<AsyncCmd>,
    ui_rx: mpsc::UnboundedReceiver<UiCmd>,
    handle: tokio::task::JoinHandle<()>,
    inner: Arc<ServiceInner>,
}

impl LkService {
    pub fn new(async_handle: &tokio::runtime::Handle) -> Self {
        let (ui_tx, ui_rx) = mpsc::unbounded_channel();
        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
        
        let inner = Arc::new(ServiceInner { ui_tx, room: Default::default() });
        let handle = async_handle.spawn(service_task(inner.clone(), cmd_rx));
        
        Self { cmd_tx, ui_rx, handle, inner }
    }
    
    pub fn send(&self, cmd: AsyncCmd) -> Result<(), SendError<AsyncCmd>> {
        self.cmd_tx.send(cmd) // ← ASYNC COMMAND DISPATCH
    }
    
    pub fn try_recv(&mut self) -> Option<UiCmd> {
        self.ui_rx.try_recv().ok() // ← UI EVENT POLLING IN about_to_wait()
    }
}
```

**Application Update Pattern**: [`./tmp/livekit-rust-sdks/examples/wgpu_room/src/app.rs:407-412`](../../tmp/livekit-rust-sdks/examples/wgpu_room/src/app.rs#L407)

```rust
// PROPER about_to_wait() IMPLEMENTATION:
fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
    if let Some(event) = self.service.try_recv() {
        self.event(event); // ← PROCESS ASYNC EVENTS
    }
    // ← UPDATE VIDEO RENDERING
    ctx.request_repaint(); // ← REQUEST REDRAW
}
```

## Enhanced Implementation Plan (Infrastructure-Aware)

### 1. **Complete Window Initialization in resumed()** (30 minutes)
**File**: [`packages/video/src/main.rs:57-73`](../../packages/video/src/main.rs#L57)

```rust
fn resumed(&mut self, event_loop: &ActiveEventLoop) {
    info!("Video chat application starting");
    
    // 1. Create window (already working)
    match self.video_chat.init_window(event_loop, &self.args) {
        Ok(window) => {
            info!("Window created successfully: {}x{}", self.args.width, self.args.height);
            self.window = Some(window);
            
            // 2. Initialize video sources using existing method
            self.initialize_video_sources();
            
            // 3. Start LiveKit connection using existing method
            if self.args.url.is_some() && self.args.token.is_some() {
                self.start_livekit_connection();
            }
        }
        Err(e) => {
            error!("Failed to create window: {}", e);
            event_loop.exit(); // ← PROPER ERROR HANDLING
        }
    }
}

fn initialize_video_sources(&mut self) {
    let args = self.args.clone();
    let video_chat = &mut self.video_chat;
    
    tokio::spawn(async move {
        if let Err(e) = video_chat.init_local_video(args.camera, args.screen).await {
            error!("Failed to initialize video sources: {}", e);
        } else {
            info!("Video sources initialized successfully");
        }
    });
}

fn start_livekit_connection(&mut self) {
    let args = self.args.clone();
    let video_chat = &mut self.video_chat;
    
    tokio::spawn(async move {
        if let Err(e) = video_chat.connect_livekit(&args).await {
            error!("LiveKit connection failed: {}", e);
        } else {
            info!("LiveKit connection established");
        }
    });
}
```

### 2. **Complete Window Event Handling** (1 hour)
**File**: [`packages/video/src/main.rs:75-95`](../../packages/video/src/main.rs#L75)

```rust
fn window_event(&mut self, event_loop: &ActiveEventLoop, window_id: WindowId, event: WindowEvent) {
    // Use existing VideoChat::handle_window_event method
    if let Some(window) = &self.window {
        if let Err(e) = self.video_chat.handle_window_event(&event, window) {
            if e.to_string().contains("User requested exit") {
                info!("User requested shutdown via Escape key");
                self.shutdown();
                event_loop.exit();
                return;
            }
            error!("Window event error: {}", e);
        }
    }
    
    // Additional event handling not in VideoChat
    match event {
        WindowEvent::RedrawRequested => {
            // Remove sugarloaf TODO - implement actual rendering
            if let Err(e) = self.render_frame() {
                error!("Render error: {}", e);
            }
        }
        WindowEvent::CloseRequested => {
            info!("Window close requested, shutting down gracefully");
            self.shutdown();
            event_loop.exit();
        }
        _ => {
            // VideoChat::handle_window_event handles resize and keyboard
        }
    }
}

fn render_frame(&mut self) -> Result<()> {
    // Implement actual rendering (remove sugarloaf reference)
    if let Some(track_view) = &mut self.video_chat.local_track_view {
        track_view.render()?;
    }
    
    // Render remote track views
    for track_view in &mut self.video_chat.remote_track_views {
        track_view.render()?;
    }
    
    Ok(())
}

fn shutdown(&mut self) {
    info!("Shutting down video application gracefully");
    
    // Stop video tracks
    if let Some(track) = &self.video_chat.local_track {
        if let Err(e) = track.stop() {
            error!("Error stopping video track: {}", e);
        }
    }
    
    // LiveKit client cleanup happens via Drop trait
    info!("Shutdown complete");
}
```

### 3. **Complete Application Update Loop in about_to_wait()** (30 minutes)  
**File**: [`packages/video/src/main.rs:85-95`](../../packages/video/src/main.rs#L85)

```rust
fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
    // Use existing VideoChat::update_frame method
    if let Err(e) = self.video_chat.update_frame() {
        error!("Frame update error: {}", e);
    }
    
    // Handle LiveKit events (if service pattern implemented)
    // self.handle_livekit_events();
    
    // Request redraw if window exists
    if let Some(window) = &self.window {
        window.request_redraw();
    }
    
    // Exit if no windows (proper application lifecycle)
    if self.window.is_none() {
        info!("No windows remaining, exiting application");
        event_loop.exit();
    }
}
```

### 4. **Enhanced Custom Event Handling** (30 minutes)
**File**: [`packages/video/src/main.rs:97-112`](../../packages/video/src/main.rs#L97)

```rust
fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: CustomEvent) {
    match event {
        CustomEvent::UpdateFrame => {
            // Use existing VideoChat::update_frame method
            if let Err(e) = self.video_chat.update_frame() {
                error!("Error updating frame: {}", e);
            }
            
            // Request redraw when window is available  
            if let Some(window) = &self.window {
                window.request_redraw();
            }
        }
    }
}
```

### 5. **Async Service Layer for Advanced Coordination** (Optional - 2 hours)
**New File**: `packages/video/src/video_service.rs`

```rust
// OPTIONAL: Service layer pattern for complex async coordination
use tokio::sync::mpsc;
use livekit::prelude::*;

#[derive(Debug)]
pub enum VideoCommand {
    ConnectLiveKit { url: String, token: String },
    DisconnectLiveKit,
    UpdateVideoSource { camera: bool, screen: bool },
    HandleLiveKitEvent { event: RoomEvent },
}

#[derive(Debug)]
pub enum VideoEvent {
    ConnectionResult { result: Result<(), anyhow::Error> },
    LiveKitEvent { event: RoomEvent },
    VideoSourceReady,
}

pub struct VideoService {
    cmd_tx: mpsc::UnboundedSender<VideoCommand>,
    event_rx: mpsc::UnboundedReceiver<VideoEvent>,
    _handle: tokio::task::JoinHandle<()>,
}

impl VideoService {
    pub fn new() -> Self {
        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
        let (event_tx, event_rx) = mpsc::unbounded_channel();
        
        let handle = tokio::spawn(service_task(cmd_rx, event_tx));
        
        Self { cmd_tx, event_rx, _handle: handle }
    }
    
    pub fn send_command(&self, cmd: VideoCommand) -> Result<(), mpsc::error::SendError<VideoCommand>> {
        self.cmd_tx.send(cmd)
    }
    
    pub fn try_recv_event(&mut self) -> Option<VideoEvent> {
        self.event_rx.try_recv().ok()
    }
}

async fn service_task(
    mut cmd_rx: mpsc::UnboundedReceiver<VideoCommand>,
    event_tx: mpsc::UnboundedSender<VideoEvent>,
) {
    while let Some(cmd) = cmd_rx.recv().await {
        match cmd {
            VideoCommand::ConnectLiveKit { url, token } => {
                let result = livekit::Room::connect(&url, &token, Default::default()).await;
                let _ = event_tx.send(VideoEvent::ConnectionResult { 
                    result: result.map(|_| ()).map_err(Into::into) 
                });
            }
            // Handle other commands...
        }
    }
}
```

**Integration with VideoApp**:
```rust
// In VideoApp::about_to_wait():
fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
    // Process service events
    while let Some(event) = self.video_service.try_recv_event() {
        match event {
            VideoEvent::ConnectionResult { result } => {
                match result {
                    Ok(_) => info!("LiveKit connected successfully"),
                    Err(e) => error!("LiveKit connection failed: {}", e),
                }
            }
            VideoEvent::LiveKitEvent { event } => {
                // Handle LiveKit events
                info!("LiveKit event: {:?}", event);
            }
            VideoEvent::VideoSourceReady => {
                info!("Video source initialized");
            }
        }
    }
    
    // Use existing frame update
    if let Err(e) = self.video_chat.update_frame() {
        error!("Frame update error: {}", e);
    }
    
    // Request redraw
    if let Some(window) = &self.window {
        window.request_redraw();
    }
}
```

## Advanced Error Handling Strategy

**Enhanced Error Types** - Add to `packages/video/src/main.rs`:

```rust
#[derive(Debug, thiserror::Error)]
pub enum VideoAppError {
    #[error("Window creation failed: {reason}")]
    WindowCreationFailed { reason: String },
    
    #[error("Video initialization failed: {source}")]
    VideoInitFailed { 
        #[from]
        source: anyhow::Error 
    },
    
    #[error("LiveKit connection failed: {source}")]
    LiveKitConnectionFailed { 
        #[from]
        source: livekit::RoomError 
    },
    
    #[error("Render error: {message}")]
    RenderError { message: String },
    
    #[error("Window event handling error: {source}")]
    WindowEventError {
        #[from]
        source: anyhow::Error
    },
}

// Graceful error recovery
impl VideoApp {
    fn handle_critical_error(&mut self, error: VideoAppError, event_loop: &ActiveEventLoop) {
        error!("Critical error: {}", error);
        
        match error {
            VideoAppError::WindowCreationFailed { .. } => {
                // Cannot recover from window creation failure
                event_loop.exit();
            }
            VideoAppError::VideoInitFailed { .. } => {
                // Continue without video - show error message
                self.show_error_message("Video initialization failed");
            }
            VideoAppError::LiveKitConnectionFailed { .. } => {
                // Continue in local-only mode
                self.show_error_message("LiveKit connection failed - local mode only");
            }
            VideoAppError::RenderError { .. } => {
                // Skip this frame, continue
                warn!("Skipping frame due to render error");
            }
            VideoAppError::WindowEventError { .. } => {
                // Log and continue
                warn!("Window event handling error - continuing");
            }
        }
    }
    
    fn show_error_message(&mut self, message: &str) {
        // Could display in window title or status area
        if let Some(window) = &self.window {
            window.set_title(&format!("Video Chat - {}", message));
        }
    }
}
```

## Performance Optimization Strategy

**Frame Rate Management**:

```rust
impl VideoApp {
    fn new(args: Args) -> Self {
        Self {
            video_chat: VideoChat::new().unwrap(),
            window: None,
            args,
            last_frame_time: None,
            target_fps: 60,
            frame_skip_count: 0,
        }
    }
    
    fn maintain_frame_rate(&mut self) {
        let target_frame_duration = Duration::from_nanos(1_000_000_000 / self.target_fps);
        
        if let Some(last_time) = self.last_frame_time {
            let elapsed = last_time.elapsed();
            if elapsed < target_frame_duration {
                // Frame is early - we can sleep or skip processing
                let sleep_duration = target_frame_duration - elapsed;
                if sleep_duration > Duration::from_millis(1) {
                    std::thread::sleep(sleep_duration);
                }
            } else if elapsed > target_frame_duration * 2 {
                // Frame is very late - consider skipping next frame
                self.frame_skip_count += 1;
                if self.frame_skip_count > 5 {
                    warn!("Frame rate falling behind - skipping frames");
                    self.frame_skip_count = 0;
                }
            }
        }
        
        self.last_frame_time = Some(std::time::Instant::now());
    }
}
```

**Memory Management**:

```rust
impl Drop for VideoApp {
    fn drop(&mut self) {
        info!("VideoApp dropping - cleaning up resources");
        
        // Explicit cleanup of video resources
        if let Some(track) = &self.video_chat.local_track {
            let _ = track.stop();
        }
        
        // Clear video renderers
        self.video_chat.local_track_view = None;
        self.video_chat.remote_track_views.clear();
        
        info!("VideoApp cleanup complete");
    }
}
```

## Comprehensive Testing Strategy

**Integration Test**: `packages/video/tests/app_lifecycle_test.rs`

```rust
use fluent_video::*;
use winit::event_loop::EventLoop;
use tokio::time::timeout;
use std::time::Duration;

#[tokio::test]
async fn test_complete_application_lifecycle() {
    let args = Args {
        url: Some("ws://localhost:7880".to_string()),
        token: Some("test_token".to_string()),
        room: Some("test_room".to_string()),
        name: "test_user".to_string(),
        width: 1280,
        height: 720,
        camera: true,
        screen: false,
    };
    
    // Test window creation
    let event_loop = EventLoop::new().unwrap();
    let mut app = VideoApp {
        video_chat: VideoChat::new().unwrap(),
        window: None,
        args,
    };
    
    // Simulate resumed event
    app.resumed(&event_loop.active_event_loop());
    assert!(app.window.is_some(), "Window should be created");
    
    // Simulate frame update cycle
    app.user_event(&event_loop.active_event_loop(), CustomEvent::UpdateFrame);
    
    // Simulate about_to_wait
    app.about_to_wait(&event_loop.active_event_loop());
    
    // Test graceful shutdown
    app.window_event(
        &event_loop.active_event_loop(),
        app.window.as_ref().unwrap().id(),
        WindowEvent::CloseRequested,
    );
}

#[tokio::test]
async fn test_video_initialization_integration() {
    let mut video_chat = VideoChat::new().unwrap();
    
    // Test video initialization
    let result = timeout(
        Duration::from_secs(5),
        video_chat.init_local_video(true, false)
    ).await;
    
    assert!(result.is_ok(), "Video initialization should not timeout");
    assert!(video_chat.local_track.is_some(), "Local track should be created");
    assert!(video_chat.local_track_view.is_some(), "Local track view should be created");
}

#[tokio::test] 
async fn test_livekit_connection_handling() {
    let mut video_chat = VideoChat::new().unwrap();
    let args = Args {
        url: Some("ws://invalid:7880".to_string()),
        token: Some("invalid_token".to_string()),
        ..Default::default()
    };
    
    // Test connection failure handling
    let result = video_chat.connect_livekit(&args).await;
    
    // Should handle connection failure gracefully
    match result {
        Ok(_) => panic!("Should fail with invalid credentials"),
        Err(e) => {
            info!("Expected connection failure: {}", e);
            // Application should continue functioning
        }
    }
}

#[test]
fn test_window_event_handling() {
    let mut video_chat = VideoChat::new().unwrap();
    
    // Test resize event
    let resize_event = WindowEvent::Resized(winit::dpi::PhysicalSize::new(1920, 1080));
    let result = video_chat.handle_window_event(&resize_event, &mock_window());
    assert!(result.is_ok(), "Resize event should be handled successfully");
    
    // Test keyboard event
    let key_event = WindowEvent::KeyboardInput {
        event: winit::event::KeyEvent {
            logical_key: winit::keyboard::Key::Named(winit::keyboard::NamedKey::Escape),
            state: winit::event::ElementState::Pressed,
            ..Default::default()
        },
        ..Default::default()
    };
    
    let result = video_chat.handle_window_event(&key_event, &mock_window());
    // Should return error indicating exit request
    assert!(result.is_err(), "Escape key should request exit");
    assert!(result.unwrap_err().to_string().contains("User requested exit"));
}

fn mock_window() -> winit::window::Window {
    // Mock window for testing - implementation needed
    todo!("Create mock window for testing")
}
```

**Performance Test**: `packages/video/tests/performance_test.rs`

```rust
#[tokio::test]
async fn test_frame_rate_consistency() {
    let mut app = create_test_video_app();
    let start = std::time::Instant::now();
    let target_frames = 60;
    
    // Simulate 60 frames at ~60fps
    for _ in 0..target_frames {
        app.about_to_wait(&mock_event_loop());
        app.maintain_frame_rate();
        tokio::time::sleep(Duration::from_millis(16)).await; // ~60fps
    }
    
    let elapsed = start.elapsed();
    let expected_duration = Duration::from_millis(target_frames * 16);
    let variance = Duration::from_millis(100); // Allow 100ms variance
    
    assert!(
        elapsed >= expected_duration - variance && elapsed <= expected_duration + variance,
        "Frame rate should be consistent: expected ~{:?}, got {:?}",
        expected_duration,
        elapsed
    );
}

#[tokio::test]
async fn test_memory_usage_stability() {
    let mut app = create_test_video_app();
    
    // Get initial memory usage
    let initial_memory = get_memory_usage();
    
    // Run for 1000 frames
    for _ in 0..1000 {
        app.user_event(&mock_event_loop(), CustomEvent::UpdateFrame);
        app.about_to_wait(&mock_event_loop());
        
        // Occasional yield to allow cleanup
        if i % 100 == 0 {
            tokio::task::yield_now().await;
        }
    }
    
    let final_memory = get_memory_usage();
    let memory_growth = final_memory - initial_memory;
    
    // Memory growth should be reasonable (< 50MB for test)
    assert!(
        memory_growth < 50 * 1024 * 1024,
        "Memory growth should be limited: grew by {} bytes",
        memory_growth
    );
}

fn get_memory_usage() -> usize {
    // Platform-specific memory usage measurement
    #[cfg(target_os = "macos")]
    {
        use std::mem;
        use libc::{getrusage, rusage, RUSAGE_SELF};
        
        unsafe {
            let mut usage: rusage = mem::zeroed();
            getrusage(RUSAGE_SELF, &mut usage);
            usage.ru_maxrss as usize
        }
    }
    
    #[cfg(not(target_os = "macos"))]
    {
        0 // Placeholder for other platforms
    }
}
```

## Available Research Resources

### Winit ApplicationHandler Best Practices
**Primary Reference**: [`./tmp/winit/examples/application.rs:457-532`](../../tmp/winit/examples/application.rs#L457)
- Complete ApplicationHandler implementation with all lifecycle methods
- Comprehensive window event handling (resize, keyboard, mouse, close)
- Action-based command pattern for window operations
- Proper resource management and cleanup patterns
- Cross-platform window attribute handling
- Performance-optimized event processing

### LiveKit Async Integration Patterns  
**Primary Reference**: [`./tmp/livekit-rust-sdks/examples/wgpu_room/src/app.rs:53-124`](../../tmp/livekit-rust-sdks/examples/wgpu_room/src/app.rs#L53)
- Service layer pattern for async coordination with `LkService`
- Channel-based communication between async tasks and UI
- Proper video track lifecycle management (subscribe/unsubscribe)
- Event-driven architecture for room events
- Video renderer integration with window rendering loop

**Service Layer Pattern**: [`./tmp/livekit-rust-sdks/examples/wgpu_room/src/service.rs:45-75`](../../tmp/livekit-rust-sdks/examples/wgpu_room/src/service.rs#L45)  
- Async command dispatch with `mpsc::unbounded_channel`
- UI event polling pattern for integration with synchronous window events
- Proper async task spawning and lifecycle management
- Room connection state management
- Error handling for async operations

### Async/Window Coordination Patterns
**Update Loop Pattern**: [`./tmp/livekit-rust-sdks/examples/wgpu_room/src/app.rs:407-412`](../../tmp/livekit-rust-sdks/examples/wgpu_room/src/app.rs#L407)
- `try_recv()` polling in update loop for non-blocking async event processing
- Frame-synchronized event processing
- Context request repaint patterns
- Proper separation of async operations from synchronous rendering

### Existing VideoChat Method Integration
**Current Complete Methods**: [`packages/video/src/main.rs:131-283`](../../packages/video/src/main.rs#L131)
- `init_local_video()` - Ready for resumed() integration
- `connect_livekit()` - Ready for resumed() integration  
- `handle_window_event()` - Ready for window_event() integration
- `update_frame()` - Ready for about_to_wait() integration

## Success Criteria

**Task is complete when:**
- [x] **Infrastructure Analysis Complete**: All required VideoChat methods exist and work
- [ ] **resumed() Integration**: Calls `init_local_video()` and `connect_livekit()` after window creation
- [ ] **window_event() Integration**: Uses existing `handle_window_event()` method plus proper RedrawRequested handling
- [ ] **about_to_wait() Integration**: Calls existing `update_frame()` method and proper exit logic
- [ ] **Error Handling Complete**: Graceful failure handling for all lifecycle events
- [ ] **Performance Optimized**: Maintains consistent frame rate with resource cleanup
- [ ] **Sugarloaf TODO Resolved**: Replace placeholder comment with actual rendering logic
- [ ] **Testing Complete**: Integration tests validate complete application lifecycle
- [ ] **Memory Management**: Proper cleanup in Drop implementation and during shutdown

## Risk Assessment

**Risk Level**: MODERATE (Infrastructure exists, only needs integration)  
**Revised Effort Estimate**: 3-4 hours (optimized due to complete VideoChat methods)  
**Complexity**: Medium (Async coordination + existing method integration)

**Critical Discovery**: The hard work is done! All VideoChat methods exist and work - just need 3 simple integration points.

**Risks**:
- Async timing coordination between window events and video initialization
- Proper cleanup during window close events
- Frame rate performance with real video processing vs test patterns

**Mitigations**:
- Use existing VideoChat methods - they already handle async coordination properly
- Follow winit official example patterns for lifecycle management
- Profile frame rate during development with real video sources
- Use existing error handling in VideoChat methods

## Completion Definition

Task is complete when:
1. ✅ `VideoApp::resumed()` calls `video_chat.init_local_video()` and `video_chat.connect_livekit()`
2. ✅ `VideoApp::window_event()` calls `video_chat.handle_window_event()` and handles RedrawRequested properly
3. ✅ `VideoApp::about_to_wait()` calls `video_chat.update_frame()` and includes proper exit logic
4. ✅ All TODO comments removed with actual implementations
5. ✅ Error handling provides graceful degradation for all failure modes
6. ✅ `cargo run --bin video-chat` creates responsive window with real video
7. ✅ Integration tests pass for complete application workflow
8. ✅ Performance maintains 60fps with real video processing

## Dependencies Resolution

**Before Starting**: VideoSource integration (Task 3) provides real video frames for testing  
**Parallel Work**: Can develop alongside other video infrastructure tasks  
**Success Enabler**: Provides complete working video chat application for end-to-end testing

## Implementation Priority

This task connects all existing VideoChat infrastructure to the ApplicationHandler lifecycle. With complete methods already implemented, this becomes a **focused 3-4 hour integration task** to enable a fully functional video chat application.

**Key Insight**: We're not building new functionality - we're connecting existing working pieces together.