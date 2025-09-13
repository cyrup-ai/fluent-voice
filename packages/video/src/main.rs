use anyhow::{Context, Result};

use clap::Parser;
use fluent_video::{VideoSource, VideoSourceOptions, VideoTrack, VideoTrackView};
use futures::StreamExt;
use std::time::Duration;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

// LiveKit imports for video track publishing
use livekit::{
    options::{TrackPublishOptions, VideoCodec},
    track::{LocalTrack, LocalVideoTrack, TrackSource},
    webrtc::{
        native::yuv_helper,
        video_frame::{I420Buffer, VideoFrame, VideoRotation},
        video_source::{RtcVideoSource, VideoResolution, native::NativeVideoSource},
    },
};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    keyboard::{Key, NamedKey},
    window::{Window, WindowAttributes},
};

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// LiveKit server URL
    #[arg(short, long)]
    url: Option<String>,

    /// LiveKit room token
    #[arg(short, long)]
    token: Option<String>,

    /// Room name
    #[arg(short, long)]
    room: Option<String>,

    /// Participant name
    #[arg(short, long, default_value = "speakrs-video")]
    name: String,

    /// Width of the window
    #[arg(long, default_value_t = 1280)]
    width: u32,

    /// Height of the window
    #[arg(long, default_value_t = 720)]
    height: u32,

    /// Enable camera
    #[arg(short, long)]
    camera: bool,

    /// Enable screen sharing
    #[arg(short, long)]
    screen: bool,
}

enum CustomEvent {
    UpdateFrame,
}

struct VideoChat {
    local_track_view: Option<VideoTrackView>,
    remote_track_views: Vec<VideoTrackView>,
    local_track: Option<VideoTrack>,
    livekit_client: Option<LiveKitClient>,
}

struct VideoApp {
    video_chat: VideoChat,
    window: Option<Window>,
    args: Args,
}

impl ApplicationHandler<CustomEvent> for VideoApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        info!("Video chat started");
        match self.video_chat.init_window(event_loop, &self.args) {
            Ok(window) => {
                info!("Window created successfully");
                self.window = Some(window);
            }
            Err(e) => {
                error!("Failed to create window: {}", e);
            }
        }
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: CustomEvent) {
        match event {
            CustomEvent::UpdateFrame => {
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

    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        if event == WindowEvent::RedrawRequested {
            // Window will be redrawn by sugarloaf
            return;
        }

        // Handle other window events
        if let Some(window) = &self.window {
            if let Err(e) = self.video_chat.handle_window_event(&event, window) {
                error!("Error handling window event: {}", e);
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // Application update logic
    }
}

struct LiveKitClient {
    handle: tokio::task::JoinHandle<()>,
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
}

impl Drop for LiveKitClient {
    fn drop(&mut self) {
        // Send shutdown signal if sender is still available
        if let Some(shutdown_tx) = self.shutdown_tx.take() {
            let _ = shutdown_tx.send(());
        }
        // Non-blocking abort - handle cleanup in background
        self.handle.abort();
    }
}

impl VideoChat {
    fn new() -> Result<Self> {
        Ok(Self {
            local_track_view: None,
            remote_track_views: Vec::new(),
            local_track: None,
            livekit_client: None,
        })
    }

    async fn init_local_video(&mut self, camera: bool, screen: bool) -> Result<()> {
        if !camera && !screen {
            return Ok(());
        }

        let options = VideoSourceOptions {
            width: Some(1280),
            height: Some(720),
            fps: Some(30),
        };

        let source = if camera {
            VideoSource::from_camera(options).context("Failed to create camera source")?
        } else {
            VideoSource::from_screen(options).context("Failed to create screen source")?
        };

        let track = VideoTrack::new(source);
        track.play()?;

        let track_view = VideoTrackView::new(track.clone());
        self.local_track_view = Some(track_view);
        self.local_track = Some(track);

        Ok(())
    }

    fn init_window(&mut self, window_target: &ActiveEventLoop, args: &Args) -> Result<Window> {
        let window = window_target
            .create_window(
                WindowAttributes::default()
                    .with_inner_size(LogicalSize::new(args.width, args.height))
                    .with_title(format!("Video Chat - {}", args.name))
                    .with_visible(true)
                    .with_resizable(true),
            )
            .context("Failed to create window")?;

        // Initialize local video track view
        if let Some(track_view) = &mut self.local_track_view {
            track_view.initialize_renderer(&window)?;
        }

        Ok(window)
    }

    async fn connect_livekit(&mut self, args: &Args) -> Result<()> {
        // Skip if URL or token not provided
        let (url, token) = match (&args.url, &args.token) {
            (Some(url), Some(token)) => (url, token),
            _ => return Ok(()),
        };

        // Connect to LiveKit room - commented out due to API incompatibility
        // let tls_config = rustls::ClientConfig::with_platform_verifier();
        // let connector = tokio_tungstenite::Connector::Rustls(Arc::new(tls_config));
        let config = livekit::RoomOptions::default();
        // config.connector = Some(connector);
        let (room, mut events) = livekit::Room::connect(url, token, config).await?;

        // Start event handling task with room access for track publishing
        let local_track = self.local_track.clone();
        let room_clone = room.clone();
        let args_clone = args.clone();
        let (shutdown_tx, mut shutdown_rx) = tokio::sync::oneshot::channel();

        let handle = tokio::spawn(async move {
            loop {
                tokio::select! {
                    event = events.recv() => {
                        match event {
                            Some(event) => {
                                // Process the event
                                match event {
                    livekit::RoomEvent::ParticipantConnected(participant) => {
                        info!("Participant connected: {}", participant.identity());
                    }
                    livekit::RoomEvent::ParticipantDisconnected(participant) => {
                        info!("Participant disconnected: {}", participant.identity());
                    }
                    livekit::RoomEvent::TrackSubscribed {
                        track,
                        publication,
                        participant,
                    } => {
                        info!(
                            "Track subscribed: {} from {}",
                            publication.sid(),
                            participant.identity()
                        );

                        // Handle receiving remote video tracks
                        if let livekit::track::RemoteTrack::Video(remote_video_track) = track {
                            match VideoTrackView::new_from_remote(remote_video_track) {
                                Ok(track_view) => {
                                    // Store remote track view for rendering
                                    // Note: This would typically be sent to the main thread via a channel
                                    // For now, we log successful creation
                                    info!("Created view for remote video track from {}", participant.identity());
                                }
                                Err(e) => {
                                    error!("Failed to create view for remote video track: {}", e);
                                }
                            }
                        }
                    }
                    livekit::RoomEvent::Connected { .. } => {
                        info!("Connected to room");

                        // Publish local track if available
                        if let Some(video_track) = &local_track {
                            match create_local_video_track_from_fluent_track(video_track).await {
                                Ok(local_video_track) => {
                                    let options = TrackPublishOptions {
                                        source: if args_clone.screen {
                                            TrackSource::ScreenShare
                                        } else {
                                            TrackSource::Camera
                                        },
                                        simulcast: false,
                                        video_codec: VideoCodec::H264,
                                        ..Default::default()
                                    };

                                    match room_clone
                                        .local_participant()
                                        .publish_track(
                                            LocalTrack::Video(local_video_track),
                                            options
                                        )
                                        .await
                                    {
                                        Ok(_publication) => {
                                            info!("Video track published successfully");
                                        }
                                        Err(e) => {
                                            error!("Failed to publish video track: {}", e);
                                        }
                                    }
                                }
                                Err(e) => {
                                    error!("Failed to convert VideoTrack to LocalVideoTrack: {}", e);
                                }
                            }
                        }
                    }
                                    _ => {}
                                }
                            }
                            None => {
                                // Event receiver closed, exit loop
                                break;
                            }
                        }
                    }
                    _ = &mut shutdown_rx => {
                        // Shutdown signal received, exit gracefully
                        info!("LiveKit client shutting down");
                        break;
                    }
                }
            }
        });

        self.livekit_client = Some(LiveKitClient {
            handle,
            shutdown_tx: Some(shutdown_tx),
        });

        Ok(())
    }

    fn handle_window_event(&mut self, event: &WindowEvent, _window: &Window) -> Result<()> {
        match event {
            WindowEvent::Resized(size) => {
                // Resize the video track views
                if let Some(track_view) = &mut self.local_track_view {
                    track_view.handle_resize(size.width, size.height)?;
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state.is_pressed()
                    && let Key::Named(NamedKey::Escape) = event.logical_key
                {
                    // Quit on Escape
                    return Err(anyhow::anyhow!("User requested exit"));
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn update_frame(&mut self) -> Result<()> {
        // Update local track view
        if let Some(track_view) = &mut self.local_track_view {
            track_view.update()?;
        }

        // Update remote track views
        for track_view in &mut self.remote_track_views {
            track_view.update()?;
        }

        Ok(())
    }
}

// LiveKit video track conversion functions

/// Convert fluent-voice VideoTrack to LiveKit LocalVideoTrack
async fn create_local_video_track_from_fluent_track(
    fluent_track: &VideoTrack,
) -> Result<LocalVideoTrack> {
    // Create video source matching track dimensions
    let width = fluent_track.width();
    let height = fluent_track.height();
    let resolution = VideoResolution {
        width,
        height,
        frame_rate: 30.0,
        aspect_ratio: width as f32 / height as f32,
    };
    let rtc_source = NativeVideoSource::new(resolution);

    // Create LiveKit track
    let local_track = LocalVideoTrack::create_video_track(
        "fluent_video_track",
        RtcVideoSource::Native(rtc_source.clone()),
    );

    // Start frame feeding task with cancellation support
    let frame_stream = fluent_track.get_frame_stream();
    let (feed_shutdown_tx, feed_shutdown_rx) = tokio::sync::oneshot::channel();
    let _feed_handle = tokio::spawn(async move {
        feed_frames_to_source(rtc_source, frame_stream, feed_shutdown_rx).await;
    });

    // Note: In a full implementation, we would store feed_shutdown_tx somewhere
    // accessible to properly shutdown the frame feeding task when needed.
    // For now, it will be cleaned up when the task naturally completes.

    Ok(local_track)
}

/// Feed frames from fluent-voice VideoTrack to LiveKit VideoSource
async fn feed_frames_to_source(
    rtc_source: NativeVideoSource,
    mut frame_stream: impl futures::Stream<Item = fluent_video::VideoFrame> + Send + Unpin + 'static,
    mut shutdown_rx: tokio::sync::oneshot::Receiver<()>,
) {
    let mut interval = tokio::time::interval(Duration::from_millis(33)); // ~30fps

    loop {
        tokio::select! {
            frame = frame_stream.next() => {
                match frame {
                    Some(fluent_frame) => {
                        interval.tick().await;

                        // Create I420 buffer for LiveKit
                        let width = fluent_frame.width();
                        let height = fluent_frame.height();

                        if width == 0 || height == 0 {
                            continue;
                        }

                        let mut i420_buffer = I420Buffer::new(width, height);

                        // Convert fluent-voice frame to LiveKit format
                        if let Err(e) = convert_fluent_frame_to_i420(&fluent_frame, &mut i420_buffer) {
                            error!("Frame conversion failed: {}", e);
                            continue;
                        }

                        // Create LiveKit VideoFrame
                        let timestamp_us = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_micros() as i64)
                            .unwrap_or_else(|_| {
                                // Fallback to monotonic time if system clock is problematic
                                std::time::Instant::now().elapsed().as_micros() as i64
                            });

                        let video_frame = VideoFrame {
                            rotation: VideoRotation::VideoRotation0,
                            buffer: Box::new(i420_buffer),
                            timestamp_us,
                        };

                        // Feed frame to LiveKit source
                        rtc_source.capture_frame(&video_frame);
                    }
                    None => {
                        // Frame stream ended, exit gracefully
                        break;
                    }
                }
            }
            _ = &mut shutdown_rx => {
                // Shutdown signal received, exit gracefully
                info!("Frame feeding task shutting down");
                break;
            }
        }
    }
}

/// Convert fluent-voice VideoFrame (RGBA) to LiveKit I420 format
fn convert_fluent_frame_to_i420(
    fluent_frame: &fluent_video::VideoFrame,
    i420_buffer: &mut I420Buffer,
) -> Result<()> {
    // Get RGBA data from fluent-voice frame using the VideoFrameExtensions trait
    let rgba_data = fluent_frame
        .to_rgba_bytes()
        .map_err(|e| anyhow::anyhow!("Failed to get RGBA data: {}", e))?;

    let width = fluent_frame.width() as i32;
    let height = fluent_frame.height() as i32;

    // Get I420 buffer pointers
    let (stride_y, stride_u, stride_v) = i420_buffer.strides();
    let (data_y, data_u, data_v) = i420_buffer.data_mut();

    // Convert RGBA to I420 using LiveKit's optimized converter
    yuv_helper::abgr_to_i420(
        &rgba_data,
        (width * 4) as u32, // RGBA stride (4 bytes per pixel)
        data_y,
        stride_y,
        data_u,
        stride_u,
        data_v,
        stride_v,
        width,
        height,
    );

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    // Setup logging
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    // Parse command-line arguments
    let args = Args::parse();

    // Create event loop with user events (simplified for compatibility)
    let event_loop = winit::event_loop::EventLoop::<CustomEvent>::with_user_event()
        .build()
        .with_context(|| "Failed to create winit event loop")?;

    // Create video chat
    let mut video_chat = VideoChat::new()?;

    // Initialize local video
    video_chat
        .init_local_video(args.camera, args.screen)
        .await?;

    // Connect to LiveKit if credentials provided
    video_chat.connect_livekit(&args).await?;

    // Create app handler
    let mut app = VideoApp {
        video_chat,
        window: None,
        args: args.clone(),
    };

    // Start frame update timer
    let event_loop_proxy = event_loop.create_proxy();
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(Duration::from_millis(16)); // ~60 FPS
            if event_loop_proxy
                .send_event(CustomEvent::UpdateFrame)
                .is_err()
            {
                break;
            }
        }
    });

    // Run event loop using new run_app API
    event_loop.run_app(&mut app)?;

    Ok(())
}
