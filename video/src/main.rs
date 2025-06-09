use anyhow::{Context, Result};
use async_trait::async_trait;
use clap::Parser;
use futures::StreamExt;
use raw_window_handle::{HasRawWindowHandle, RawDisplayHandle, RawWindowHandle};
use rio_window::{
    dpi::{LogicalSize, PhysicalSize},
    event::{Event, StartCause, WindowEvent},
    event_loop::{ControlFlow, EventLoop, EventLoopBuilder, EventLoopProxy, EventLoopWindowTarget},
    keyboard::{Key, NamedKey},
    monitor::MonitorHandle,
    window::{Window, WindowAttributes, WindowId},
};
use speakrs_video::{VideoSource, VideoSourceOptions, VideoTrack, VideoTrackView};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::runtime::Runtime;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
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

struct WindowWrapper {
    window: Window,
}

impl HasRawWindowHandle for WindowWrapper {
    fn raw_window_handle(&self) -> RawWindowHandle {
        self.window.raw_window_handle().unwrap()
    }
}

impl HasRawDisplayHandle for WindowWrapper {
    fn raw_display_handle(&self) -> RawDisplayHandle {
        self.window.raw_display_handle().unwrap()
    }
}

enum CustomEvent {
    UpdateFrame,
    NewParticipant(String),
    ParticipantLeft(String),
}

struct VideoChat {
    local_track_view: Option<VideoTrackView>,
    remote_track_views: Vec<VideoTrackView>,
    local_track: Option<VideoTrack>,
    livekit_client: Option<LiveKitClient>,
    runtime: Runtime,
}

struct LiveKitClient {
    room: livekit::Room,
    _handle: tokio::task::JoinHandle<()>,
}

#[async_trait]
trait LiveKitRoomHandler {
    async fn handle_event(&mut self, event: livekit::RoomEvent);
}

impl VideoChat {
    fn new() -> Result<Self> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .context("Failed to create Tokio runtime")?;

        Ok(Self {
            local_track_view: None,
            remote_track_views: Vec::new(),
            local_track: None,
            livekit_client: None,
            runtime,
        })
    }

    fn init_local_video(&mut self, camera: bool, screen: bool) -> Result<()> {
        if !camera && !screen {
            return Ok(());
        }

        let options = VideoSourceOptions {
            width: Some(1280),
            height: Some(720),
            fps: Some(30),
        };

        let source = if camera {
            self.runtime.block_on(async {
                VideoSource::from_camera(options).context("Failed to create camera source")
            })?
        } else {
            self.runtime.block_on(async {
                VideoSource::from_screen(options).context("Failed to create screen source")
            })?
        };

        let track = VideoTrack::new(source);
        self.runtime.block_on(async { track.play() })?;

        let track_view = VideoTrackView::new(track.clone());
        self.local_track_view = Some(track_view);
        self.local_track = Some(track);

        Ok(())
    }

    fn init_window(
        &mut self,
        window_target: &EventLoopWindowTarget<CustomEvent>,
        args: &Args,
    ) -> Result<Window> {
        let window = WindowAttributes::default()
            .with_inner_size(LogicalSize::new(args.width, args.height))
            .with_title(format!("Video Chat - {}", args.name))
            .with_visible(true)
            .with_resizable(true)
            .build(window_target)
            .context("Failed to create window")?;

        // Initialize local video track view
        if let Some(track_view) = &mut self.local_track_view {
            let mut window_wrapper = WindowWrapper {
                window: window.clone(),
            };
            track_view.initialize_renderer(window_wrapper.raw_window_handle())?;
        }

        Ok(window)
    }

    async fn connect_livekit(&mut self, args: &Args) -> Result<()> {
        // Skip if URL or token not provided
        if args.url.is_none() || args.token.is_none() {
            return Ok(());
        }

        let url = args.url.as_ref().unwrap();
        let token = args.token.as_ref().unwrap();

        // Connect to LiveKit room
        let connector =
            tokio_tungstenite::Connector::Rustls(Arc::new(http_client_tls::tls_config()));
        let mut config = livekit::RoomOptions::default();
        config.connector = Some(connector);
        let (room, mut events) = livekit::Room::connect(url, token, config).await?;

        // Start event handling task
        let room_clone = room.clone();
        let local_track = self.local_track.clone();
        let handle = tokio::spawn(async move {
            while let Some(event) = events.recv().await {
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
                        // Handle receiving remote tracks here
                    }
                    livekit::RoomEvent::Connected { .. } => {
                        info!("Connected to room");

                        // Publish local track if available
                        if let Some(track) = &local_track {
                            // This is a placeholder - actual implementation would convert VideoTrack to livekit::track::LocalTrack
                            // room_clone.local_participant().publish_track(...).await.ok();
                        }
                    }
                    _ => {}
                }
            }
        });

        self.livekit_client = Some(LiveKitClient {
            room,
            _handle: handle,
        });

        Ok(())
    }

    fn handle_window_event(&mut self, event: &WindowEvent, window: &Window) -> Result<()> {
        match event {
            WindowEvent::Resized(size) => {
                // Resize the video track views
                if let Some(track_view) = &mut self.local_track_view {
                    track_view.handle_resize(size.width, size.height)?;
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state.is_pressed() {
                    if let Key::Named(NamedKey::Escape) = event.logical_key {
                        // Quit on Escape
                        return Err(anyhow::anyhow!("User requested exit"));
                    }
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

fn main() -> Result<()> {
    // Setup logging
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    // Parse command-line arguments
    let args = Args::parse();

    // Create event loop
    let event_loop = EventLoopBuilder::<CustomEvent>::with_user_event()
        .build()
        .context("Failed to create event loop")?;

    // Create video chat
    let mut video_chat = VideoChat::new()?;

    // Initialize local video
    video_chat.init_local_video(args.camera, args.screen)?;

    // Create window
    let window = video_chat.init_window(&event_loop, &args)?;

    // Start frame update timer
    let event_loop_proxy = event_loop.create_proxy();
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(Duration::from_millis(16)); // ~60 FPS
            if let Err(_) = event_loop_proxy.send_event(CustomEvent::UpdateFrame) {
                break;
            }
        }
    });

    // Connect to LiveKit in a separate thread
    let args_clone = args.clone();
    let mut video_chat_ref = &mut video_chat;
    if args.url.is_some() && args.token.is_some() {
        video_chat.runtime.spawn(async move {
            if let Err(e) = video_chat_ref.connect_livekit(&args_clone).await {
                error!("Failed to connect to LiveKit: {}", e);
            }
        });
    }

    // Run event loop
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;

        match event {
            Event::NewEvents(StartCause::Init) => {
                info!("Video chat started");
            }
            Event::WindowEvent { event, window_id } if window_id == window.id() => {
                if let Err(e) = video_chat.handle_window_event(&event, &window) {
                    error!("Error handling window event: {}", e);
                    *control_flow = ControlFlow::Exit;
                }

                match event {
                    WindowEvent::CloseRequested => {
                        *control_flow = ControlFlow::Exit;
                    }
                    _ => {}
                }
            }
            Event::UserEvent(CustomEvent::UpdateFrame) => {
                if let Err(e) = video_chat.update_frame() {
                    error!("Error updating frame: {}", e);
                }
                window.request_redraw();
            }
            Event::UserEvent(CustomEvent::NewParticipant(name)) => {
                info!("New participant: {}", name);
            }
            Event::UserEvent(CustomEvent::ParticipantLeft(name)) => {
                info!("Participant left: {}", name);
            }
            Event::MainEventsCleared => {
                // Application update logic
            }
            Event::RedrawRequested(_) => {
                // Window will be redrawn by sugarloaf
            }
            _ => {}
        }
    })
}
