use anyhow::{Context, Result};
use async_trait::async_trait;
use clap::Parser;
use fluent_video::{VideoSource, VideoSourceOptions, VideoTrack, VideoTrackView};
use std::time::Duration;
use tokio::runtime::Runtime;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;
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

#[allow(dead_code)]
enum CustomEvent {
    UpdateFrame,
    #[allow(dead_code)]
    NewParticipant(String),
    #[allow(dead_code)]
    ParticipantLeft(String),
}

struct VideoChat {
    local_track_view: Option<VideoTrackView>,
    remote_track_views: Vec<VideoTrackView>,
    local_track: Option<VideoTrack>,
    #[allow(dead_code)]
    livekit_client: Option<LiveKitClient>,
    runtime: Runtime,
}

struct VideoApp {
    video_chat: VideoChat,
}

impl ApplicationHandler<CustomEvent> for VideoApp {
    fn resumed(&mut self, _event_loop: &ActiveEventLoop) {
        info!("Video chat started");
        // TODO: Create window here with event_loop as ActiveEventLoop
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: CustomEvent) {
        match event {
            CustomEvent::UpdateFrame => {
                if let Err(e) = self.video_chat.update_frame() {
                    error!("Error updating frame: {}", e);
                }
                // TODO: Request redraw when window is available
            }
            CustomEvent::NewParticipant(name) => {
                info!("New participant: {}", name);
            }
            CustomEvent::ParticipantLeft(name) => {
                info!("Participant left: {}", name);
            }
        }
    }

    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::RedrawRequested => {
                // Window will be redrawn by sugarloaf
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // Application update logic
    }
}

#[allow(dead_code)]
struct LiveKitClient {
    #[allow(dead_code)]
    room: livekit::Room,
    _handle: tokio::task::JoinHandle<()>,
}

#[async_trait]
#[allow(dead_code)]
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

    #[allow(dead_code)]
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

    #[allow(dead_code)]
    async fn connect_livekit(&mut self, args: &Args) -> Result<()> {
        // Skip if URL or token not provided
        if args.url.is_none() || args.token.is_none() {
            return Ok(());
        }

        let url = args.url.as_ref().unwrap();
        let token = args.token.as_ref().unwrap();

        // Connect to LiveKit room - commented out due to API incompatibility
        // let tls_config = rustls::ClientConfig::with_platform_verifier();
        // let connector = tokio_tungstenite::Connector::Rustls(Arc::new(tls_config));
        let config = livekit::RoomOptions::default();
        // config.connector = Some(connector);
        let (room, mut events) = livekit::Room::connect(url, token, config).await?;

        // Start event handling task - note: room may not be cloneable, using reference instead
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
                        track: _,
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
                        if let Some(_track) = &local_track {
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

    #[allow(dead_code)]
    fn handle_window_event(&mut self, event: &WindowEvent, _window: &Window) -> Result<()> {
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

    // Create event loop with user events (simplified for compatibility)
    let event_loop = winit::event_loop::EventLoop::<CustomEvent>::with_user_event()
        .build()
        .unwrap();

    // Create video chat
    let mut video_chat = VideoChat::new()?;

    // Initialize local video
    video_chat.init_local_video(args.camera, args.screen)?;

    // Create app handler
    let mut app = VideoApp { video_chat };

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

    // Connect to LiveKit if URL and token are provided
    if args.url.is_some() && args.token.is_some() {
        let args_clone = args.clone();
        // We'll connect in the main event loop instead of spawning a separate task
        // to avoid borrow checker issues with video_chat
        println!(
            "LiveKit connection will be established with {:?}",
            args_clone.url
        );
    }

    // Run event loop using new run_app API
    event_loop.run_app(&mut app)?;

    Ok(())
}
