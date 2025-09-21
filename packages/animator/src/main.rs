use anyhow::Result;
use clap::{Parser, Subcommand};
use crossterm::ExecutableCommand;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::{Terminal, backend::CrosstermBackend};
use std::io;

use fluent_voice_animator::{
    app::{GuiApp, TerminalApp},
    audioio::cpal::DefaultAudioDeviceWithCPAL,
    cfg::{ScopeSource, SourceOptions, UiOptions},
};

#[derive(Parser, Debug)]
#[command(author, version, about)]
pub struct AnimatorArgs {
    #[clap(subcommand)]
    pub mode: AnimatorMode,
}

#[derive(Debug, Clone, Subcommand)]
pub enum AnimatorMode {
    /// Terminal-based oscilloscope (original functionality)
    Terminal {
        #[clap(subcommand)]
        source: ScopeSource,
        #[command(flatten)]
        opts: SourceOptions,
        #[command(flatten)]
        ui: UiOptions,
    },
    /// GUI-based room visualizer with LiveKit integration
    Room {
        /// LiveKit server URL
        #[arg(long, env = "LIVEKIT_URL")]
        url: Option<String>,
        /// API key for LiveKit
        #[arg(long, env = "LIVEKIT_API_KEY")]
        api_key: Option<String>,
        /// API secret for LiveKit
        #[arg(long, env = "LIVEKIT_API_SECRET")]
        api_secret: Option<String>,
        /// Room name to join
        #[arg(long, default_value = "fluent-voice-demo")]
        room_name: String,
        /// Participant name
        #[arg(long, default_value = "Rust Visualizer")]
        participant_name: String,
        /// Auto-connect on startup
        #[arg(long)]
        auto_connect: bool,
    },
}

fn main() -> Result<()> {
    let args = AnimatorArgs::parse();

    match args.mode {
        AnimatorMode::Terminal {
            source,
            mut opts,
            ui,
        } => run_terminal_mode(source, opts, ui),
        AnimatorMode::Room {
            url,
            api_key,
            api_secret,
            room_name,
            participant_name,
            auto_connect,
        } => run_gui_mode(
            url,
            api_key,
            api_secret,
            room_name,
            participant_name,
            auto_connect,
        ),
    }
}

fn run_terminal_mode(source: ScopeSource, mut opts: SourceOptions, ui: UiOptions) -> Result<()> {
    opts.tune();

    // Handle device listing first
    if let ScopeSource::Audio { list: true, .. } = &source {
        #[cfg(feature = "microphone")]
        {
            use cpal::traits::{DeviceTrait, HostTrait};
            let host = cpal::default_host();
            println!("Available audio input devices:");
            if let Ok(devices) = host.input_devices() {
                for device in devices {
                    let name = device
                        .name()
                        .unwrap_or_else(|_| "Unknown Device".to_string());
                    println!("  - {}", name);
                }
            }
        }
        #[cfg(not(feature = "microphone"))]
        {
            println!("Microphone feature not enabled - cannot list devices");
        }
        return Ok(());
    }

    // Initialize audio source
    let source = match source {
        ScopeSource::Audio {
            device,
            timeout,
            list: false,
        } => DefaultAudioDeviceWithCPAL::instantiate(device, opts.clone(), timeout)?,
        _ => unreachable!("Audio source should be guaranteed by checks above"),
    };

    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    stdout.execute(EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create and run the terminal app
    let mut app = TerminalApp::new(&ui, &opts);
    let result = app.run(source, &mut terminal);

    // Restore terminal
    disable_raw_mode()?;
    terminal.backend_mut().execute(LeaveAlternateScreen)?;

    result
}

fn run_gui_mode(
    url: Option<String>,
    api_key: Option<String>,
    api_secret: Option<String>,
    room_name: String,
    participant_name: String,
    auto_connect: bool,
) -> Result<()> {
    // Initialize tracing for GUI mode
    tracing_subscriber::fmt::init();

    // Configure eframe options
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_title("Fluent Voice Animator - Room Visualizer"),
        ..Default::default()
    };

    // Run the GUI application
    eframe::run_native(
        "Fluent Voice Room Visualizer",
        options,
        Box::new(move |cc| {
            let mut app = GuiApp::new(cc);

            // Set connection parameters if provided
            if let Some(url) = url {
                app.room_url = url;
            }
            if let Some(api_key) = api_key {
                app.api_key = api_key;
            }
            if let Some(api_secret) = api_secret {
                app.api_secret = api_secret;
            }
            app.room_name = room_name;
            app.participant_name = participant_name;

            // Auto-connect if requested
            if auto_connect
                && !app.room_url.is_empty()
                && !app.api_key.is_empty()
                && !app.api_secret.is_empty()
            {
                app.connect_to_room();
            }

            Box::new(app)
        }),
    )
    .map_err(|e| anyhow::anyhow!("Failed to run GUI: {}", e))
}
