use anyhow::Result;
use clap::Parser;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::ExecutableCommand;
use ratatui::{backend::CrosstermBackend, Terminal};
use std::io;

use fluent_voice_animator::{
    app::App,
    cfg::{ScopeArgs, ScopeSource},
    audioio::cpal::DefaultAudioDeviceWithCPAL,
};

fn main() -> Result<()> {
    // Parse command line arguments
    let mut args = ScopeArgs::parse();
    args.opts.tune();
    
    // Clone values we need before moving args
    let opts = args.opts.clone();
    let ui = args.ui.clone();

    // Handle device listing first
    if let ScopeSource::Audio { list: true, .. } = &args.source {
        #[cfg(feature = "microphone")]
        {
            use cpal::traits::{DeviceTrait, HostTrait};
            let host = cpal::default_host();
            println!("Available audio input devices:");
            if let Ok(devices) = host.input_devices() {
                for device in devices {
                    let name = device.name().unwrap_or_else(|_| "Unknown Device".to_string());
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

    // Initialize audio source - move ownership to avoid lifetime issues
    let source = match args.source {
        ScopeSource::Audio { device, timeout, list: false } => {
            DefaultAudioDeviceWithCPAL::instantiate(
                device,
                opts.clone(),
                timeout,
            )?
        }
        _ => unreachable!("Audio source should be guaranteed by checks above")
    };

    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    stdout.execute(EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create and run the app
    let mut app = App::new(&ui, &opts);
    let result = app.run(source, &mut terminal);

    // Restore terminal
    disable_raw_mode()?;
    terminal.backend_mut().execute(LeaveAlternateScreen)?;

    result
}