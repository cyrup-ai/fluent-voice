use clap::Parser;
use crossterm::{
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{Terminal, backend::CrosstermBackend};
use scope::app::App;
use scope::cfg::{ScopeArgs, ScopeSource};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = ScopeArgs::parse();
    args.opts.tune();

    let source = match args.source {
        ScopeSource::Audio {
            device,
            timeout,
            list,
        } => {
            if list {
                use cpal::traits::{DeviceTrait, HostTrait};
                let host = cpal::default_host();
                for dev in host.input_devices().unwrap() {
                    println!("> {}", dev.name().unwrap());
                    for config in dev.supported_input_configs().unwrap() {
                        let (min_buf, max_buf) = match config.buffer_size() {
                            cpal::SupportedBufferSize::Range { min, max } => (*min, *max),
                            cpal::SupportedBufferSize::Unknown => (0, 0),
                        };
                        println!(
                            "  + {}ch {}-{}hz {}-{}buf ({})",
                            config.channels(),
                            config.min_sample_rate().0,
                            config.max_sample_rate().0,
                            min_buf,
                            max_buf,
                            config.sample_format()
                        );
                    }
                }
                return Ok(());
            }
            scope::input::cpal::DefaultAudioDeviceWithCPAL::instantiate(
                device.as_deref(),
                &args.opts,
                timeout,
            )?
        }
    };

    let mut app = App::new(&args.ui, &args.opts);

    // setup terminal
    enable_raw_mode()?;
    execute!(std::io::stdout(), EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(std::io::stdout()))?;
    terminal.hide_cursor()?;

    let result = app.run(source, &mut terminal);

    // restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen,)?;
    terminal.show_cursor()?;

    if let Err(e) = result {
        eprintln!("[!] Error executing app: {:?}", e);
    }

    Ok(())
}
