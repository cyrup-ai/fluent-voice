use clap::{Parser, Subcommand};

use crate::music::Note;

const HELP_TEMPLATE: &str = "{before-help}\
{name} {version} -- by {author}
{about}

{usage-heading} {usage}

{all-args}{after-help}
";

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None, help_template = HELP_TEMPLATE)]
pub struct ScopeArgs {
    #[clap(subcommand)]
    pub source: ScopeSource,

    #[command(flatten)]
    pub opts: SourceOptions,

    #[command(flatten)]
    pub ui: UiOptions,
}

#[derive(Debug, Clone, Parser)]
pub struct UiOptions {
    /// floating point vertical scale, from 0 to 1
    #[arg(short, long, value_name = "x", default_value_t = 1.0)]
    pub scale: f32,

    /// use vintage looking scatter mode instead of line mode
    #[arg(long, default_value_t = false)]
    pub scatter: bool,

    /// don't draw reference line
    #[arg(long, default_value_t = false)]
    pub no_reference: bool,

    /// hide UI and only draw waveforms
    #[arg(long, default_value_t = false)]
    pub no_ui: bool,

    /// don't use braille dots for drawing lines
    #[arg(long, default_value_t = false)]
    pub no_braille: bool,
}

#[derive(Debug, Clone, Subcommand)]
pub enum ScopeSource {
    /// use new experimental CPAL backend
    Audio {
        /// source device to attach to
        device: Option<String>,

        /// timeout (in seconds) waiting for audio stream
        #[arg(long, default_value_t = 60)]
        timeout: u64,

        /// just list available devices and quit
        #[arg(long, default_value_t = false)]
        list: bool,
    },
}

#[derive(Debug, Clone, Parser)]
pub struct SourceOptions {
    /// number of channels to open
    #[arg(long, value_name = "N", default_value_t = 2)]
    pub channels: usize,

    /// size of audio buffer, and width of scope
    #[arg(short, long, value_name = "SIZE", default_value_t = 2048)]
    pub buffer: u32,

    /// sample rate to use
    #[arg(long, value_name = "HZ", default_value_t = 48000)]
    pub sample_rate: u32,

    /// tune buffer size to be in tune with given note (overrides buffer option)
    #[arg(long, value_name = "NOTE")]
    pub tune: Option<String>,

    /// audio bit depth (8, 16, 24, or 32)
    #[arg(long, value_name = "BITS", default_value_t = 16)]
    pub bit_depth: u32,
}

impl SourceOptions {
    pub fn tune(&mut self) {
        if let Some(txt) = &self.tune {
            match txt.parse::<Note>() {
                Ok(note) => {
                    self.buffer = note.tune_buffer_size(self.sample_rate);
                    // Ensure buffer size is aligned to channel count and bit depth
                    let alignment = self.channels as u32 * (self.bit_depth / 8);
                    let remainder = self.buffer % alignment;
                    if remainder != 0 {
                        self.buffer += alignment - remainder;
                    }
                }
                Err(_) => {
                    eprintln!("[!] Unrecognized note '{txt}', ignoring option");
                }
            }
        }
    }
}
