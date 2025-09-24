//! Termcolor crate for cross-platform colored terminal output with Cyrup.ai theming

pub mod ansi;
pub mod colored_message;
pub mod macros;
pub mod theme;
mod traits;
mod types;
mod writers;

// Writer modules
mod ansi_writer;
mod buffer_writer;
mod color_writer;
mod formatting_writer;

// Re-export core traits and types
pub use ansi::{AnsiColor, ansi_color, ansi_color_only, ansi_spec};
pub use traits::WriteColor;
pub use types::{
    Color, ColorChoice, ColorChoiceParseError, ColorSpec, ColorSpecParseError,
    HyperlinkSpec, ParseColorError,
};
pub use writers::{
    Ansi, Buffer, BufferWriter, BufferedStandardStream, NoColor,
    StandardStream, StandardStreamLock, StringWriter, TermString,
};

// Re-export theme system
pub use theme::{
    CyrupTheme, CyrupThemeBuilder, SemanticColor, ThemeConfig,
    get_current_theme, get_global_theme, set_global_theme,
    with_temporary_theme, write_colored, write_colored_bold,
    write_colored_italic,
};

// Re-export colored message builder
pub use colored_message::{ColoredMessage, MessageStyle};

// Re-export terminal color detection functions
pub use color_writer::color_choice_ext::ColorChoiceExt;
pub use color_writer::color_support::{
    color_count, detect_color_support, should_use_color, supports_truecolor,
};
