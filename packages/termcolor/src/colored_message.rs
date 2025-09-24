//! Zero-allocation ColoredMessage builder for elegant structured terminal output
//!
//! Provides fluent API for building complex colored messages with perfect ergonomics.

use crate::theme::{
    SemanticColor, get_current_theme, write_colored, write_colored_bold,
    write_colored_italic,
};
use crate::{BufferedStandardStream, ColorChoice, WriteColor};
use std::io::{self, Write};

/// Zero-allocation message builder for structured colored output
#[derive(Debug)]
pub struct ColoredMessage {
    parts: Vec<MessagePart>,
}

/// Individual part of a colored message
#[derive(Debug, Clone)]
struct MessagePart {
    text: String,
    semantic: SemanticColor,
    style: MessageStyle,
}

/// Text styling options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageStyle {
    Normal,
    Bold,
    Italic,
    BoldItalic,
}

impl ColoredMessage {
    /// Create new empty colored message
    #[inline(always)]
    pub fn new() -> Self {
        Self { parts: Vec::new() }
    }

    /// Create new message builder (alias for new)
    #[inline(always)]
    pub fn builder() -> Self {
        Self::new()
    }

    /// Add text with primary color
    #[inline(always)]
    pub fn primary(mut self, text: impl Into<String>) -> Self {
        self.parts.push(MessagePart {
            text: text.into(),
            semantic: SemanticColor::Primary,
            style: MessageStyle::Normal,
        });
        self
    }

    /// Add text with secondary color
    #[inline(always)]
    pub fn secondary(mut self, text: impl Into<String>) -> Self {
        self.parts.push(MessagePart {
            text: text.into(),
            semantic: SemanticColor::Secondary,
            style: MessageStyle::Normal,
        });
        self
    }

    /// Add text with accent color
    #[inline(always)]
    pub fn accent(mut self, text: impl Into<String>) -> Self {
        self.parts.push(MessagePart {
            text: text.into(),
            semantic: SemanticColor::Accent,
            style: MessageStyle::Normal,
        });
        self
    }

    /// Add text with success color
    #[inline(always)]
    pub fn success(mut self, text: impl Into<String>) -> Self {
        self.parts.push(MessagePart {
            text: text.into(),
            semantic: SemanticColor::Success,
            style: MessageStyle::Normal,
        });
        self
    }

    /// Add text with warning color
    #[inline(always)]
    pub fn warning(mut self, text: impl Into<String>) -> Self {
        self.parts.push(MessagePart {
            text: text.into(),
            semantic: SemanticColor::Warning,
            style: MessageStyle::Normal,
        });
        self
    }

    /// Add text with error color
    #[inline(always)]
    pub fn error(mut self, text: impl Into<String>) -> Self {
        self.parts.push(MessagePart {
            text: text.into(),
            semantic: SemanticColor::Error,
            style: MessageStyle::Normal,
        });
        self
    }

    /// Add text with info color
    #[inline(always)]
    pub fn info(mut self, text: impl Into<String>) -> Self {
        self.parts.push(MessagePart {
            text: text.into(),
            semantic: SemanticColor::Info,
            style: MessageStyle::Normal,
        });
        self
    }

    /// Add text with debug color
    #[inline(always)]
    pub fn debug(mut self, text: impl Into<String>) -> Self {
        self.parts.push(MessagePart {
            text: text.into(),
            semantic: SemanticColor::Debug,
            style: MessageStyle::Normal,
        });
        self
    }

    /// Add text with primary text color
    #[inline(always)]
    pub fn text_primary(mut self, text: impl Into<String>) -> Self {
        self.parts.push(MessagePart {
            text: text.into(),
            semantic: SemanticColor::TextPrimary,
            style: MessageStyle::Normal,
        });
        self
    }

    /// Add text with secondary text color
    #[inline(always)]
    pub fn text_secondary(mut self, text: impl Into<String>) -> Self {
        self.parts.push(MessagePart {
            text: text.into(),
            semantic: SemanticColor::TextSecondary,
            style: MessageStyle::Normal,
        });
        self
    }

    /// Add text with muted text color
    #[inline(always)]
    pub fn text_muted(mut self, text: impl Into<String>) -> Self {
        self.parts.push(MessagePart {
            text: text.into(),
            semantic: SemanticColor::TextMuted,
            style: MessageStyle::Normal,
        });
        self
    }

    /// Add newline
    #[inline(always)]
    pub fn newline(mut self) -> Self {
        self.parts.push(MessagePart {
            text: "\n".to_string(),
            semantic: SemanticColor::TextPrimary,
            style: MessageStyle::Normal,
        });
        self
    }

    /// Add space
    #[inline(always)]
    pub fn space(mut self) -> Self {
        self.parts.push(MessagePart {
            text: " ".to_string(),
            semantic: SemanticColor::TextPrimary,
            style: MessageStyle::Normal,
        });
        self
    }

    /// Add tab
    #[inline(always)]
    pub fn tab(mut self) -> Self {
        self.parts.push(MessagePart {
            text: "\t".to_string(),
            semantic: SemanticColor::TextPrimary,
            style: MessageStyle::Normal,
        });
        self
    }

    /// Make the last added text bold
    #[inline(always)]
    pub fn bold(mut self) -> Self {
        if let Some(last) = self.parts.last_mut() {
            last.style = match last.style {
                MessageStyle::Normal => MessageStyle::Bold,
                MessageStyle::Italic => MessageStyle::BoldItalic,
                other => other,
            };
        }
        self
    }

    /// Make the last added text italic
    #[inline(always)]
    pub fn italic(mut self) -> Self {
        if let Some(last) = self.parts.last_mut() {
            last.style = match last.style {
                MessageStyle::Normal => MessageStyle::Italic,
                MessageStyle::Bold => MessageStyle::BoldItalic,
                other => other,
            };
        }
        self
    }

    /// Add custom text with semantic color and style
    #[inline(always)]
    pub fn custom(
        mut self,
        text: impl Into<String>,
        semantic: SemanticColor,
        style: MessageStyle,
    ) -> Self {
        self.parts.push(MessagePart { text: text.into(), semantic, style });
        self
    }

    /// Print message to stdout
    #[inline(always)]
    pub fn print(self) -> io::Result<()> {
        let mut stdout = BufferedStandardStream::stdout(ColorChoice::Auto);
        self.write_to(&mut stdout)?;
        stdout.flush()
    }

    /// Print message to stderr
    #[inline(always)]
    pub fn eprint(self) -> io::Result<()> {
        let mut stderr = BufferedStandardStream::stderr(ColorChoice::Auto);
        self.write_to(&mut stderr)?;
        stderr.flush()
    }

    /// Print message to stdout with newline
    #[inline(always)]
    pub fn println(self) -> io::Result<()> {
        let mut stdout = BufferedStandardStream::stdout(ColorChoice::Auto);
        self.write_to(&mut stdout)?;
        writeln!(stdout)?;
        stdout.flush()
    }

    /// Print message to stderr with newline
    #[inline(always)]
    pub fn eprintln(self) -> io::Result<()> {
        let mut stderr = BufferedStandardStream::stderr(ColorChoice::Auto);
        self.write_to(&mut stderr)?;
        writeln!(stderr)?;
        stderr.flush()
    }

    /// Write message to any WriteColor implementation
    #[inline(always)]
    pub fn write_to<W: WriteColor>(self, writer: &mut W) -> io::Result<()> {
        for part in self.parts {
            match part.style {
                MessageStyle::Normal => {
                    write_colored(writer, part.semantic, &part.text)?;
                }
                MessageStyle::Bold => {
                    write_colored_bold(writer, part.semantic, &part.text)?;
                }
                MessageStyle::Italic => {
                    write_colored_italic(writer, part.semantic, &part.text)?;
                }
                MessageStyle::BoldItalic => {
                    // Use bold+italic color spec
                    if let Some(theme) = get_current_theme() {
                        let mut spec = theme.spec(part.semantic);
                        spec.set_bold(true);
                        spec.set_italic(true);
                        writer.set_color(&spec)?;
                        write!(writer, "{}", part.text)?;
                        writer.reset()?;
                    } else {
                        write!(writer, "{}", part.text)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Convert to string (without colors)
    #[inline(always)]
    pub fn to_plain_string(self) -> String {
        self.parts
            .into_iter()
            .map(|part| part.text)
            .collect::<Vec<_>>()
            .join("")
    }
}

impl Default for ColoredMessage {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for ColoredMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for part in &self.parts {
            write!(f, "{}", part.text)?;
        }
        Ok(())
    }
}

/// Convenient constructors for common message patterns
impl ColoredMessage {
    /// Create a timestamp message
    #[inline(always)]
    pub fn timestamp(
        text: impl Into<String>,
        semantic: SemanticColor,
    ) -> Self {
        Self::new()
            .text_muted("[")
            .custom(text, semantic, MessageStyle::Normal)
            .text_muted("]")
            .space()
    }

    /// Create a level indicator message (INFO, WARN, ERROR, etc.)
    #[inline(always)]
    pub fn level(level: impl Into<String>, semantic: SemanticColor) -> Self {
        Self::new().custom(level, semantic, MessageStyle::Bold).space()
    }

    /// Create a structured log message with timestamp and level
    #[inline(always)]
    pub fn log(
        timestamp: impl Into<String>,
        level: impl Into<String>,
        message: impl Into<String>,
        level_semantic: SemanticColor,
    ) -> Self {
        let mut msg = Self::timestamp(timestamp, SemanticColor::TextMuted);
        let level_msg = Self::level(level, level_semantic);
        msg.parts.extend(level_msg.parts);
        msg.text_primary(message)
    }

    /// Create a progress message with percentage
    #[inline(always)]
    pub fn progress(
        current: usize,
        total: usize,
        message: impl Into<String>,
    ) -> Self {
        let percentage = (current as f64 / total as f64 * 100.0) as u8;
        let bar_width = 20;
        let filled = (current * bar_width / total).min(bar_width);
        let bar = "█".repeat(filled) + &"░".repeat(bar_width - filled);

        Self::new()
            .text_muted("[")
            .info(bar)
            .text_muted("] ")
            .accent(format!("{percentage}%"))
            .space()
            .text_primary(message)
    }

    /// Create a success checkmark message
    #[inline(always)]
    pub fn success_check(message: impl Into<String>) -> Self {
        Self::new().success("✅").space().text_primary(message)
    }

    /// Create a warning triangle message
    #[inline(always)]
    pub fn warning_triangle(message: impl Into<String>) -> Self {
        Self::new().warning("⚠️ ").space().text_primary(message)
    }

    /// Create an error X message
    #[inline(always)]
    pub fn error_x(message: impl Into<String>) -> Self {
        Self::new().error("❌").space().text_primary(message)
    }

    /// Create an info i message
    #[inline(always)]
    pub fn info_i(message: impl Into<String>) -> Self {
        Self::new().info("ℹ️ ").space().text_primary(message)
    }

    /// Create a Cyrup.ai branded header
    #[inline(always)]
    pub fn cyrup_header() -> Self {
        Self::new()
            .primary("🤖 CYRUP AI Assistant")
            .newline()
            .secondary("   Powered by Claude 3.5 Sonnet")
    }
}
