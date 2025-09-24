//! ANSI escape sequence functionality for terminal color output.
//!
//! This module provides low-level ANSI escape sequence generation functions
//! that can be used to write colored output to terminals that support ANSI
//! color codes.

use crate::{Color, ColorSpec};
use std::fmt;
use std::io;

/// Writes an ANSI escape sequence corresponding to the given color specification.
///
/// If `reset` is true, then the reset escape sequence will be written before
/// any color escape codes.
///
/// The caller must provide their own `IoWrite` to write to. Callers should
/// prefer higher level types in this crate, such as `StandardStream` or
/// `Buffer`.
pub fn ansi_spec<W: io::Write>(
    mut wtr: W,
    spec: &ColorSpec,
) -> io::Result<()> {
    if spec.reset() {
        write!(wtr, "\x1B[0m")?;
    }
    if spec.bold() {
        write!(wtr, "\x1B[1m")?;
    }
    if spec.dimmed() {
        write!(wtr, "\x1B[2m")?;
    }
    if spec.italic() {
        write!(wtr, "\x1B[3m")?;
    }
    if spec.underline() {
        write!(wtr, "\x1B[4m")?;
    }
    if spec.strikethrough() {
        write!(wtr, "\x1B[9m")?;
    }
    if let Some(c) = spec.fg() {
        ansi_color(&mut wtr, c, false)?;
    }
    if let Some(c) = spec.bg() {
        ansi_color(&mut wtr, c, true)?;
    }
    if spec.intense() && spec.fg().is_some() {
        write!(wtr, "\x1B[1m")?;
    }
    Ok(())
}

/// Writes an ANSI escape sequence corresponding to the given color.
///
/// If `bg` is true, then the color is treated as a background color.
/// Otherwise, it's treated as a foreground color.
///
/// The caller must provide their own `IoWrite` to write to. Callers should
/// prefer higher level types in this crate, such as `StandardStream` or
/// `Buffer`.
pub fn ansi_color<W: io::Write>(
    mut wtr: W,
    color: &Color,
    bg: bool,
) -> io::Result<()> {
    match *color {
        Color::Black => {
            if bg {
                write!(wtr, "\x1B[40m")
            } else {
                write!(wtr, "\x1B[30m")
            }
        }
        Color::Blue => {
            if bg {
                write!(wtr, "\x1B[44m")
            } else {
                write!(wtr, "\x1B[34m")
            }
        }
        Color::Green => {
            if bg {
                write!(wtr, "\x1B[42m")
            } else {
                write!(wtr, "\x1B[32m")
            }
        }
        Color::Red => {
            if bg {
                write!(wtr, "\x1B[41m")
            } else {
                write!(wtr, "\x1B[31m")
            }
        }
        Color::Cyan => {
            if bg {
                write!(wtr, "\x1B[46m")
            } else {
                write!(wtr, "\x1B[36m")
            }
        }
        Color::Magenta => {
            if bg {
                write!(wtr, "\x1B[45m")
            } else {
                write!(wtr, "\x1B[35m")
            }
        }
        Color::Yellow => {
            if bg {
                write!(wtr, "\x1B[43m")
            } else {
                write!(wtr, "\x1B[33m")
            }
        }
        Color::White => {
            if bg {
                write!(wtr, "\x1B[47m")
            } else {
                write!(wtr, "\x1B[37m")
            }
        }
        Color::Ansi256(n) => {
            if bg {
                write!(wtr, "\x1B[48;5;{n}m")
            } else {
                write!(wtr, "\x1B[38;5;{n}m")
            }
        }
        Color::Rgb(r, g, b) => {
            if bg {
                write!(wtr, "\x1B[48;2;{r};{g};{b}m")
            } else {
                write!(wtr, "\x1B[38;2;{r};{g};{b}m")
            }
        }
    }
}

/// A convenience function for creating a color specification that can be
/// formatted to an ANSI color string.
///
/// This is a shorthand for creating a `ColorSpec` with the given foreground
/// and background colors.
pub fn ansi_color_only(fg: Option<Color>, bg: Option<Color>) -> AnsiColor {
    AnsiColor { fg, bg }
}

/// A color specification that can be formatted to an ANSI color string.
///
/// This is created by the `ansi_color_only` function.
pub struct AnsiColor {
    fg: Option<Color>,
    bg: Option<Color>,
}

impl fmt::Display for AnsiColor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut buf = Vec::new();
        if let Some(ref c) = self.fg {
            ansi_color(&mut buf, c, false).map_err(|_| fmt::Error)?;
        }
        if let Some(ref c) = self.bg {
            ansi_color(&mut buf, c, true).map_err(|_| fmt::Error)?;
        }
        write!(f, "{}", String::from_utf8_lossy(&buf))
    }
}
