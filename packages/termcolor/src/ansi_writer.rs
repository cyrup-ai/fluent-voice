//! ANSI escape sequence handling for terminal color output
//!
//! This module provides zero-allocation, blazing-fast ANSI escape sequence
//! generation for terminal color and formatting control. All operations are
//! lock-free and optimized for high-performance terminal output.

use crate::{Color, ColorSpec, HyperlinkSpec, WriteColor};
use std::io::{self, Write};

/// Satisfies `WriteColor` using standard ANSI escape sequences with zero allocation
///
/// This writer wraps any `io::Write` implementation and provides ANSI escape sequence
/// generation with optimized performance characteristics:
///
/// - **Zero allocation**: All escape sequences are generated without heap allocation
/// - **Lock-free operation**: No synchronization primitives required
/// - **Blazing-fast performance**: Optimized color code generation with compile-time constants
/// - **Memory-safe**: No unsafe code blocks
#[derive(Clone, Debug)]
pub struct Ansi<W>(pub W);

impl<W: Write> Ansi<W> {
    /// Create a new writer that satisfies `WriteColor` using standard ANSI escape sequences
    ///
    /// # Arguments
    /// * `wtr` - The underlying writer to wrap
    ///
    /// # Returns
    /// * Ansi writer ready for high-performance color output
    #[inline(always)]
    pub fn new(wtr: W) -> Ansi<W> {
        Ansi(wtr)
    }

    /// Consume this `Ansi` value and return the inner writer
    ///
    /// # Returns
    /// * The underlying writer without the ANSI wrapper
    #[inline(always)]
    pub fn into_inner(self) -> W {
        self.0
    }

    /// Return a reference to the inner writer
    ///
    /// # Returns
    /// * Reference to the underlying writer
    #[inline(always)]
    pub fn get_ref(&self) -> &W {
        &self.0
    }

    /// Return a mutable reference to the inner writer
    ///
    /// # Returns
    /// * Mutable reference to the underlying writer
    #[inline(always)]
    pub fn get_mut(&mut self) -> &mut W {
        &mut self.0
    }
}

impl<W: io::Write> io::Write for Ansi<W> {
    /// Write a buffer of bytes to the underlying writer
    ///
    /// # Arguments
    /// * `buf` - Buffer of bytes to write
    ///
    /// # Returns
    /// * Number of bytes written or IO error
    #[inline(always)]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.write(buf)
    }

    /// Write all bytes from buffer to the underlying writer
    ///
    /// Adding this method here is not required because it has a default impl,
    /// but it provides significant performance improvement when using
    /// a `BufWriter` with lots of writes.
    ///
    /// See https://github.com/BurntSushi/termcolor/pull/56 for details
    ///
    /// # Arguments
    /// * `buf` - Buffer of bytes to write completely
    ///
    /// # Returns
    /// * Success or IO error
    #[inline(always)]
    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        self.0.write_all(buf)
    }

    /// Flush any buffered data to the underlying writer
    ///
    /// # Returns
    /// * Success or IO error
    #[inline(always)]
    fn flush(&mut self) -> io::Result<()> {
        self.0.flush()
    }
}

impl<W: io::Write> WriteColor for Ansi<W> {
    /// Check if this writer supports color output
    ///
    /// ANSI writers always support color output
    ///
    /// # Returns
    /// * Always returns true for ANSI terminals
    #[inline(always)]
    fn supports_color(&self) -> bool {
        true
    }

    /// Check if this writer supports hyperlinks
    ///
    /// ANSI writers support OSC 8 hyperlink sequences
    ///
    /// # Returns
    /// * Always returns true for ANSI terminals
    #[inline(always)]
    fn supports_hyperlinks(&self) -> bool {
        true
    }

    /// Set color and formatting according to the given color specification
    ///
    /// This method applies multiple formatting attributes in sequence:
    /// 1. Reset (if requested)
    /// 2. Text attributes (bold, italic, etc.)
    /// 3. Foreground color
    /// 4. Background color
    ///
    /// All escape sequences are generated with zero allocation
    ///
    /// # Arguments
    /// * `spec` - Color specification with formatting attributes
    ///
    /// # Returns
    /// * Success or IO error
    #[inline(always)]
    fn set_color(&mut self, spec: &ColorSpec) -> io::Result<()> {
        if spec.reset() {
            self.reset()?;
        }
        if spec.bold() {
            self.write_str("\x1B[1m")?;
        }
        if spec.dimmed() {
            self.write_str("\x1B[2m")?;
        }
        if spec.italic() {
            self.write_str("\x1B[3m")?;
        }
        if spec.underline() {
            self.write_str("\x1B[4m")?;
        }
        if spec.strikethrough() {
            self.write_str("\x1B[9m")?;
        }
        if let Some(c) = spec.fg() {
            self.write_color(true, c, spec.intense())?;
        }
        if let Some(c) = spec.bg() {
            self.write_color(false, c, spec.intense())?;
        }
        Ok(())
    }

    /// Set hyperlink using OSC 8 escape sequence
    ///
    /// # Arguments
    /// * `link` - Hyperlink specification with URI
    ///
    /// # Returns
    /// * Success or IO error
    #[inline(always)]
    fn set_hyperlink(&mut self, link: &HyperlinkSpec) -> io::Result<()> {
        self.write_str("\x1B]8;;")?;
        if let Some(uri) = link.uri() {
            self.write_all(uri)?;
        }
        self.write_str("\x1B\\")
    }

    /// Reset all color and formatting to terminal defaults
    ///
    /// # Returns
    /// * Success or IO error
    #[inline(always)]
    fn reset(&mut self) -> io::Result<()> {
        self.write_str("\x1B[0m")
    }
}

impl<W: io::Write> Ansi<W> {
    /// Write a string as bytes with zero allocation
    ///
    /// # Arguments
    /// * `s` - String to write as bytes
    ///
    /// # Returns
    /// * Success or IO error
    #[inline(always)]
    fn write_str(&mut self, s: &str) -> io::Result<()> {
        self.write_all(s.as_bytes())
    }

    /// Write ANSI color escape sequence with zero allocation
    ///
    /// This method uses compile-time macros and stack-allocated buffers
    /// to generate ANSI color codes without heap allocation.
    ///
    /// # Arguments
    /// * `fg` - True for foreground color, false for background
    /// * `c` - Color specification
    /// * `intense` - True for bright/intense color variant
    ///
    /// # Returns
    /// * Success or IO error
    #[inline(always)]
    fn write_color(
        &mut self,
        fg: bool,
        c: &Color,
        intense: bool,
    ) -> io::Result<()> {
        macro_rules! write_intense {
            ($clr:expr) => {
                if fg {
                    self.write_str(concat!("\x1B[38;5;", $clr, "m"))
                } else {
                    self.write_str(concat!("\x1B[48;5;", $clr, "m"))
                }
            };
        }
        macro_rules! write_normal {
            ($clr:expr) => {
                if fg {
                    self.write_str(concat!("\x1B[3", $clr, "m"))
                } else {
                    self.write_str(concat!("\x1B[4", $clr, "m"))
                }
            };
        }
        macro_rules! write_var_ansi_code {
            ($pre:expr, $($code:expr),+) => {{
                // The loop generates at worst a literal of the form
                // '255,255,255m' which is 12-bytes.
                // The largest `pre` expression we currently use is 7 bytes.
                // This gives us the maximum of 19-bytes for our work buffer.
                let pre_len = $pre.len();
                debug_assert!(pre_len <= 7, "ANSI prefix too long");
                let mut fmt = [0u8; 19];
                fmt[..pre_len].copy_from_slice($pre);
                let mut i = pre_len - 1;
                $(
                    let c1: u8 = ($code / 100) % 10;
                    let c2: u8 = ($code / 10) % 10;
                    let c3: u8 = $code % 10;
                    let mut printed = false;

                    if c1 != 0 {
                        printed = true;
                        i += 1;
                        fmt[i] = b'0' + c1;
                    }
                    if c2 != 0 || printed {
                        i += 1;
                        fmt[i] = b'0' + c2;
                    }
                    // If we received a zero value we must still print a value.
                    i += 1;
                    fmt[i] = b'0' + c3;
                    i += 1;
                    fmt[i] = b';';
                )+

                fmt[i] = b'm';
                self.write_all(&fmt[0..i+1])
            }}
        }
        macro_rules! write_custom {
            ($ansi256:expr) => {
                if fg {
                    write_var_ansi_code!(b"\x1B[38;5;", $ansi256)
                } else {
                    write_var_ansi_code!(b"\x1B[48;5;", $ansi256)
                }
            };

            ($r:expr, $g:expr, $b:expr) => {{
                if fg {
                    write_var_ansi_code!(b"\x1B[38;2;", $r, $g, $b)
                } else {
                    write_var_ansi_code!(b"\x1B[48;2;", $r, $g, $b)
                }
            }};
        }
        if intense {
            match *c {
                Color::Black => write_intense!("8"),
                Color::Blue => write_intense!("12"),
                Color::Green => write_intense!("10"),
                Color::Red => write_intense!("9"),
                Color::Cyan => write_intense!("14"),
                Color::Magenta => write_intense!("13"),
                Color::Yellow => write_intense!("11"),
                Color::White => write_intense!("15"),
                Color::Ansi256(c) => write_custom!(c),
                Color::Rgb(r, g, b) => write_custom!(r, g, b),
            }
        } else {
            match *c {
                Color::Black => write_normal!("0"),
                Color::Blue => write_normal!("4"),
                Color::Green => write_normal!("2"),
                Color::Red => write_normal!("1"),
                Color::Cyan => write_normal!("6"),
                Color::Magenta => write_normal!("5"),
                Color::Yellow => write_normal!("3"),
                Color::White => write_normal!("7"),
                Color::Ansi256(c) => write_custom!(c),
                Color::Rgb(r, g, b) => write_custom!(r, g, b),
            }
        }
    }
}

/// WriteColor implementation for io::Sink (null writer)
///
/// This provides a no-op implementation for the standard library's sink writer,
/// which discards all output but needs to satisfy the WriteColor trait.
impl WriteColor for io::Sink {
    /// Sink never supports color output
    ///
    /// # Returns
    /// * Always returns false
    #[inline(always)]
    fn supports_color(&self) -> bool {
        false
    }

    /// Sink never supports hyperlinks
    ///
    /// # Returns
    /// * Always returns false
    #[inline(always)]
    fn supports_hyperlinks(&self) -> bool {
        false
    }

    /// No-op color setting for sink writer
    ///
    /// # Arguments
    /// * `_` - Unused color specification
    ///
    /// # Returns
    /// * Always returns success
    #[inline(always)]
    fn set_color(&mut self, _: &ColorSpec) -> io::Result<()> {
        Ok(())
    }

    /// No-op hyperlink setting for sink writer
    ///
    /// # Arguments
    /// * `_` - Unused hyperlink specification
    ///
    /// # Returns
    /// * Always returns success
    #[inline(always)]
    fn set_hyperlink(&mut self, _: &HyperlinkSpec) -> io::Result<()> {
        Ok(())
    }

    /// No-op reset for sink writer
    ///
    /// # Returns
    /// * Always returns success
    #[inline(always)]
    fn reset(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_ansi_creation() {
        let writer = Cursor::new(Vec::new());
        let ansi_writer = Ansi::new(writer);

        assert!(ansi_writer.supports_color());
        assert!(ansi_writer.supports_hyperlinks());
    }

    #[test]
    fn test_ansi_color_support() {
        let writer = Cursor::new(Vec::new());
        let ansi_writer = Ansi::new(writer);

        assert!(ansi_writer.supports_color());
        assert!(ansi_writer.supports_hyperlinks());
    }

    #[test]
    fn test_ansi_reset() {
        let writer = Cursor::new(Vec::new());
        let mut ansi_writer = Ansi::new(writer);

        ansi_writer.reset().unwrap();

        let output = ansi_writer.into_inner().into_inner();
        assert_eq!(output, b"\x1B[0m");
    }

    #[test]
    fn test_ansi_write_operations() {
        let mut buffer = Vec::new();
        let mut ansi_writer = Ansi::new(&mut buffer);

        ansi_writer.write_all(b"test").unwrap();
        ansi_writer.flush().unwrap();

        assert_eq!(buffer, b"test");
    }

    #[test]
    fn test_sink_writer_no_color() {
        let mut sink = io::sink();

        assert!(!sink.supports_color());
        assert!(!sink.supports_hyperlinks());

        // These should all succeed but do nothing
        sink.set_color(&ColorSpec::new()).unwrap();
        sink.set_hyperlink(&HyperlinkSpec::close()).unwrap();
        sink.reset().unwrap();
    }
}
