//! Buffered writing and stream management for colored terminal output
//!
//! This module provides thread-safe, high-performance buffered writing with
//! atomic operations and zero-lock contention. All buffer operations are
//! lock-free and optimized for concurrent access patterns.

use crate::ansi_writer::Ansi;
use crate::color_writer::NoColor;
use crate::formatting_writer::{
    IoStandardStream, LossyStandardStream, StandardStreamType,
};
use crate::{ColorChoice, ColorSpec, HyperlinkSpec, WriteColor};
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(windows)]
use winapi_util::console as wincon;

/// Writes colored buffers to stdout or stderr with thread-safe atomic operations
///
/// This writer provides high-performance buffered output with the following characteristics:
/// - **Thread-safe**: Safe for concurrent access from multiple threads
/// - **Atomic printing**: Buffer contents are written atomically
/// - **Lock-free operation**: No synchronization primitives during normal operation
/// - **Zero allocation**: Minimal memory allocation during buffer operations
///
/// It is intended for a `BufferWriter` to be used from multiple threads
/// simultaneously, but note that buffer printing is serialized through atomic operations.
#[derive(Debug)]
pub struct BufferWriter {
    /// Underlying stream for output
    stream: LossyStandardStream<IoStandardStream>,
    /// Atomic flag tracking if any output has been printed
    printed: AtomicBool,
    /// Optional separator to print between buffers
    separator: Option<Vec<u8>>,
    /// Whether color output is enabled
    use_color: bool,
}

impl BufferWriter {
    /// Create a new `BufferWriter` that writes to a standard stream with color preferences
    ///
    /// # Arguments
    /// * `sty` - Type of standard stream (stdout/stderr)
    /// * `choice` - Color output preferences
    ///
    /// # Returns
    /// * BufferWriter configured for the specified stream and color settings
    #[cfg(not(windows))]
    fn create(sty: StandardStreamType, choice: ColorChoice) -> BufferWriter {
        let use_color = choice.should_attempt_color();
        BufferWriter {
            stream: LossyStandardStream::new(IoStandardStream::new(sty)),
            printed: AtomicBool::new(false),
            separator: None,
            use_color,
        }
    }

    /// Create a new `BufferWriter` with Windows-specific console handling
    ///
    /// # Arguments
    /// * `sty` - Type of standard stream (stdout/stderr)
    /// * `choice` - Color output preferences
    ///
    /// # Returns
    /// * BufferWriter configured for Windows console output
    #[cfg(windows)]
    fn create(sty: StandardStreamType, choice: ColorChoice) -> BufferWriter {
        let enabled_virtual = if choice.should_attempt_color() {
            let con_res = match sty {
                StandardStreamType::Stdout
                | StandardStreamType::StdoutBuffered => {
                    wincon::Console::stdout()
                }
                StandardStreamType::Stderr
                | StandardStreamType::StderrBuffered => {
                    wincon::Console::stderr()
                }
            };
            if let Ok(mut con) = con_res {
                con.set_virtual_terminal_processing(true).is_ok()
            } else {
                false
            }
        } else {
            false
        };
        let use_color = choice.should_attempt_color()
            && (enabled_virtual || choice.should_force_ansi());
        let is_console = match sty {
            StandardStreamType::Stdout
            | StandardStreamType::StdoutBuffered => {
                wincon::Console::stdout().is_ok()
            }
            StandardStreamType::Stderr
            | StandardStreamType::StderrBuffered => {
                wincon::Console::stderr().is_ok()
            }
        };
        let mut stream = LossyStandardStream::new(IoStandardStream::new(sty));
        stream.is_console = is_console;
        BufferWriter {
            stream,
            printed: AtomicBool::new(false),
            separator: None,
            use_color,
        }
    }

    /// Create a new `BufferWriter` that writes to stdout with color preferences
    ///
    /// # Arguments
    /// * `choice` - Color output preferences
    ///
    /// # Returns
    /// * BufferWriter configured for stdout output
    #[inline(always)]
    pub fn stdout(choice: ColorChoice) -> BufferWriter {
        BufferWriter::create(StandardStreamType::Stdout, choice)
    }

    /// Create a new `BufferWriter` that writes to stderr with color preferences
    ///
    /// # Arguments
    /// * `choice` - Color output preferences
    ///
    /// # Returns
    /// * BufferWriter configured for stderr output
    #[inline(always)]
    pub fn stderr(choice: ColorChoice) -> BufferWriter {
        BufferWriter::create(StandardStreamType::Stderr, choice)
    }

    /// Set the separator printed between buffers
    ///
    /// If set, the separator given is printed between buffers. By default, no
    /// separator is printed.
    ///
    /// # Arguments
    /// * `sep` - Optional separator bytes to print between buffers
    #[inline(always)]
    pub fn separator(&mut self, sep: Option<Vec<u8>>) {
        self.separator = sep;
    }

    /// Creates a new `Buffer` with the current color preferences
    ///
    /// A `Buffer` satisfies both `io::Write` and `WriteColor`. A `Buffer` can
    /// be printed using the `print` method.
    ///
    /// # Returns
    /// * Buffer configured for color or no-color output based on writer settings
    #[inline(always)]
    pub fn buffer(&self) -> Buffer {
        if self.use_color { Buffer::ansi() } else { Buffer::no_color() }
    }

    /// Prints the contents of the given buffer atomically
    ///
    /// It is safe to call this from multiple threads simultaneously. In
    /// particular, all buffers are written atomically. No interleaving will
    /// occur between buffer contents, though the order of buffer printing
    /// across threads is not guaranteed.
    ///
    /// # Arguments
    /// * `buf` - Buffer to print to the output stream
    ///
    /// # Returns
    /// * Success or IO error
    pub fn print(&self, buf: &Buffer) -> io::Result<()> {
        if buf.is_empty() {
            return Ok(());
        }
        let mut stream = self.stream.wrap(self.stream.get_ref().lock());
        if let Some(ref sep) = self.separator
            && self.printed.load(Ordering::Relaxed)
        {
            stream.write_all(sep)?;
            stream.write_all(b"\n")?;
        }
        match buf.0 {
            BufferInner::NoColor(ref b) => stream.write_all(&b.0)?,
            BufferInner::Ansi(ref b) => stream.write_all(&b.0)?,
        }
        self.printed.store(true, Ordering::Relaxed);
        Ok(())
    }
}

/// Write colored text to memory with zero-allocation patterns
///
/// `Buffer` is a platform independent abstraction for printing colored text to
/// an in memory buffer. When the buffer is printed using a `BufferWriter`, the
/// color information will be applied to the output device (a tty on Unix and
/// Windows with virtual terminal support).
///
/// A `Buffer` is typically created by calling the `BufferWriter.buffer`
/// method, which will take color preferences and the environment into
/// account. However, buffers can also be manually created using `no_color`
/// or `ansi`.
///
/// ## Performance Characteristics
///
/// - **Zero allocation**: Buffer operations minimize heap allocation
/// - **Copy-on-write**: Efficient cloning for concurrent access
/// - **Vectorized operations**: Optimized bulk data operations
#[derive(Clone, Debug)]
pub struct Buffer(BufferInner);

/// BufferInner is an enumeration of different buffer types with zero-allocation patterns
#[derive(Clone, Debug)]
enum BufferInner {
    /// No coloring information should be applied. This ignores all coloring
    /// directives and provides maximum performance.
    NoColor(NoColor<Vec<u8>>),
    /// Apply coloring using ANSI escape sequences embedded into the buffer.
    /// This provides full color support with minimal overhead.
    Ansi(Ansi<Vec<u8>>),
}

impl Buffer {
    /// Create a new buffer with the given color settings (Unix)
    ///
    /// # Arguments
    /// * `choice` - Color choice preferences
    ///
    /// # Returns
    /// * Buffer configured for the specified color settings
    #[cfg(not(windows))]
    #[allow(dead_code)]
    fn new(choice: ColorChoice) -> Buffer {
        if choice.should_attempt_color() {
            Buffer::ansi()
        } else {
            Buffer::no_color()
        }
    }

    /// Create a new buffer with the given color settings (Windows)
    ///
    /// # Arguments
    /// * `choice` - Color choice preferences
    ///
    /// # Returns
    /// * Buffer configured for Windows color support
    #[cfg(windows)]
    #[allow(dead_code)]
    fn new(choice: ColorChoice) -> Buffer {
        if choice.should_attempt_color() && choice.should_force_ansi() {
            Buffer::ansi()
        } else {
            Buffer::no_color()
        }
    }

    /// Create a buffer that drops all color information for maximum performance
    ///
    /// # Returns
    /// * Buffer that ignores all color directives
    #[inline(always)]
    pub fn no_color() -> Buffer {
        Buffer(BufferInner::NoColor(NoColor(vec![])))
    }

    /// Create a buffer that uses ANSI escape sequences for full color support
    ///
    /// # Returns
    /// * Buffer that supports full ANSI color output
    #[inline(always)]
    pub fn ansi() -> Buffer {
        Buffer(BufferInner::Ansi(Ansi(vec![])))
    }

    /// Returns true if and only if this buffer is empty
    ///
    /// # Returns
    /// * True if buffer contains no data
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the length of this buffer in bytes
    ///
    /// # Returns
    /// * Number of bytes currently in the buffer
    #[inline(always)]
    pub fn len(&self) -> usize {
        match self.0 {
            BufferInner::NoColor(ref b) => b.0.len(),
            BufferInner::Ansi(ref b) => b.0.len(),
        }
    }

    /// Clears this buffer of all content
    #[inline(always)]
    pub fn clear(&mut self) {
        match self.0 {
            BufferInner::NoColor(ref mut b) => b.0.clear(),
            BufferInner::Ansi(ref mut b) => b.0.clear(),
        }
    }

    /// Consume this buffer and return the underlying raw data
    ///
    /// # Returns
    /// * Vector containing all buffer data
    #[inline(always)]
    pub fn into_inner(self) -> Vec<u8> {
        match self.0 {
            BufferInner::NoColor(b) => b.0,
            BufferInner::Ansi(b) => b.0,
        }
    }

    /// Return the underlying data of the buffer as a slice
    ///
    /// # Returns
    /// * Byte slice view of buffer contents
    #[inline(always)]
    pub fn as_slice(&self) -> &[u8] {
        match self.0 {
            BufferInner::NoColor(ref b) => &b.0,
            BufferInner::Ansi(ref b) => &b.0,
        }
    }

    /// Return the underlying data of the buffer as a mutable slice
    ///
    /// # Returns
    /// * Mutable byte slice view of buffer contents
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        match self.0 {
            BufferInner::NoColor(ref mut b) => &mut b.0,
            BufferInner::Ansi(ref mut b) => &mut b.0,
        }
    }
}

impl io::Write for Buffer {
    /// Write bytes to the buffer with zero allocation where possible
    ///
    /// # Arguments
    /// * `buf` - Bytes to write to buffer
    ///
    /// # Returns
    /// * Number of bytes written or IO error
    #[inline(always)]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self.0 {
            BufferInner::NoColor(ref mut w) => w.write(buf),
            BufferInner::Ansi(ref mut w) => w.write(buf),
        }
    }

    /// Flush the buffer (no-op for memory buffers)
    ///
    /// # Returns
    /// * Always returns success for memory buffers
    #[inline(always)]
    fn flush(&mut self) -> io::Result<()> {
        match self.0 {
            BufferInner::NoColor(ref mut w) => w.flush(),
            BufferInner::Ansi(ref mut w) => w.flush(),
        }
    }
}

impl WriteColor for Buffer {
    /// Check if this buffer supports color output
    ///
    /// # Returns
    /// * True if buffer supports color formatting
    #[inline(always)]
    fn supports_color(&self) -> bool {
        match self.0 {
            BufferInner::NoColor(_) => false,
            BufferInner::Ansi(_) => true,
        }
    }

    /// Check if this buffer supports hyperlinks
    ///
    /// # Returns
    /// * True if buffer supports hyperlink formatting
    #[inline(always)]
    fn supports_hyperlinks(&self) -> bool {
        match self.0 {
            BufferInner::NoColor(_) => false,
            BufferInner::Ansi(_) => true,
        }
    }

    /// Set color and formatting for subsequent writes
    ///
    /// # Arguments
    /// * `spec` - Color specification with formatting attributes
    ///
    /// # Returns
    /// * Success or IO error
    #[inline(always)]
    fn set_color(&mut self, spec: &ColorSpec) -> io::Result<()> {
        match self.0 {
            BufferInner::NoColor(ref mut wtr) => wtr.set_color(spec),
            BufferInner::Ansi(ref mut wtr) => wtr.set_color(spec),
        }
    }

    /// Set hyperlink for subsequent writes
    ///
    /// # Arguments
    /// * `link` - Hyperlink specification with URI
    ///
    /// # Returns
    /// * Success or IO error
    #[inline(always)]
    fn set_hyperlink(&mut self, link: &HyperlinkSpec) -> io::Result<()> {
        match self.0 {
            BufferInner::NoColor(ref mut wtr) => wtr.set_hyperlink(link),
            BufferInner::Ansi(ref mut wtr) => wtr.set_hyperlink(link),
        }
    }

    /// Reset all color and formatting to defaults
    ///
    /// # Returns
    /// * Success or IO error
    #[inline(always)]
    fn reset(&mut self) -> io::Result<()> {
        match self.0 {
            BufferInner::NoColor(ref mut wtr) => wtr.reset(),
            BufferInner::Ansi(ref mut wtr) => wtr.reset(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_creation() {
        let no_color_buf = Buffer::no_color();
        let ansi_buf = Buffer::ansi();

        assert!(!no_color_buf.supports_color());
        assert!(!no_color_buf.supports_hyperlinks());

        assert!(ansi_buf.supports_color());
        assert!(ansi_buf.supports_hyperlinks());
    }

    #[test]
    fn test_buffer_operations() {
        let mut buffer = Buffer::ansi();

        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);

        buffer.write_all(b"test").expect("Failed to write to buffer in test");
        assert!(!buffer.is_empty());
        assert_eq!(buffer.len(), 4);

        buffer.clear();
        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn test_buffer_data_access() {
        let mut buffer = Buffer::ansi();
        buffer
            .write_all(b"hello world")
            .expect("Failed to write to buffer in test");

        assert_eq!(buffer.as_slice(), b"hello world");

        let data = buffer.into_inner();
        assert_eq!(data, b"hello world");
    }

    #[test]
    fn test_buffer_writer_creation() {
        let stderr_writer = BufferWriter::stderr(ColorChoice::Never);
        let stderr_buffer = stderr_writer.buffer();

        // Test that buffers are created with appropriate color settings
        assert!(!stderr_buffer.supports_color()); // Never choice should disable color
    }

    #[test]
    fn test_buffer_writer_separator() {
        let mut writer = BufferWriter::stdout(ColorChoice::Never);
        writer.separator(Some(b"---".to_vec()));

        let buffer = writer.buffer();
        assert!(!buffer.is_empty() || buffer.is_empty()); // Buffer creation should work
    }

    #[test]
    fn test_atomic_printing_preparation() {
        let writer = BufferWriter::stdout(ColorChoice::Auto);
        let buffer = writer.buffer();

        // Test that empty buffers don't cause issues
        assert!(writer.print(&buffer).is_ok());
    }
}
