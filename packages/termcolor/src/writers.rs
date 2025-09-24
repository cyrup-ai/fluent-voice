//! High-performance terminal color writers with modular architecture
//!
//! This module provides blazing-fast, zero-allocation writers for colored terminal output
//! with excellent separation of concerns and production-ready performance characteristics.
//!
//! ## Architecture
//!
//! The writers module is organized into specialized submodules:
//!
//! - **ansi_writer**: ANSI escape sequence handling with zero allocation
//! - **buffer_writer**: Buffered writing and stream management
//! - **color_writer**: Color specification and WriteColor implementations  
//! - **formatting_writer**: Text formatting and cross-platform stream management
//!
//! ## Usage
//!
//! ```rust
//! use termcolor::{StandardStream, ColorChoice, ColorSpec, Color, WriteColor};
//! use std::io::Write;
//!
//! let mut stdout = StandardStream::stdout(ColorChoice::Always);
//! stdout.set_color(ColorSpec::new().set_fg(Some(Color::Red)))?;
//! writeln!(&mut stdout, "This text is red!")?;
//! stdout.reset()?;
//! ```
//!
//! ## Performance Characteristics
//!
//! - **Zero allocation**: Optimized patterns minimize heap allocation
//! - **Lock-free operation**: No synchronization primitives during normal operation
//! - **Cross-platform**: Handles Windows console and Unix terminal differences
//! - **Blazing-fast**: Optimized write paths for maximum throughput

// Re-export all the modular components for backward compatibility
pub use crate::ansi_writer::Ansi;
pub use crate::buffer_writer::{Buffer, BufferWriter};
pub use crate::color_writer::NoColor;
pub use crate::formatting_writer::{
    BufferedStandardStream, StandardStream, StandardStreamLock,
};

// Additional utility types for compatibility
use crate::{ColorSpec, HyperlinkSpec, WriteColor};
use std::io;

/// WriteColor implementation for Vec<u8> (in-memory buffer)
///
/// This provides a no-op WriteColor implementation for byte vectors,
/// which is useful for testing and in-memory operations.
impl WriteColor for Vec<u8> {
    /// Vec<u8> does not support color output
    ///
    /// # Returns
    /// * Always returns false
    #[inline(always)]
    fn supports_color(&self) -> bool {
        false
    }

    /// Vec<u8> does not support hyperlinks
    ///
    /// # Returns
    /// * Always returns false
    #[inline(always)]
    fn supports_hyperlinks(&self) -> bool {
        false
    }

    /// No-op color setting for Vec<u8>
    ///
    /// # Arguments
    /// * `_spec` - Unused color specification
    ///
    /// # Returns
    /// * Always returns success
    #[inline(always)]
    fn set_color(&mut self, _spec: &ColorSpec) -> io::Result<()> {
        Ok(())
    }

    /// No-op hyperlink setting for Vec<u8>
    ///
    /// # Arguments
    /// * `_link` - Unused hyperlink specification
    ///
    /// # Returns
    /// * Always returns success
    #[inline(always)]
    fn set_hyperlink(&mut self, _link: &HyperlinkSpec) -> io::Result<()> {
        Ok(())
    }

    /// No-op reset for Vec<u8>
    ///
    /// # Returns
    /// * Always returns success
    #[inline(always)]
    fn reset(&mut self) -> io::Result<()> {
        Ok(())
    }

    /// Vec<u8> is not synchronous output
    ///
    /// # Returns
    /// * Always returns false
    #[inline(always)]
    fn is_synchronous(&self) -> bool {
        false
    }
}

/// A wrapper for String that implements both Write and WriteColor
///
/// This fixes compatibility where libraries expect WriteColor on string-like writers.
/// The StringWriter collects written data into an internal String buffer without any
/// color formatting (colors are ignored for maximum performance).
///
/// ## Performance Features
///
/// - **Zero allocation**: Efficient string building with minimal allocation
/// - **UTF-8 validation**: Ensures all written data is valid UTF-8
/// - **No-op color operations**: Maximum performance by ignoring color directives
#[derive(Debug, Default)]
pub struct StringWriter {
    /// The internal string buffer that collects written data
    pub inner: String,
}

impl StringWriter {
    /// Creates a new empty StringWriter
    ///
    /// # Returns
    /// * Empty StringWriter ready for writing
    #[inline(always)]
    pub fn new() -> Self {
        Self { inner: String::new() }
    }

    /// Creates a new StringWriter with the specified capacity
    ///
    /// # Arguments
    /// * `capacity` - Initial capacity for the internal string buffer
    ///
    /// # Returns
    /// * StringWriter with pre-allocated capacity
    #[inline(always)]
    pub fn with_capacity(capacity: usize) -> Self {
        Self { inner: String::with_capacity(capacity) }
    }

    /// Consumes the StringWriter and returns the internal String
    ///
    /// # Returns
    /// * Internal string containing all written data
    #[inline(always)]
    pub fn into_string(self) -> String {
        self.inner
    }

    /// Returns a string slice of the internal buffer
    ///
    /// # Returns
    /// * String slice view of the written data
    #[inline(always)]
    pub fn as_str(&self) -> &str {
        &self.inner
    }
}

impl io::Write for StringWriter {
    /// Write UTF-8 bytes to the internal string buffer
    ///
    /// # Arguments
    /// * `buf` - Buffer of UTF-8 bytes to write
    ///
    /// # Returns
    /// * Number of bytes written or UTF-8 validation error
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match std::str::from_utf8(buf) {
            Ok(s) => {
                self.inner.push_str(s);
                Ok(buf.len())
            }
            Err(_) => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid UTF-8 sequence in StringWriter",
            )),
        }
    }

    /// Flush the StringWriter (no-op for string buffers)
    ///
    /// # Returns
    /// * Always returns success
    #[inline(always)]
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl WriteColor for StringWriter {
    /// StringWriter does not support color output
    ///
    /// # Returns
    /// * Always returns false
    #[inline(always)]
    fn supports_color(&self) -> bool {
        false
    }

    /// StringWriter does not support hyperlinks
    ///
    /// # Returns
    /// * Always returns false
    #[inline(always)]
    fn supports_hyperlinks(&self) -> bool {
        false
    }

    /// No-op color setting for maximum performance
    ///
    /// # Arguments
    /// * `_spec` - Unused color specification
    ///
    /// # Returns
    /// * Always returns success
    #[inline(always)]
    fn set_color(&mut self, _spec: &ColorSpec) -> io::Result<()> {
        Ok(())
    }

    /// No-op hyperlink setting for maximum performance
    ///
    /// # Arguments
    /// * `_link` - Unused hyperlink specification
    ///
    /// # Returns
    /// * Always returns success
    #[inline(always)]
    fn set_hyperlink(&mut self, _link: &HyperlinkSpec) -> io::Result<()> {
        Ok(())
    }

    /// No-op reset for maximum performance
    ///
    /// # Returns
    /// * Always returns success
    #[inline(always)]
    fn reset(&mut self) -> io::Result<()> {
        Ok(())
    }

    /// StringWriter is not synchronous
    ///
    /// # Returns
    /// * Always returns false
    #[inline(always)]
    fn is_synchronous(&self) -> bool {
        false
    }
}

/// A String wrapper that implements both `io::Write` and `WriteColor`
///
/// This type provides backward compatibility with libraries that expect
/// string-like writers to implement `WriteColor`. It's a zero-cost wrapper around
/// `String` that adds the necessary trait implementations.
///
/// ## Performance Features
///
/// - **Zero-cost wrapper**: No runtime overhead over String
/// - **Transparent representation**: Direct memory layout compatibility
/// - **Full String compatibility**: All standard String operations available
#[derive(Debug, Default, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct TermString(pub String);

impl TermString {
    /// Creates a new empty `TermString`
    ///
    /// # Returns
    /// * Empty TermString ready for writing
    #[inline(always)]
    pub fn new() -> Self {
        Self(String::new())
    }

    /// Creates a new `TermString` with the specified capacity
    ///
    /// # Arguments
    /// * `capacity` - Initial capacity for the internal string
    ///
    /// # Returns
    /// * TermString with pre-allocated capacity
    #[inline(always)]
    pub fn with_capacity(capacity: usize) -> Self {
        Self(String::with_capacity(capacity))
    }

    /// Consumes the `TermString` and returns the inner `String`
    ///
    /// # Returns
    /// * Inner String containing all data
    #[inline(always)]
    pub fn into_inner(self) -> String {
        self.0
    }

    /// Returns a string slice of the `TermString` contents
    ///
    /// # Returns
    /// * String slice view of the contents
    #[inline(always)]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Appends a string slice to the end of this `TermString`
    ///
    /// # Arguments
    /// * `s` - String slice to append
    #[inline(always)]
    pub fn push_str(&mut self, s: &str) {
        self.0.push_str(s)
    }
}

impl From<String> for TermString {
    /// Convert a String to TermString with zero cost
    ///
    /// # Arguments
    /// * `s` - String to convert
    ///
    /// # Returns
    /// * TermString wrapping the input string
    #[inline(always)]
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<TermString> for String {
    /// Convert a TermString to String with zero cost
    ///
    /// # Arguments
    /// * `ts` - TermString to convert
    ///
    /// # Returns
    /// * Inner String
    #[inline(always)]
    fn from(ts: TermString) -> Self {
        ts.0
    }
}

impl AsRef<str> for TermString {
    /// Get string slice reference
    ///
    /// # Returns
    /// * String slice view
    #[inline(always)]
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for TermString {
    /// Format the TermString for display
    ///
    /// # Arguments
    /// * `f` - Formatter
    ///
    /// # Returns
    /// * Format result
    #[inline(always)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl io::Write for TermString {
    /// Write UTF-8 bytes to the TermString
    ///
    /// # Arguments
    /// * `buf` - Buffer of UTF-8 bytes to write
    ///
    /// # Returns
    /// * Number of bytes written or UTF-8 validation error
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match std::str::from_utf8(buf) {
            Ok(s) => {
                self.0.push_str(s);
                Ok(buf.len())
            }
            Err(_) => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid UTF-8 sequence in TermString",
            )),
        }
    }

    /// Flush the TermString (no-op for string buffers)
    ///
    /// # Returns
    /// * Always returns success
    #[inline(always)]
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl WriteColor for TermString {
    /// TermString does not support color output
    ///
    /// # Returns
    /// * Always returns false
    #[inline(always)]
    fn supports_color(&self) -> bool {
        false
    }

    /// TermString does not support hyperlinks
    ///
    /// # Returns
    /// * Always returns false
    #[inline(always)]
    fn supports_hyperlinks(&self) -> bool {
        false
    }

    /// No-op color setting for maximum performance
    ///
    /// # Arguments
    /// * `_spec` - Unused color specification
    ///
    /// # Returns
    /// * Always returns success
    #[inline(always)]
    fn set_color(&mut self, _spec: &ColorSpec) -> io::Result<()> {
        Ok(())
    }

    /// No-op hyperlink setting for maximum performance
    ///
    /// # Arguments
    /// * `_link` - Unused hyperlink specification
    ///
    /// # Returns
    /// * Always returns success
    #[inline(always)]
    fn set_hyperlink(&mut self, _link: &HyperlinkSpec) -> io::Result<()> {
        Ok(())
    }

    /// No-op reset for maximum performance
    ///
    /// # Returns
    /// * Always returns success
    #[inline(always)]
    fn reset(&mut self) -> io::Result<()> {
        Ok(())
    }

    /// TermString is not synchronous
    ///
    /// # Returns
    /// * Always returns false
    #[inline(always)]
    fn is_synchronous(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ColorChoice, ColorSpec};
    use std::io::Write;

    #[test]
    fn test_modular_structure_integration() {
        // Test that all components work together
        let stdout = StandardStream::stdout(ColorChoice::Never);
        let buffer = BufferWriter::stdout(ColorChoice::Always).buffer();

        // Should be able to use all writers
        assert!(!stdout.supports_color()); // Never choice
        assert!(buffer.supports_color()); // Always choice
    }

    #[test]
    fn test_string_writer() {
        let mut writer = StringWriter::new();

        assert!(!writer.supports_color());
        assert!(!writer.supports_hyperlinks());

        writer.write_all(b"hello world").unwrap();
        writer.flush().unwrap();

        assert_eq!(writer.as_str(), "hello world");
        assert_eq!(writer.into_string(), "hello world");
    }

    #[test]
    fn test_term_string() {
        let mut term_string = TermString::new();

        assert!(!term_string.supports_color());
        assert!(!term_string.supports_hyperlinks());

        term_string.write_all(b"test").unwrap();
        term_string.push_str(" string");

        assert_eq!(term_string.as_str(), "test string");
        assert_eq!(String::from(term_string), "test string");
    }

    #[test]
    fn test_vec_u8_write_color() {
        let mut vec = Vec::new();

        assert!(!vec.supports_color());
        assert!(!vec.supports_hyperlinks());

        // All operations should be no-ops
        vec.set_color(&ColorSpec::new()).unwrap();
        vec.set_hyperlink(&crate::HyperlinkSpec::close()).unwrap();
        vec.reset().unwrap();

        vec.write_all(b"data").unwrap();
        assert_eq!(vec, b"data");
    }

    #[test]
    fn test_backward_compatibility() {
        // Test that all original APIs are still available
        let _stdout = StandardStream::stdout(ColorChoice::Auto);
        let _stderr = StandardStream::stderr(ColorChoice::Always);
        let _buffered = BufferedStandardStream::stdout(ColorChoice::Never);
        let _buffer_writer = BufferWriter::stdout(ColorChoice::Auto);
        let _buffer = Buffer::ansi();
        let _no_color_buffer = Buffer::no_color();

        // Test writer types
        let _ansi_writer = Ansi::new(std::io::sink());
        let _no_color_writer = NoColor::new(std::io::sink());
    }

    #[test]
    fn test_utf8_validation() {
        let mut string_writer = StringWriter::new();
        let mut term_string = TermString::new();

        // Valid UTF-8 should work
        assert!(string_writer.write(b"hello").is_ok());
        assert!(term_string.write(b"world").is_ok());

        // Invalid UTF-8 should fail
        assert!(string_writer.write(b"\xFF\xFE").is_err());
        assert!(term_string.write(b"\xFF\xFE").is_err());
    }
}
