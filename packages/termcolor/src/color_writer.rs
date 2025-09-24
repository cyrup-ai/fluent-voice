//! Color specification and WriteColor implementations for high-performance output
//!
//! This module provides the core color writing logic with zero-allocation patterns
//! and blazing-fast performance characteristics. All implementations are lock-free
//! and optimized for high-throughput terminal output.

use crate::{ColorSpec, HyperlinkSpec, WriteColor};
use std::io::{self, Write};

/// Satisfies `WriteColor` but ignores all color options for maximum performance
///
/// This writer provides a high-performance path for applications that need
/// the `WriteColor` interface but want to disable all color output:
///
/// - **Zero overhead**: All color operations are no-ops with zero cost
/// - **Maximum throughput**: No color processing overhead
/// - **Memory efficient**: No color state or escape sequence generation
/// - **Lock-free**: No synchronization required
#[derive(Clone, Debug)]
pub struct NoColor<W>(pub W);

impl<W: Write> NoColor<W> {
    /// Create a new writer that satisfies `WriteColor` but drops all color information
    ///
    /// This provides maximum performance for applications that want to use the
    /// `WriteColor` trait but disable all color output.
    ///
    /// # Arguments
    /// * `wtr` - The underlying writer to wrap
    ///
    /// # Returns
    /// * NoColor writer that ignores all color directives
    #[inline(always)]
    pub fn new(wtr: W) -> NoColor<W> {
        NoColor(wtr)
    }

    /// Consume this `NoColor` value and return the inner writer
    ///
    /// # Returns
    /// * The underlying writer without the NoColor wrapper
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

impl<W: io::Write> io::Write for NoColor<W> {
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

    /// Flush any buffered data to the underlying writer
    ///
    /// # Returns
    /// * Success or IO error
    #[inline(always)]
    fn flush(&mut self) -> io::Result<()> {
        self.0.flush()
    }
}

impl<W: io::Write> WriteColor for NoColor<W> {
    /// Check if this writer supports color output
    ///
    /// NoColor writers never support color output for maximum performance
    ///
    /// # Returns
    /// * Always returns false
    #[inline(always)]
    fn supports_color(&self) -> bool {
        false
    }

    /// Check if this writer supports hyperlinks
    ///
    /// NoColor writers never support hyperlinks for maximum performance
    ///
    /// # Returns
    /// * Always returns false
    #[inline(always)]
    fn supports_hyperlinks(&self) -> bool {
        false
    }

    /// Set color and formatting (no-op for maximum performance)
    ///
    /// This method does nothing and returns immediately for zero overhead
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

    /// Set hyperlink (no-op for maximum performance)
    ///
    /// This method does nothing and returns immediately for zero overhead
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

    /// Reset color and formatting (no-op for maximum performance)
    ///
    /// This method does nothing and returns immediately for zero overhead
    ///
    /// # Returns
    /// * Always returns success
    #[inline(always)]
    fn reset(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Color support detection utilities for optimal performance
pub mod color_support {
    use crate::ColorChoice;

    /// Detect if color output should be enabled based on environment
    ///
    /// This function provides fast color support detection with zero allocation:
    /// - Checks environment variables (NO_COLOR, FORCE_COLOR, etc.)
    /// - Detects terminal capabilities
    /// - Respects user preferences
    ///
    /// # Arguments
    /// * `choice` - User's color choice preference
    ///
    /// # Returns
    /// * True if color output should be enabled
    #[inline(always)]
    pub fn should_use_color(choice: &ColorChoice) -> bool {
        match choice {
            ColorChoice::Always => true,
            ColorChoice::Never => false,
            ColorChoice::Auto => detect_color_support(),
            ColorChoice::AlwaysAnsi => true,
        }
    }

    /// Fast color support detection for automatic mode
    ///
    /// # Returns
    /// * True if terminal supports color output
    #[inline(always)]
    pub fn detect_color_support() -> bool {
        // Check NO_COLOR environment variable (universal disable)
        if std::env::var_os("NO_COLOR").is_some() {
            return false;
        }

        // Check FORCE_COLOR environment variable (universal enable)
        if std::env::var_os("FORCE_COLOR").is_some() {
            return true;
        }

        // Check if stdout is a terminal
        #[cfg(unix)]
        {
            // Use environment variable detection instead of isatty
            std::env::var_os("TERM").is_some()
                && std::env::var_os("TERM") != Some("dumb".into())
        }

        #[cfg(windows)]
        {
            // On Windows, assume color support in modern terminals
            true
        }

        #[cfg(not(any(unix, windows)))]
        {
            // Conservative default for unknown platforms
            false
        }
    }

    /// Check if terminal supports true color (24-bit RGB)
    ///
    /// # Returns
    /// * True if terminal supports RGB color output
    #[inline(always)]
    pub fn supports_truecolor() -> bool {
        std::env::var("COLORTERM")
            .map(|v| v == "truecolor" || v == "24bit")
            .unwrap_or(false)
    }

    /// Get the number of colors supported by the terminal
    ///
    /// # Returns
    /// * Number of colors supported (16, 256, or 16777216 for true color)
    #[inline(always)]
    pub fn color_count() -> u32 {
        if supports_truecolor() {
            16_777_216 // 24-bit RGB
        } else if std::env::var("TERM")
            .map(|term| term.contains("256"))
            .unwrap_or(false)
        {
            256
        } else {
            16 // Standard ANSI colors
        }
    }
}

/// Performance optimized color choice utilities
pub mod color_choice_ext {
    use crate::ColorChoice;

    /// Extension methods for ColorChoice with zero-allocation implementations
    pub trait ColorChoiceExt {
        /// Check if this choice should attempt color output
        fn should_attempt_color(&self) -> bool;

        /// Check if this choice should force ANSI output
        fn should_force_ansi(&self) -> bool;
    }

    impl ColorChoiceExt for ColorChoice {
        /// Fast color attempt check with zero allocation
        ///
        /// # Returns
        /// * True if color output should be attempted
        #[inline(always)]
        fn should_attempt_color(&self) -> bool {
            match *self {
                ColorChoice::Always | ColorChoice::AlwaysAnsi => true,
                ColorChoice::Never => false,
                ColorChoice::Auto => {
                    super::color_support::detect_color_support()
                }
            }
        }

        /// Fast ANSI force check with zero allocation
        ///
        /// # Returns
        /// * True if ANSI output should be forced regardless of terminal detection
        #[inline(always)]
        fn should_force_ansi(&self) -> bool {
            matches!(*self, ColorChoice::AlwaysAnsi)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::color_choice_ext::*;
    use super::color_support::*;
    use super::*;
    use crate::ColorChoice;
    use std::io::Cursor;

    #[test]
    fn test_no_color_creation() {
        let writer = Cursor::new(Vec::new());
        let no_color_writer = NoColor::new(writer);

        assert!(!no_color_writer.supports_color());
        assert!(!no_color_writer.supports_hyperlinks());
    }

    #[test]
    fn test_no_color_operations() {
        let mut buffer = Vec::new();
        let mut no_color_writer = NoColor::new(&mut buffer);

        // All color operations should be no-ops
        no_color_writer.set_color(&ColorSpec::new()).unwrap();
        no_color_writer.set_hyperlink(&HyperlinkSpec::close()).unwrap();
        no_color_writer.reset().unwrap();

        // Writing should work normally
        no_color_writer.write_all(b"test").unwrap();
        no_color_writer.flush().unwrap();

        assert_eq!(buffer, b"test");
    }

    #[test]
    fn test_no_color_inner_access() {
        let writer = Cursor::new(Vec::new());
        let mut no_color_writer = NoColor::new(writer);

        // Test mutable access
        no_color_writer.get_mut().write_all(b"direct").unwrap();

        // Test consumption
        let inner = no_color_writer.into_inner();
        assert_eq!(inner.into_inner(), b"direct");
    }

    #[test]
    fn test_color_choice_extensions() {
        assert!(ColorChoice::Always.should_attempt_color());
        assert!(ColorChoice::AlwaysAnsi.should_attempt_color());
        assert!(!ColorChoice::Never.should_attempt_color());

        assert!(ColorChoice::AlwaysAnsi.should_force_ansi());
        assert!(!ColorChoice::Always.should_force_ansi());
        assert!(!ColorChoice::Never.should_force_ansi());
        assert!(!ColorChoice::Auto.should_force_ansi());
    }

    #[test]
    fn test_color_support_detection() {
        // These tests depend on environment, so we just ensure they don't crash
        let _supports_color = should_use_color(&ColorChoice::Auto);
        let _supports_true = supports_truecolor();
        let _color_count = color_count();

        // At minimum, these should return reasonable values
        assert!(color_count() >= 16);
    }

    #[test]
    fn test_color_choice_behavior() {
        assert!(should_use_color(&ColorChoice::Always));
        assert!(!should_use_color(&ColorChoice::Never));
        assert!(should_use_color(&ColorChoice::AlwaysAnsi));

        // Auto depends on environment, just ensure it returns a boolean
        let _auto_result = should_use_color(&ColorChoice::Auto);
    }
}
