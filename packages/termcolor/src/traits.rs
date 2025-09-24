use crate::{ColorSpec, HyperlinkSpec};
use std::io;

/// This trait describes the behavior of writers that support colored output.
pub trait WriteColor: io::Write {
    /// Returns true if and only if the underlying writer supports colors.
    fn supports_color(&self) -> bool;

    /// Set the color settings of the writer.
    ///
    /// Subsequent writes to this writer will use these settings until either
    /// `reset` is called or new color settings are set.
    ///
    /// If there was a problem setting the color settings, then an error is
    /// returned.
    fn set_color(&mut self, spec: &ColorSpec) -> io::Result<()>;

    /// Reset the current color settings to their original settings.
    ///
    /// If there was a problem resetting the color settings, then an error is
    /// returned.
    ///
    /// Note that this does not reset hyperlinks. Those need to be
    /// reset on their own, e.g., by calling `set_hyperlink` with
    /// [`HyperlinkSpec::none`].
    fn reset(&mut self) -> io::Result<()>;

    /// Set the current hyperlink of the writer.
    ///
    /// The typical way to use this is to first call it with a
    /// [`HyperlinkSpec::open`] to write the actual URI to a tty that supports
    /// [OSC-8]. At this point, the caller can now write the label for the
    /// hyperlink. This may include coloring or other styles. Once the caller
    /// has finished writing the label, one should call this method again with
    /// [`HyperlinkSpec::close`].
    ///
    /// If there was a problem setting the hyperlink, then an error is
    /// returned.
    ///
    /// This defaults to doing nothing.
    ///
    /// [OSC8]: https://github.com/Alhadis/OSC8-Adoption/
    fn set_hyperlink(&mut self, _link: &HyperlinkSpec) -> io::Result<()> {
        Ok(())
    }

    /// Returns true if and only if the underlying writer supports hyperlinks.
    ///
    /// This can be used to avoid generating hyperlink URIs unnecessarily.
    ///
    /// This defaults to `false`.
    fn supports_hyperlinks(&self) -> bool {
        false
    }

    /// Returns true if and only if the underlying writer must synchronously
    /// interact with an end user's device in order to control colors. By
    /// default, this always returns `false`.
    ///
    /// In particular, this returns true when the underlying writer is a
    /// Windows console that was created before Windows 10 build 14931 (2016).
    fn is_synchronous(&self) -> bool {
        false
    }
}

impl<T: ?Sized + WriteColor> WriteColor for &mut T {
    fn supports_color(&self) -> bool {
        (**self).supports_color()
    }
    fn supports_hyperlinks(&self) -> bool {
        (**self).supports_hyperlinks()
    }
    fn set_color(&mut self, spec: &ColorSpec) -> io::Result<()> {
        (**self).set_color(spec)
    }
    fn set_hyperlink(&mut self, link: &HyperlinkSpec) -> io::Result<()> {
        (**self).set_hyperlink(link)
    }
    fn reset(&mut self) -> io::Result<()> {
        (**self).reset()
    }
}

impl<T: ?Sized + WriteColor> WriteColor for Box<T> {
    fn supports_color(&self) -> bool {
        (**self).supports_color()
    }
    fn supports_hyperlinks(&self) -> bool {
        (**self).supports_hyperlinks()
    }
    fn set_color(&mut self, spec: &ColorSpec) -> io::Result<()> {
        (**self).set_color(spec)
    }
    fn set_hyperlink(&mut self, link: &HyperlinkSpec) -> io::Result<()> {
        (**self).set_hyperlink(link)
    }
    fn reset(&mut self) -> io::Result<()> {
        (**self).reset()
    }
}
