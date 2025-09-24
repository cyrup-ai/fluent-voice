use crate::{Color, ColorChoice, ColorSpec, HyperlinkSpec, WriteColor};
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(windows)]
use winapi_util::console as wincon;

/// `std::io` implements `Stdout` and `Stderr` (and their `Lock` variants) as
/// separate types, which makes it difficult to abstract over them. We use
/// some simple internal enum types to work around this.

enum StandardStreamType {
    Stdout,
    Stderr,
    StdoutBuffered,
    StderrBuffered}

#[derive(Debug)]
enum IoStandardStream {
    Stdout(io::Stdout),
    Stderr(io::Stderr),
    StdoutBuffered(io::BufWriter<io::Stdout>),
    StderrBuffered(io::BufWriter<io::Stderr>)}

impl IoStandardStream {
    fn new(sty: StandardStreamType) -> IoStandardStream {
        match sty {
            StandardStreamType::Stdout => {
                IoStandardStream::Stdout(io::stdout())
            }
            StandardStreamType::Stderr => {
                IoStandardStream::Stderr(io::stderr())
            }
            StandardStreamType::StdoutBuffered => {
                let wtr = io::BufWriter::new(io::stdout());
                IoStandardStream::StdoutBuffered(wtr)
            }
            StandardStreamType::StderrBuffered => {
                let wtr = io::BufWriter::new(io::stderr());
                IoStandardStream::StderrBuffered(wtr)
            }
        }
    }

    fn lock(&self) -> IoStandardStreamLock<'_> {
        match *self {
            IoStandardStream::Stdout(ref s) => {
                IoStandardStreamLock::StdoutLock(s.lock())
            }
            IoStandardStream::Stderr(ref s) => {
                IoStandardStreamLock::StderrLock(s.lock())
            }
            IoStandardStream::StdoutBuffered(_)
            | IoStandardStream::StderrBuffered(_) => {
                // We don't permit this case to ever occur in the public API,
                // so it's OK to panic.
                panic!("cannot lock a buffered standard stream")
            }
        }
    }
}

impl io::Write for IoStandardStream {
    #[inline(always)]
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        match *self {
            IoStandardStream::Stdout(ref mut s) => s.write(b),
            IoStandardStream::Stderr(ref mut s) => s.write(b),
            IoStandardStream::StdoutBuffered(ref mut s) => s.write(b),
            IoStandardStream::StderrBuffered(ref mut s) => s.write(b)}
    }

    #[inline(always)]
    fn flush(&mut self) -> io::Result<()> {
        match *self {
            IoStandardStream::Stdout(ref mut s) => s.flush(),
            IoStandardStream::Stderr(ref mut s) => s.flush(),
            IoStandardStream::StdoutBuffered(ref mut s) => s.flush(),
            IoStandardStream::StderrBuffered(ref mut s) => s.flush()}
    }
}

// Same rigmarole for the locked variants of the standard streams.

#[derive(Debug)]
enum IoStandardStreamLock<'a> {
    StdoutLock(io::StdoutLock<'a>),
    StderrLock(io::StderrLock<'a>)}

impl<'a> io::Write for IoStandardStreamLock<'a> {
    #[inline(always)]
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        match *self {
            IoStandardStreamLock::StdoutLock(ref mut s) => s.write(b),
            IoStandardStreamLock::StderrLock(ref mut s) => s.write(b)}
    }

    #[inline(always)]
    fn flush(&mut self) -> io::Result<()> {
        match *self {
            IoStandardStreamLock::StdoutLock(ref mut s) => s.flush(),
            IoStandardStreamLock::StderrLock(ref mut s) => s.flush()}
    }
}

/// A standard stream for writing to stdout or stderr.
///
/// This satisfies both `io::Write` and `WriteColor`, and buffers writes
/// until either `flush` is called or the buffer is full.
#[derive(Debug)]
pub struct StandardStream {
    wtr: LossyStandardStream<WriterInner<IoStandardStream>>}

/// `StandardStreamLock` is a locked reference to a `StandardStream`.
///
/// This implements the `io::Write` and `WriteColor` traits, and is constructed
/// via the `Write::lock` method.
///
/// The lifetime `'a` refers to the lifetime of the corresponding
/// `StandardStream`.
#[derive(Debug)]
pub struct StandardStreamLock<'a> {
    wtr: LossyStandardStream<WriterInnerLock<IoStandardStreamLock<'a>>>}

/// Like `StandardStream`, but does buffered writing.
#[derive(Debug)]
pub struct BufferedStandardStream {
    wtr: LossyStandardStream<WriterInner<IoStandardStream>>}

/// WriterInner is a (limited) generic representation of a writer.
#[derive(Debug)]
enum WriterInner<W> {
    NoColor(NoColor<W>),
    Ansi(Ansi<W>)}

/// WriterInnerLock is a (limited) generic representation of a writer.
#[derive(Debug)]
enum WriterInnerLock<W> {
    NoColor(NoColor<W>),
    Ansi(Ansi<W>)}

impl StandardStream {
    /// Create a new `StandardStream` with the given color preferences that
    /// writes to standard output.
    ///
    /// If coloring is desired, ANSI escape sequences are used.
    pub fn stdout(choice: ColorChoice) -> StandardStream {
        let wtr = WriterInner::create(StandardStreamType::Stdout, choice);
        StandardStream { wtr: LossyStandardStream::new(wtr) }
    }

    /// Create a new `StandardStream` with the given color preferences that
    /// writes to standard error.
    ///
    /// If coloring is desired, ANSI escape sequences are used.
    pub fn stderr(choice: ColorChoice) -> StandardStream {
        let wtr = WriterInner::create(StandardStreamType::Stderr, choice);
        StandardStream { wtr: LossyStandardStream::new(wtr) }
    }

    /// Lock the underlying writer.
    ///
    /// The lock guard returned also satisfies `io::Write` and
    /// `WriteColor`.
    ///
    /// This method is **not reentrant**. It may panic if `lock` is called
    /// while a `StandardStreamLock` is still alive.
    pub fn lock(&self) -> StandardStreamLock<'_> {
        StandardStreamLock::from_stream(self)
    }
}

impl<'a> StandardStreamLock<'a> {
    fn from_stream(stream: &StandardStream) -> StandardStreamLock<'_> {
        let locked = match *stream.wtr.get_ref() {
            WriterInner::NoColor(ref w) => {
                WriterInnerLock::NoColor(NoColor(w.0.lock()))
            }
            WriterInner::Ansi(ref w) => {
                WriterInnerLock::Ansi(Ansi(w.0.lock()))
            }
        };
        StandardStreamLock { wtr: stream.wtr.wrap(locked) }
    }
}

impl BufferedStandardStream {
    /// Create a new `BufferedStandardStream` with the given color preferences
    /// that writes to standard output via a buffered writer.
    ///
    /// If coloring is desired, ANSI escape sequences are used.
    pub fn stdout(choice: ColorChoice) -> BufferedStandardStream {
        let wtr =
            WriterInner::create(StandardStreamType::StdoutBuffered, choice);
        BufferedStandardStream { wtr: LossyStandardStream::new(wtr) }
    }

    /// Create a new `BufferedStandardStream` with the given color preferences
    /// that writes to standard error via a buffered writer.
    ///
    /// If coloring is desired, ANSI escape sequences are used.
    pub fn stderr(choice: ColorChoice) -> BufferedStandardStream {
        let wtr =
            WriterInner::create(StandardStreamType::StderrBuffered, choice);
        BufferedStandardStream { wtr: LossyStandardStream::new(wtr) }
    }
}

impl WriterInner<IoStandardStream> {
    /// Create a new inner writer for a standard stream with the given color
    /// preferences.
    #[cfg(not(windows))]
    fn create(
        sty: StandardStreamType,
        choice: ColorChoice,
    ) -> WriterInner<IoStandardStream> {
        if choice.should_attempt_color() {
            WriterInner::Ansi(Ansi(IoStandardStream::new(sty)))
        } else {
            WriterInner::NoColor(NoColor(IoStandardStream::new(sty)))
        }
    }

    #[cfg(windows)]
    fn create(
        sty: StandardStreamType,
        choice: ColorChoice,
    ) -> WriterInner<IoStandardStream> {
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
        if choice.should_attempt_color()
            && (enabled_virtual || choice.should_force_ansi())
        {
            WriterInner::Ansi(Ansi(IoStandardStream::new(sty)))
        } else {
            WriterInner::NoColor(NoColor(IoStandardStream::new(sty)))
        }
    }
}

impl io::Write for StandardStream {
    #[inline]
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        self.wtr.write(b)
    }

    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        self.wtr.flush()
    }
}

impl WriteColor for StandardStream {
    #[inline]
    fn supports_color(&self) -> bool {
        self.wtr.supports_color()
    }

    #[inline]
    fn supports_hyperlinks(&self) -> bool {
        self.wtr.supports_hyperlinks()
    }

    #[inline]
    fn set_color(&mut self, spec: &ColorSpec) -> io::Result<()> {
        self.wtr.set_color(spec)
    }

    #[inline]
    fn set_hyperlink(&mut self, link: &HyperlinkSpec) -> io::Result<()> {
        self.wtr.set_hyperlink(link)
    }

    #[inline]
    fn reset(&mut self) -> io::Result<()> {
        self.wtr.reset()
    }
}

impl<'a> io::Write for StandardStreamLock<'a> {
    #[inline]
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        self.wtr.write(b)
    }

    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        self.wtr.flush()
    }
}

impl<'a> WriteColor for StandardStreamLock<'a> {
    #[inline]
    fn supports_color(&self) -> bool {
        self.wtr.supports_color()
    }

    #[inline]
    fn supports_hyperlinks(&self) -> bool {
        self.wtr.supports_hyperlinks()
    }

    #[inline]
    fn set_color(&mut self, spec: &ColorSpec) -> io::Result<()> {
        self.wtr.set_color(spec)
    }

    #[inline]
    fn set_hyperlink(&mut self, link: &HyperlinkSpec) -> io::Result<()> {
        self.wtr.set_hyperlink(link)
    }

    #[inline]
    fn reset(&mut self) -> io::Result<()> {
        self.wtr.reset()
    }
}

impl io::Write for BufferedStandardStream {
    #[inline]
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        self.wtr.write(b)
    }

    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        self.wtr.flush()
    }
}

impl WriteColor for BufferedStandardStream {
    #[inline]
    fn supports_color(&self) -> bool {
        self.wtr.supports_color()
    }

    #[inline]
    fn supports_hyperlinks(&self) -> bool {
        self.wtr.supports_hyperlinks()
    }

    #[inline]
    fn set_color(&mut self, spec: &ColorSpec) -> io::Result<()> {
        self.wtr.set_color(spec)
    }

    #[inline]
    fn set_hyperlink(&mut self, link: &HyperlinkSpec) -> io::Result<()> {
        self.wtr.set_hyperlink(link)
    }

    #[inline]
    fn reset(&mut self) -> io::Result<()> {
        self.wtr.reset()
    }
}

impl<W: io::Write> io::Write for WriterInner<W> {
    #[inline(always)]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match *self {
            WriterInner::NoColor(ref mut wtr) => wtr.write(buf),
            WriterInner::Ansi(ref mut wtr) => wtr.write(buf)}
    }

    #[inline(always)]
    fn flush(&mut self) -> io::Result<()> {
        match *self {
            WriterInner::NoColor(ref mut wtr) => wtr.flush(),
            WriterInner::Ansi(ref mut wtr) => wtr.flush()}
    }
}

impl<W: io::Write> WriteColor for WriterInner<W> {
    fn supports_color(&self) -> bool {
        match *self {
            WriterInner::NoColor(_) => false,
            WriterInner::Ansi(_) => true}
    }

    fn supports_hyperlinks(&self) -> bool {
        match *self {
            WriterInner::NoColor(_) => false,
            WriterInner::Ansi(_) => true}
    }

    fn set_color(&mut self, spec: &ColorSpec) -> io::Result<()> {
        match *self {
            WriterInner::NoColor(ref mut wtr) => wtr.set_color(spec),
            WriterInner::Ansi(ref mut wtr) => wtr.set_color(spec)}
    }

    fn set_hyperlink(&mut self, link: &HyperlinkSpec) -> io::Result<()> {
        match *self {
            WriterInner::NoColor(ref mut wtr) => wtr.set_hyperlink(link),
            WriterInner::Ansi(ref mut wtr) => wtr.set_hyperlink(link)}
    }

    fn reset(&mut self) -> io::Result<()> {
        match *self {
            WriterInner::NoColor(ref mut wtr) => wtr.reset(),
            WriterInner::Ansi(ref mut wtr) => wtr.reset()}
    }
}

impl<W: io::Write> io::Write for WriterInnerLock<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match *self {
            WriterInnerLock::NoColor(ref mut wtr) => wtr.write(buf),
            WriterInnerLock::Ansi(ref mut wtr) => wtr.write(buf)}
    }

    fn flush(&mut self) -> io::Result<()> {
        match *self {
            WriterInnerLock::NoColor(ref mut wtr) => wtr.flush(),
            WriterInnerLock::Ansi(ref mut wtr) => wtr.flush()}
    }
}

impl<W: io::Write> WriteColor for WriterInnerLock<W> {
    fn supports_color(&self) -> bool {
        match *self {
            WriterInnerLock::NoColor(_) => false,
            WriterInnerLock::Ansi(_) => true}
    }

    fn supports_hyperlinks(&self) -> bool {
        match *self {
            WriterInnerLock::NoColor(_) => false,
            WriterInnerLock::Ansi(_) => true}
    }

    fn set_color(&mut self, spec: &ColorSpec) -> io::Result<()> {
        match *self {
            WriterInnerLock::NoColor(ref mut wtr) => wtr.set_color(spec),
            WriterInnerLock::Ansi(ref mut wtr) => wtr.set_color(spec)}
    }

    fn set_hyperlink(&mut self, link: &HyperlinkSpec) -> io::Result<()> {
        match *self {
            WriterInnerLock::NoColor(ref mut wtr) => wtr.set_hyperlink(link),
            WriterInnerLock::Ansi(ref mut wtr) => wtr.set_hyperlink(link)}
    }

    fn reset(&mut self) -> io::Result<()> {
        match *self {
            WriterInnerLock::NoColor(ref mut wtr) => wtr.reset(),
            WriterInnerLock::Ansi(ref mut wtr) => wtr.reset()}
    }
}

/// Writes colored buffers to stdout or stderr.
///
/// Writable buffers can be obtained by calling `buffer` on a `BufferWriter`.
///
/// This writer works with terminals that support ANSI escape sequences.
///
/// It is intended for a `BufferWriter` to be used from multiple threads
/// simultaneously, but note that buffer printing is serialized.
#[derive(Debug)]
pub struct BufferWriter {
    stream: LossyStandardStream<IoStandardStream>,
    printed: AtomicBool,
    separator: Option<Vec<u8>>,
    use_color: bool}

impl BufferWriter {
    /// Create a new `BufferWriter` that writes to a standard stream with the
    /// given color preferences.
    #[cfg(not(windows))]
    fn create(sty: StandardStreamType, choice: ColorChoice) -> BufferWriter {
        let use_color = choice.should_attempt_color();
        BufferWriter {
            stream: LossyStandardStream::new(IoStandardStream::new(sty)),
            printed: AtomicBool::new(false),
            separator: None,
            use_color}
    }

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
            use_color}
    }

    /// Create a new `BufferWriter` that writes to stdout with the given
    /// color preferences.
    pub fn stdout(choice: ColorChoice) -> BufferWriter {
        BufferWriter::create(StandardStreamType::Stdout, choice)
    }

    /// Create a new `BufferWriter` that writes to stderr with the given
    /// color preferences.
    pub fn stderr(choice: ColorChoice) -> BufferWriter {
        BufferWriter::create(StandardStreamType::Stderr, choice)
    }

    /// If set, the separator given is printed between buffers. By default, no
    /// separator is printed.
    ///
    /// The default value is `None`.
    pub fn separator(&mut self, sep: Option<Vec<u8>>) {
        self.separator = sep;
    }

    /// Creates a new `Buffer` with the current color preferences.
    ///
    /// A `Buffer` satisfies both `io::Write` and `WriteColor`. A `Buffer` can
    /// be printed using the `print` method.
    pub fn buffer(&self) -> Buffer {
        if self.use_color { Buffer::ansi() } else { Buffer::no_color() }
    }

    /// Prints the contents of the given buffer.
    ///
    /// It is safe to call this from multiple threads simultaneously. In
    /// particular, all buffers are written atomically. No interleaving will
    /// occur.
    pub fn print(&self, buf: &Buffer) -> io::Result<()> {
        if buf.is_empty() {
            return Ok(());
        }
        let mut stream = self.stream.wrap(self.stream.get_ref().lock());
        if let Some(ref sep) = self.separator {
            if self.printed.load(Ordering::Relaxed) {
                stream.write_all(sep)?;
                stream.write_all(b"\n")?;
            }
        }
        match buf.0 {
            BufferInner::NoColor(ref b) => stream.write_all(&b.0)?,
            BufferInner::Ansi(ref b) => stream.write_all(&b.0)?}
        self.printed.store(true, Ordering::Relaxed);
        Ok(())
    }
}

/// Write colored text to memory.
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
#[derive(Clone, Debug)]
pub struct Buffer(BufferInner);

/// BufferInner is an enumeration of different buffer types.
#[derive(Clone, Debug)]
enum BufferInner {
    /// No coloring information should be applied. This ignores all coloring
    /// directives.
    NoColor(NoColor<Vec<u8>>),
    /// Apply coloring using ANSI escape sequences embedded into the buffer.
    Ansi(Ansi<Vec<u8>>)}

impl Buffer {
    /// Create a new buffer with the given color settings.
    #[cfg(not(windows))]
    #[allow(dead_code)]
    fn new(choice: ColorChoice) -> Buffer {
        if choice.should_attempt_color() {
            Buffer::ansi()
        } else {
            Buffer::no_color()
        }
    }

    /// Create a new buffer with the given color settings.
    #[cfg(windows)]
    #[allow(dead_code)]
    fn new(choice: ColorChoice) -> Buffer {
        if choice.should_attempt_color() && choice.should_force_ansi() {
            Buffer::ansi()
        } else {
            Buffer::no_color()
        }
    }

    /// Create a buffer that drops all color information.
    pub fn no_color() -> Buffer {
        Buffer(BufferInner::NoColor(NoColor(vec![])))
    }

    /// Create a buffer that uses ANSI escape sequences.
    pub fn ansi() -> Buffer {
        Buffer(BufferInner::Ansi(Ansi(vec![])))
    }

    /// Returns true if and only if this buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the length of this buffer in bytes.
    pub fn len(&self) -> usize {
        match self.0 {
            BufferInner::NoColor(ref b) => b.0.len(),
            BufferInner::Ansi(ref b) => b.0.len()}
    }

    /// Clears this buffer.
    pub fn clear(&mut self) {
        match self.0 {
            BufferInner::NoColor(ref mut b) => b.0.clear(),
            BufferInner::Ansi(ref mut b) => b.0.clear()}
    }

    /// Consume this buffer and return the underlying raw data.
    pub fn into_inner(self) -> Vec<u8> {
        match self.0 {
            BufferInner::NoColor(b) => b.0,
            BufferInner::Ansi(b) => b.0}
    }

    /// Return the underlying data of the buffer.
    pub fn as_slice(&self) -> &[u8] {
        match self.0 {
            BufferInner::NoColor(ref b) => &b.0,
            BufferInner::Ansi(ref b) => &b.0}
    }

    /// Return the underlying data of the buffer as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        match self.0 {
            BufferInner::NoColor(ref mut b) => &mut b.0,
            BufferInner::Ansi(ref mut b) => &mut b.0}
    }
}

impl io::Write for Buffer {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self.0 {
            BufferInner::NoColor(ref mut w) => w.write(buf),
            BufferInner::Ansi(ref mut w) => w.write(buf)}
    }

    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        match self.0 {
            BufferInner::NoColor(ref mut w) => w.flush(),
            BufferInner::Ansi(ref mut w) => w.flush()}
    }
}

impl WriteColor for Buffer {
    #[inline]
    fn supports_color(&self) -> bool {
        match self.0 {
            BufferInner::NoColor(_) => false,
            BufferInner::Ansi(_) => true}
    }

    #[inline]
    fn supports_hyperlinks(&self) -> bool {
        match self.0 {
            BufferInner::NoColor(_) => false,
            BufferInner::Ansi(_) => true}
    }

    #[inline]
    fn set_color(&mut self, spec: &ColorSpec) -> io::Result<()> {
        match self.0 {
            BufferInner::NoColor(ref mut w) => w.set_color(spec),
            BufferInner::Ansi(ref mut w) => w.set_color(spec)}
    }

    #[inline]
    fn set_hyperlink(&mut self, link: &HyperlinkSpec) -> io::Result<()> {
        match self.0 {
            BufferInner::NoColor(ref mut w) => w.set_hyperlink(link),
            BufferInner::Ansi(ref mut w) => w.set_hyperlink(link)}
    }

    #[inline]
    fn reset(&mut self) -> io::Result<()> {
        match self.0 {
            BufferInner::NoColor(ref mut w) => w.reset(),
            BufferInner::Ansi(ref mut w) => w.reset()}
    }
}

/// Satisfies `WriteColor` but ignores all color options.
#[derive(Clone, Debug)]
pub struct NoColor<W>(pub W);

impl<W: Write> NoColor<W> {
    /// Create a new writer that satisfies `WriteColor` but drops all color
    /// information.
    pub fn new(wtr: W) -> NoColor<W> {
        NoColor(wtr)
    }

    /// Consume this `NoColor` value and return the inner writer.
    pub fn into_inner(self) -> W {
        self.0
    }

    /// Return a reference to the inner writer.
    pub fn get_ref(&self) -> &W {
        &self.0
    }

    /// Return a mutable reference to the inner writer.
    pub fn get_mut(&mut self) -> &mut W {
        &mut self.0
    }
}

impl<W: io::Write> io::Write for NoColor<W> {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.write(buf)
    }

    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        self.0.flush()
    }
}

impl<W: io::Write> WriteColor for NoColor<W> {
    #[inline]
    fn supports_color(&self) -> bool {
        false
    }

    #[inline]
    fn supports_hyperlinks(&self) -> bool {
        false
    }

    #[inline]
    fn set_color(&mut self, _: &ColorSpec) -> io::Result<()> {
        Ok(())
    }

    #[inline]
    fn set_hyperlink(&mut self, _: &HyperlinkSpec) -> io::Result<()> {
        Ok(())
    }

    #[inline]
    fn reset(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Satisfies `WriteColor` using standard ANSI escape sequences.
#[derive(Clone, Debug)]
pub struct Ansi<W>(pub W);

impl<W: Write> Ansi<W> {
    /// Create a new writer that satisfies `WriteColor` using standard ANSI
    /// escape sequences.
    pub fn new(wtr: W) -> Ansi<W> {
        Ansi(wtr)
    }

    /// Consume this `Ansi` value and return the inner writer.
    pub fn into_inner(self) -> W {
        self.0
    }

    /// Return a reference to the inner writer.
    pub fn get_ref(&self) -> &W {
        &self.0
    }

    /// Return a mutable reference to the inner writer.
    pub fn get_mut(&mut self) -> &mut W {
        &mut self.0
    }
}

impl<W: io::Write> io::Write for Ansi<W> {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.write(buf)
    }

    // Adding this method here is not required because it has a default impl,
    // but it seems to provide a perf improvement in some cases when using
    // a `BufWriter` with lots of writes.
    //
    // See https://github.com/BurntSushi/termcolor/pull/56 for more details
    // and a minimized example.
    #[inline]
    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        self.0.write_all(buf)
    }

    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        self.0.flush()
    }
}

impl<W: io::Write> WriteColor for Ansi<W> {
    #[inline]
    fn supports_color(&self) -> bool {
        true
    }

    #[inline]
    fn supports_hyperlinks(&self) -> bool {
        true
    }

    #[inline]
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
        if let Some(ref c) = spec.fg() {
            self.write_color(true, c, spec.intense())?;
        }
        if let Some(ref c) = spec.bg() {
            self.write_color(false, c, spec.intense())?;
        }
        Ok(())
    }

    #[inline]
    fn set_hyperlink(&mut self, link: &HyperlinkSpec) -> io::Result<()> {
        self.write_str("\x1B]8;;")?;
        if let Some(uri) = link.uri() {
            self.write_all(uri)?;
        }
        self.write_str("\x1B\\")
    }

    #[inline]
    fn reset(&mut self) -> io::Result<()> {
        self.write_str("\x1B[0m")
    }
}

impl<W: io::Write> Ansi<W> {
    fn write_str(&mut self, s: &str) -> io::Result<()> {
        self.write_all(s.as_bytes())
    }

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
                assert!(pre_len <= 7);
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
                Color::__Nonexhaustive => unreachable!()}
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
                Color::__Nonexhaustive => unreachable!()}
        }
    }
}

impl WriteColor for io::Sink {
    fn supports_color(&self) -> bool {
        false
    }

    fn supports_hyperlinks(&self) -> bool {
        false
    }

    fn set_color(&mut self, _: &ColorSpec) -> io::Result<()> {
        Ok(())
    }

    fn set_hyperlink(&mut self, _: &HyperlinkSpec) -> io::Result<()> {
        Ok(())
    }

    fn reset(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[derive(Debug)]
struct LossyStandardStream<W> {
    wtr: W,
    #[cfg(windows)]
    is_console: bool}

impl<W: io::Write> LossyStandardStream<W> {
    #[cfg(not(windows))]
    fn new(wtr: W) -> LossyStandardStream<W> {
        LossyStandardStream { wtr }
    }

    #[cfg(windows)]
    fn new(wtr: W) -> LossyStandardStream<W> {
        LossyStandardStream { wtr, is_console: false }
    }

    #[cfg(not(windows))]
    fn wrap<Q: io::Write>(&self, wtr: Q) -> LossyStandardStream<Q> {
        LossyStandardStream::new(wtr)
    }

    #[cfg(windows)]
    fn wrap<Q: io::Write>(&self, wtr: Q) -> LossyStandardStream<Q> {
        LossyStandardStream { wtr, is_console: self.is_console }
    }

    fn get_ref(&self) -> &W {
        &self.wtr
    }
}

impl<W: WriteColor> WriteColor for LossyStandardStream<W> {
    fn supports_color(&self) -> bool {
        self.wtr.supports_color()
    }
    fn supports_hyperlinks(&self) -> bool {
        self.wtr.supports_hyperlinks()
    }
    fn set_color(&mut self, spec: &ColorSpec) -> io::Result<()> {
        self.wtr.set_color(spec)
    }
    fn set_hyperlink(&mut self, link: &HyperlinkSpec) -> io::Result<()> {
        self.wtr.set_hyperlink(link)
    }
    fn reset(&mut self) -> io::Result<()> {
        self.wtr.reset()
    }
}

impl<W: io::Write> io::Write for LossyStandardStream<W> {
    #[cfg(not(windows))]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.wtr.write(buf)
    }

    #[cfg(windows)]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.is_console {
            write_lossy_utf8(&mut self.wtr, buf)
        } else {
            self.wtr.write(buf)
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        self.wtr.flush()
    }
}

#[cfg(windows)]
fn write_lossy_utf8<W: io::Write>(mut w: W, buf: &[u8]) -> io::Result<usize> {
    match ::std::str::from_utf8(buf) {
        Ok(s) => w.write(s.as_bytes()),
        Err(ref e) if e.valid_up_to() == 0 => {
            w.write(b"\xEF\xBF\xBD")?;
            Ok(1)
        }
        Err(e) => w.write(&buf[..e.valid_up_to()])}
}

impl WriteColor for Vec<u8> {
    fn supports_color(&self) -> bool {
        false // Vec<u8> doesn't support color output
    }

    fn supports_hyperlinks(&self) -> bool {
        false // Vec<u8> doesn't support hyperlinks
    }

    fn set_color(&mut self, _spec: &ColorSpec) -> io::Result<()> {
        Ok(()) // No-op for Vec<u8>
    }

    fn set_hyperlink(&mut self, _link: &HyperlinkSpec) -> io::Result<()> {
        Ok(()) // No-op for Vec<u8>
    }

    fn reset(&mut self) -> io::Result<()> {
        Ok(()) // No-op for Vec<u8>
    }

    fn is_synchronous(&self) -> bool {
        false // Vec<u8> is not synchronous
    }
}

/// A wrapper for String that implements both Write and WriteColor.
///
/// This fixes naga 26.0.0 compatibility where it expects WriteColor on string-like writers.
/// The StringWriter collects written data into an internal String buffer without any
/// color formatting (colors are ignored).
#[derive(Debug, Default)]
pub struct StringWriter {
    /// The internal string buffer that collects written data.
    pub inner: String}

impl StringWriter {
    /// Creates a new empty StringWriter.
    pub fn new() -> Self {
        Self { inner: String::new() }
    }

    /// Creates a new StringWriter with the specified capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self { inner: String::with_capacity(capacity) }
    }

    /// Consumes the StringWriter and returns the internal String.
    pub fn into_string(self) -> String {
        self.inner
    }

    /// Returns a string slice of the internal buffer.
    pub fn as_str(&self) -> &str {
        &self.inner
    }
}

impl io::Write for StringWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match std::str::from_utf8(buf) {
            Ok(s) => {
                self.inner.push_str(s);
                Ok(buf.len())
            }
            Err(_) => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid UTF-8",
            ))}
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl WriteColor for StringWriter {
    fn supports_color(&self) -> bool {
        false // StringWriter doesn't support color output
    }

    fn supports_hyperlinks(&self) -> bool {
        false // StringWriter doesn't support hyperlinks  
    }

    fn set_color(&mut self, _spec: &ColorSpec) -> io::Result<()> {
        Ok(()) // No-op for StringWriter
    }

    fn set_hyperlink(&mut self, _link: &HyperlinkSpec) -> io::Result<()> {
        Ok(()) // No-op for StringWriter
    }

    fn reset(&mut self) -> io::Result<()> {
        Ok(()) // No-op for StringWriter
    }

    fn is_synchronous(&self) -> bool {
        false // StringWriter is not synchronous
    }
}

/// A String wrapper that implements both `io::Write` and `WriteColor`.
///
/// This type provides backward compatibility with libraries like naga that expect
/// string-like writers to implement `WriteColor`. It's a zero-cost wrapper around
/// `String` that adds the necessary trait implementations.
#[derive(Debug, Default, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct TermString(pub String);

impl TermString {
    /// Creates a new empty `TermString`.
    pub fn new() -> Self {
        Self(String::new())
    }

    /// Creates a new `TermString` with the specified capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self(String::with_capacity(capacity))
    }

    /// Consumes the `TermString` and returns the inner `String`.
    pub fn into_inner(self) -> String {
        self.0
    }

    /// Returns a string slice of the `TermString` contents.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Appends a string slice to the end of this `TermString`.
    pub fn push_str(&mut self, s: &str) {
        self.0.push_str(s)
    }
}

impl From<String> for TermString {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<TermString> for String {
    fn from(ts: TermString) -> Self {
        ts.0
    }
}

impl AsRef<str> for TermString {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for TermString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

// Now we can implement io::Write for our TermString!
impl io::Write for TermString {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match std::str::from_utf8(buf) {
            Ok(s) => {
                self.0.push_str(s);
                Ok(buf.len())
            }
            Err(_) => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid UTF-8",
            ))}
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(()) // String doesn't need flushing
    }
}

// And WriteColor for TermString
impl WriteColor for TermString {
    fn supports_color(&self) -> bool {
        false // TermString doesn't support color output
    }

    fn supports_hyperlinks(&self) -> bool {
        false // TermString doesn't support hyperlinks  
    }

    fn set_color(&mut self, _spec: &ColorSpec) -> io::Result<()> {
        Ok(()) // No-op for TermString
    }

    fn set_hyperlink(&mut self, _link: &HyperlinkSpec) -> io::Result<()> {
        Ok(()) // No-op for TermString
    }

    fn reset(&mut self) -> io::Result<()> {
        Ok(()) // No-op for TermString
    }

    fn is_synchronous(&self) -> bool {
        false // TermString is not synchronous
    }
}
