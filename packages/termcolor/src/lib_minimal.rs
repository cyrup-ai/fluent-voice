/*!
This crate provides a cross platform abstraction for writing colored text to
a terminal. Colors are written using ANSI escape sequences. Much of this API
was motivated by use inside command line applications, where colors or styles
can be configured by the end user and/or the environment.

This crate also provides platform independent support for writing colored text
to an in memory buffer. While this is easy to do with ANSI escape sequences
(because they are in the buffer themselves), it is trickier on older Windows
systems without virtual terminal support.

In ANSI mode, this crate also provides support for writing hyperlinks.

# Organization

The `WriteColor` trait extends the `io::Write` trait with methods for setting
colors or resetting them.

`StandardStream` and `StandardStreamLock` both satisfy `WriteColor` and are
analogous to `std::io::Stdout` and `std::io::StdoutLock`, or `std::io::Stderr`
and `std::io::StderrLock`.

`Buffer` is an in memory buffer that supports colored text. In a parallel
program, each thread might write to its own buffer. A buffer can be printed to
using a `BufferWriter`. The advantage of this design is that each thread can
work in parallel on a buffer without having to synchronize access to global
resources. Moreover, this design also prevents interleaving of buffer output.

`Ansi` and `NoColor` both satisfy `WriteColor` for arbitrary implementors of
`io::Write`. These types are useful when you know exactly what you need.

# Example: using `StandardStream`

The `StandardStream` type in this crate works similarly to `std::io::Stdout`,
except it is augmented with methods for coloring by the `WriteColor` trait.
For example, to write some green text:

```rust,no_run
# fn test() -> Result<(), Box<::std::error::Error>> {
use std::io::Write;
use termcolor::{Color, ColorChoice, ColorSpec, StandardStream, WriteColor};

let mut stdout = StandardStream::stdout(ColorChoice::Always);
stdout.set_color(ColorSpec::new().set_fg(Some(Color::Green)))?;
writeln!(&mut stdout, "green text!")?;
# Ok(()) }
```

Note that any text written to the terminal now will be colored
green when using ANSI escape sequences, even if it is written via
`stdout.write` and not the methods on `WriteColor`.
*/

pub mod traits;
pub mod types;
pub mod writers;

// Re-export the main types for convenience
pub use traits::WriteColor;
pub use types::{Color, ColorChoice, ColorSpec, HyperlinkSpec};
pub use writers::{NoColor, StringWriter};

// For now, we'll keep a minimal set of functionality
// The full implementation would include StandardStream, Buffer, etc.