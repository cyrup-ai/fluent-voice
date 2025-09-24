use std::env;
use std::fmt;
use std::str::FromStr;

/// ColorChoice represents the color preferences of an end user.
///
/// The `Default` implementation for this type will select `Auto`, which tries
/// to do the right thing based on the current environment.
///
/// The `FromStr` implementation for this type converts a lowercase kebab-case
/// string of the variant name to the corresponding variant. Any other string
/// results in an error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ColorChoice {
    /// Try very hard to emit colors. This includes emitting ANSI colors
    /// on Windows if virtual terminal processing can be enabled or if forced.
    Always,
    /// AlwaysAnsi is like Always, except it never tries to use anything other
    /// than emitting ANSI color codes.
    AlwaysAnsi,
    /// Try to use colors, but don't force the issue. If the console isn't
    /// available on Windows, or if TERM=dumb, or if `NO_COLOR` is defined, for
    /// example, then don't use colors.
    Auto,
    /// Never emit colors.
    Never,
}

/// The default is `Auto`.
impl Default for ColorChoice {
    fn default() -> ColorChoice {
        ColorChoice::Auto
    }
}

impl FromStr for ColorChoice {
    type Err = ColorChoiceParseError;

    fn from_str(s: &str) -> Result<ColorChoice, ColorChoiceParseError> {
        match s.to_lowercase().as_str() {
            "always" => Ok(ColorChoice::Always),
            "always-ansi" => Ok(ColorChoice::AlwaysAnsi),
            "never" => Ok(ColorChoice::Never),
            "auto" => Ok(ColorChoice::Auto),
            unknown => Err(ColorChoiceParseError {
                unknown_choice: unknown.to_string(),
            }),
        }
    }
}

impl ColorChoice {
    /// Returns true if we should attempt to write colored output.
    pub(crate) fn should_attempt_color(&self) -> bool {
        match *self {
            ColorChoice::Always => true,
            ColorChoice::AlwaysAnsi => true,
            ColorChoice::Never => false,
            ColorChoice::Auto => self.env_allows_color(),
        }
    }

    #[cfg(not(windows))]
    fn env_allows_color(&self) -> bool {
        match env::var_os("TERM") {
            // If TERM isn't set, then we are in a weird environment that
            // probably doesn't support colors.
            None => return false,
            Some(k) => {
                if k == "dumb" {
                    return false;
                }
            }
        }
        // If TERM != dumb, then the only way we don't allow colors at this
        // point is if NO_COLOR is set.
        if env::var_os("NO_COLOR").is_some() {
            return false;
        }
        true
    }

    #[cfg(windows)]
    fn env_allows_color(&self) -> bool {
        // On Windows, if TERM isn't set, then we shouldn't automatically
        // assume that colors aren't allowed. This is unlike Unix environments
        // where TERM is more rigorously set.
        if let Some(k) = env::var_os("TERM") {
            if k == "dumb" {
                return false;
            }
        }
        // If TERM != dumb, then the only way we don't allow colors at this
        // point is if NO_COLOR is set.
        if env::var_os("NO_COLOR").is_some() {
            return false;
        }
        true
    }

    /// Returns true if this choice should forcefully use ANSI color codes.
    ///
    /// It's possible that ANSI is still the correct choice even if this
    /// returns false.
    #[cfg(windows)]
    pub(crate) fn should_force_ansi(&self) -> bool {
        match *self {
            ColorChoice::Always => false,
            ColorChoice::AlwaysAnsi => true,
            ColorChoice::Never => false,
            ColorChoice::Auto => {
                if let Ok(term) = env::var("TERM") {
                    term != "dumb" && term != "cygwin"
                } else {
                    false
                }
            }
        }
    }
}

/// An error that occurs when parsing a `ColorChoice` fails.
#[derive(Clone, Debug)]
pub struct ColorChoiceParseError {
    unknown_choice: String,
}

impl ColorChoiceParseError {
    /// Return the string that couldn't be parsed as a valid color choice.
    pub fn invalid_choice(&self) -> &str {
        &self.unknown_choice
    }
}

impl std::error::Error for ColorChoiceParseError {}

impl fmt::Display for ColorChoiceParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "unrecognized color choice '{}': valid choices are: \
             always, always-ansi, never, auto",
            self.unknown_choice,
        )
    }
}

/// A color specification.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ColorSpec {
    pub(crate) fg_color: Option<Color>,
    pub(crate) bg_color: Option<Color>,
    pub(crate) bold: bool,
    pub(crate) intense: bool,
    pub(crate) underline: bool,
    pub(crate) dimmed: bool,
    pub(crate) italic: bool,
    pub(crate) reset: bool,
    pub(crate) strikethrough: bool,
}

impl Default for ColorSpec {
    fn default() -> ColorSpec {
        ColorSpec {
            fg_color: None,
            bg_color: None,
            bold: false,
            intense: false,
            underline: false,
            dimmed: false,
            italic: false,
            reset: true,
            strikethrough: false,
        }
    }
}

impl ColorSpec {
    /// Create a new color specification that has no colors or styles.
    pub fn new() -> ColorSpec {
        ColorSpec::default()
    }

    /// Get the foreground color.
    pub fn fg(&self) -> Option<&Color> {
        self.fg_color.as_ref()
    }

    /// Set the foreground color.
    pub fn set_fg(&mut self, color: Option<Color>) -> &mut ColorSpec {
        self.fg_color = color;
        self
    }

    /// Get the background color.
    pub fn bg(&self) -> Option<&Color> {
        self.bg_color.as_ref()
    }

    /// Set the background color.
    pub fn set_bg(&mut self, color: Option<Color>) -> &mut ColorSpec {
        self.bg_color = color;
        self
    }

    /// Get whether this is bold or not.
    pub fn bold(&self) -> bool {
        self.bold
    }

    /// Set whether the text is bolded or not.
    pub fn set_bold(&mut self, yes: bool) -> &mut ColorSpec {
        self.bold = yes;
        self
    }

    /// Get whether this is dimmed or not.
    pub fn dimmed(&self) -> bool {
        self.dimmed
    }

    /// Set whether the text is dimmed or not.
    pub fn set_dimmed(&mut self, yes: bool) -> &mut ColorSpec {
        self.dimmed = yes;
        self
    }

    /// Get whether this is italic or not.
    pub fn italic(&self) -> bool {
        self.italic
    }

    /// Set whether the text is italicized or not.
    pub fn set_italic(&mut self, yes: bool) -> &mut ColorSpec {
        self.italic = yes;
        self
    }

    /// Get whether this is underline or not.
    pub fn underline(&self) -> bool {
        self.underline
    }

    /// Set whether the text is underlined or not.
    pub fn set_underline(&mut self, yes: bool) -> &mut ColorSpec {
        self.underline = yes;
        self
    }

    /// Get whether this is strikethrough or not.
    pub fn strikethrough(&self) -> bool {
        self.strikethrough
    }

    /// Set whether the text is strikethrough or not.
    pub fn set_strikethrough(&mut self, yes: bool) -> &mut ColorSpec {
        self.strikethrough = yes;
        self
    }

    /// Get whether reset is enabled or not.
    ///
    /// reset is enabled by default. When disabled and using ANSI escape
    /// sequences, a "reset" code will be emitted every time a `ColorSpec`'s
    /// settings are applied.
    pub fn reset(&self) -> bool {
        self.reset
    }

    /// Set whether to reset the terminal whenever color settings are applied.
    ///
    /// reset is enabled by default. When disabled and using ANSI escape
    /// sequences, a "reset" code will be emitted every time a `ColorSpec`'s
    /// settings are applied.
    ///
    /// Typically this is useful if callers have a requirement to more
    /// scrupulously manage the exact sequence of escape codes that are emitted
    /// when using ANSI for colors.
    pub fn set_reset(&mut self, yes: bool) -> &mut ColorSpec {
        self.reset = yes;
        self
    }

    /// Get whether this is intense or not.
    ///
    /// On Unix-like systems, this will output the ANSI escape sequence
    /// that will print a high-intensity version of the color
    /// specified.
    ///
    /// On Windows systems, this will output the ANSI escape sequence
    /// that will print a brighter version of the color specified.
    pub fn intense(&self) -> bool {
        self.intense
    }

    /// Set whether the text is intense or not.
    ///
    /// On Unix-like systems, this will output the ANSI escape sequence
    /// that will print a high-intensity version of the color
    /// specified.
    ///
    /// On Windows systems, this will output the ANSI escape sequence
    /// that will print a brighter version of the color specified.
    pub fn set_intense(&mut self, yes: bool) -> &mut ColorSpec {
        self.intense = yes;
        self
    }

    /// Returns true if this color specification has no colors or styles.
    pub fn is_none(&self) -> bool {
        self.fg_color.is_none()
            && self.bg_color.is_none()
            && !self.bold
            && !self.underline
            && !self.dimmed
            && !self.italic
            && !self.intense
            && !self.strikethrough
    }

    /// Clears this color specification so that it has no color/style settings.
    pub fn clear(&mut self) {
        self.fg_color = None;
        self.bg_color = None;
        self.bold = false;
        self.underline = false;
        self.intense = false;
        self.dimmed = false;
        self.italic = false;
        self.strikethrough = false;
    }
}

/// The set of available colors for the terminal foreground/background.
///
/// The `Ansi256` and `Rgb` colors will only output the correct codes when
/// paired with the `Ansi` `WriteColor` implementation.
///
/// This set may expand over time.
///
/// This type has a `FromStr` impl that can parse colors from their human
/// readable form. The format is as follows:
///
/// 1. Any of the explicitly listed colors in English. They are matched
///    case insensitively.
/// 2. A single 8-bit integer, in either decimal or hexadecimal format.
/// 3. A triple of 8-bit integers separated by a comma, where each integer is
///    in decimal or hexadecimal format.
///
/// Hexadecimal numbers are written with a `0x` prefix.
#[allow(missing_docs)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Color {
    Black,
    Blue,
    Green,
    Red,
    Cyan,
    Magenta,
    Yellow,
    White,
    Ansi256(u8),
    Rgb(u8, u8, u8),
}

impl Color {
    /// Parses a numeric color string, either ANSI or RGB.
    fn from_str_numeric(s: &str) -> Result<Color, ParseColorError> {
        // The "ansi256" format is a single number (decimal or hex)
        // corresponding to one of 256 colors.
        //
        // The "rgb" format is a triple of numbers (decimal or hex) delimited
        // by a comma corresponding to one of 256^3 colors.

        fn parse_number(s: &str) -> Option<u8> {
            if let Some(stripped) = s.strip_prefix("0x") {
                u8::from_str_radix(stripped, 16).ok()
            } else {
                s.parse::<u8>().ok()
            }
        }

        let codes: Vec<&str> = s.split(',').collect();
        if codes.len() == 1 {
            if let Some(n) = parse_number(codes[0]) {
                Ok(Color::Ansi256(n))
            } else if s.chars().all(|c| c.is_ascii_hexdigit()) {
                Err(ParseColorError {
                    kind: ParseColorErrorKind::InvalidAnsi256,
                    given: s.to_string(),
                })
            } else {
                Err(ParseColorError {
                    kind: ParseColorErrorKind::InvalidName,
                    given: s.to_string(),
                })
            }
        } else if codes.len() == 3 {
            let mut v = vec![];
            for code in codes {
                let n = parse_number(code).ok_or_else(|| ParseColorError {
                    kind: ParseColorErrorKind::InvalidRgb,
                    given: s.to_string(),
                })?;
                v.push(n);
            }
            Ok(Color::Rgb(v[0], v[1], v[2]))
        } else {
            Err(if s.contains(",") {
                ParseColorError {
                    kind: ParseColorErrorKind::InvalidRgb,
                    given: s.to_string(),
                }
            } else {
                ParseColorError {
                    kind: ParseColorErrorKind::InvalidName,
                    given: s.to_string(),
                }
            })
        }
    }
}

/// An error from parsing an invalid color specification.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParseColorError {
    kind: ParseColorErrorKind,
    given: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ParseColorErrorKind {
    InvalidName,
    InvalidAnsi256,
    InvalidRgb,
}

impl ParseColorError {
    /// Return the string that couldn't be parsed as a valid color.
    pub fn invalid(&self) -> &str {
        &self.given
    }
}

impl std::error::Error for ParseColorError {
    fn description(&self) -> &str {
        use self::ParseColorErrorKind::*;
        match self.kind {
            InvalidName => "unrecognized color name",
            InvalidAnsi256 => "invalid ansi256 color number",
            InvalidRgb => "invalid RGB color triple",
        }
    }
}

impl fmt::Display for ParseColorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use self::ParseColorErrorKind::*;
        match self.kind {
            InvalidName => write!(
                f,
                "unrecognized color name '{}'. Choose from: \
                 black, blue, green, red, cyan, magenta, yellow, \
                 white",
                self.given
            ),
            InvalidAnsi256 => write!(
                f,
                "unrecognized ansi256 color number, \
                 should be '[0-255]' (or a hex number), but is '{}'",
                self.given
            ),
            InvalidRgb => write!(
                f,
                "unrecognized RGB color triple, \
                 should be '[0-255],[0-255],[0-255]' (or a hex \
                 triple), but is '{}'",
                self.given
            ),
        }
    }
}

impl FromStr for Color {
    type Err = ParseColorError;

    fn from_str(s: &str) -> Result<Color, ParseColorError> {
        match &*s.to_lowercase() {
            "black" => Ok(Color::Black),
            "blue" => Ok(Color::Blue),
            "green" => Ok(Color::Green),
            "red" => Ok(Color::Red),
            "cyan" => Ok(Color::Cyan),
            "magenta" => Ok(Color::Magenta),
            "yellow" => Ok(Color::Yellow),
            "white" => Ok(Color::White),
            _ => Color::from_str_numeric(s),
        }
    }
}

/// An error from parsing an invalid color specification.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ColorSpecParseError {
    /// An error parsing a color.
    InvalidColor(ParseColorError),
}

impl std::error::Error for ColorSpecParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ColorSpecParseError::InvalidColor(e) => Some(e),
        }
    }
}

impl fmt::Display for ColorSpecParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ColorSpecParseError::InvalidColor(e) => write!(f, "{e}"),
        }
    }
}

impl FromStr for ColorSpec {
    type Err = ColorSpecParseError;

    fn from_str(spec: &str) -> Result<ColorSpec, ColorSpecParseError> {
        let mut color_spec = ColorSpec::new();
        for part in spec.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }

            if let Some(color_str) = part.strip_prefix("fg:") {
                let color = Color::from_str(color_str)
                    .map_err(ColorSpecParseError::InvalidColor)?;
                color_spec.set_fg(Some(color));
            } else if let Some(color_str) = part.strip_prefix("bg:") {
                let color = Color::from_str(color_str)
                    .map_err(ColorSpecParseError::InvalidColor)?;
                color_spec.set_bg(Some(color));
            } else if part == "bold" {
                color_spec.set_bold(true);
            } else if part == "dimmed" {
                color_spec.set_dimmed(true);
            } else if part == "underline" {
                color_spec.set_underline(true);
            } else if part == "italic" {
                color_spec.set_italic(true);
            } else if part == "intense" {
                color_spec.set_intense(true);
            } else if part == "strikethrough" {
                color_spec.set_strikethrough(true);
            } else if part == "reset" {
                color_spec.set_reset(true);
            } else if part == "noreset" {
                color_spec.set_reset(false);
            } else {
                let color = Color::from_str(part)
                    .map_err(ColorSpecParseError::InvalidColor)?;
                color_spec.set_fg(Some(color));
            }
        }
        Ok(color_spec)
    }
}

/// A hyperlink specification.
#[derive(Clone, Debug)]
pub struct HyperlinkSpec<'a> {
    uri: Option<&'a [u8]>,
}

impl<'a> HyperlinkSpec<'a> {
    /// Creates a new hyperlink specification.
    pub fn open(uri: &'a [u8]) -> HyperlinkSpec<'a> {
        HyperlinkSpec { uri: Some(uri) }
    }

    /// Creates a hyperlink specification representing no hyperlink.
    pub fn close() -> HyperlinkSpec<'a> {
        HyperlinkSpec { uri: None }
    }

    /// Returns the URI of the hyperlink if one is attached to this spec.
    pub fn uri(&self) -> Option<&'a [u8]> {
        self.uri
    }
}

impl fmt::Display for ColorSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut buf = Vec::new();
        crate::ansi::ansi_spec(&mut buf, self).map_err(|_| fmt::Error)?;
        write!(f, "{}", String::from_utf8_lossy(&buf))
    }
}
