//! Zero-allocation theme system for tasteful colored terminal output
//!
//! Provides the Cyrup.ai brand theme with semantic colors and flexible customization.

use crate::{Color, ColorSpec, WriteColor};
use std::io;

/// Cyrup.ai brand theme with modern AI aesthetic
#[derive(Debug, Clone)]
pub struct CyrupTheme {
    // Primary brand colors
    pub primary: Color,
    pub secondary: Color,
    pub accent: Color,

    // Semantic colors
    pub success: Color,
    pub warning: Color,
    pub error: Color,
    pub info: Color,
    pub debug: Color,

    // Text colors
    pub text_primary: Color,
    pub text_secondary: Color,
    pub text_muted: Color,

    // Background accents
    pub bg_highlight: Color,
    pub bg_code: Color,
}

impl Default for CyrupTheme {
    /// Default Cyrup.ai brand colors - modern, professional AI aesthetic
    fn default() -> Self {
        Self {
            // Primary brand colors
            primary: Color::Rgb(0, 150, 255), // Cyrup blue
            secondary: Color::Rgb(64, 224, 208), // Turquoise accent
            accent: Color::Rgb(255, 165, 0),  // Orange highlight

            // Semantic colors
            success: Color::Green,  // Operations completed
            warning: Color::Yellow, // Cautions, fallbacks
            error: Color::Red,      // Failures, critical issues
            info: Color::Cyan,      // General information
            debug: Color::Magenta,  // Debug/development output

            // Text colors
            text_primary: Color::White, // Main text
            text_secondary: Color::Rgb(200, 200, 200), // Secondary text
            text_muted: Color::Rgb(128, 128, 128), // Muted/disabled text

            // Background accents
            bg_highlight: Color::Rgb(30, 30, 30), // Subtle highlights
            bg_code: Color::Rgb(40, 40, 40),      // Code blocks
        }
    }
}

impl CyrupTheme {
    /// Create new theme builder for customization
    #[inline(always)]
    pub fn builder() -> CyrupThemeBuilder {
        CyrupThemeBuilder::new()
    }

    /// Get color by semantic name
    #[inline(always)]
    pub fn get_color(&self, semantic: SemanticColor) -> Color {
        match semantic {
            SemanticColor::Primary => self.primary,
            SemanticColor::Secondary => self.secondary,
            SemanticColor::Accent => self.accent,
            SemanticColor::Success => self.success,
            SemanticColor::Warning => self.warning,
            SemanticColor::Error => self.error,
            SemanticColor::Info => self.info,
            SemanticColor::Debug => self.debug,
            SemanticColor::TextPrimary => self.text_primary,
            SemanticColor::TextSecondary => self.text_secondary,
            SemanticColor::TextMuted => self.text_muted,
            SemanticColor::BgHighlight => self.bg_highlight,
            SemanticColor::BgCode => self.bg_code,
        }
    }

    /// Create ColorSpec for semantic color with optional bold/italic
    #[inline(always)]
    pub fn spec(&self, semantic: SemanticColor) -> ColorSpec {
        let mut spec = ColorSpec::new();
        spec.set_fg(Some(self.get_color(semantic)));
        spec
    }

    /// Create bold ColorSpec for semantic color
    #[inline(always)]
    pub fn bold_spec(&self, semantic: SemanticColor) -> ColorSpec {
        let mut spec = self.spec(semantic);
        spec.set_bold(true);
        spec
    }

    /// Create italic ColorSpec for semantic color
    #[inline(always)]
    pub fn italic_spec(&self, semantic: SemanticColor) -> ColorSpec {
        let mut spec = self.spec(semantic);
        spec.set_italic(true);
        spec
    }
}

/// Zero-allocation theme builder with fluent API
#[derive(Debug, Clone)]
pub struct CyrupThemeBuilder {
    theme: CyrupTheme,
}

impl Default for CyrupThemeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CyrupThemeBuilder {
    /// Create new theme builder with Cyrup.ai defaults
    #[inline(always)]
    pub fn new() -> Self {
        Self { theme: CyrupTheme::default() }
    }

    /// Set primary brand color
    #[inline(always)]
    pub fn primary(mut self, color: Color) -> Self {
        self.theme.primary = color;
        self
    }

    /// Set secondary brand color
    #[inline(always)]
    pub fn secondary(mut self, color: Color) -> Self {
        self.theme.secondary = color;
        self
    }

    /// Set accent color
    #[inline(always)]
    pub fn accent(mut self, color: Color) -> Self {
        self.theme.accent = color;
        self
    }

    /// Set success color
    #[inline(always)]
    pub fn success(mut self, color: Color) -> Self {
        self.theme.success = color;
        self
    }

    /// Set warning color
    #[inline(always)]
    pub fn warning(mut self, color: Color) -> Self {
        self.theme.warning = color;
        self
    }

    /// Set error color
    #[inline(always)]
    pub fn error(mut self, color: Color) -> Self {
        self.theme.error = color;
        self
    }

    /// Set info color
    #[inline(always)]
    pub fn info(mut self, color: Color) -> Self {
        self.theme.info = color;
        self
    }

    /// Set debug color
    #[inline(always)]
    pub fn debug(mut self, color: Color) -> Self {
        self.theme.debug = color;
        self
    }

    /// Set primary text color
    #[inline(always)]
    pub fn text_primary(mut self, color: Color) -> Self {
        self.theme.text_primary = color;
        self
    }

    /// Set secondary text color
    #[inline(always)]
    pub fn text_secondary(mut self, color: Color) -> Self {
        self.theme.text_secondary = color;
        self
    }

    /// Set muted text color
    #[inline(always)]
    pub fn text_muted(mut self, color: Color) -> Self {
        self.theme.text_muted = color;
        self
    }

    /// Set background highlight color
    #[inline(always)]
    pub fn bg_highlight(mut self, color: Color) -> Self {
        self.theme.bg_highlight = color;
        self
    }

    /// Set code background color
    #[inline(always)]
    pub fn bg_code(mut self, color: Color) -> Self {
        self.theme.bg_code = color;
        self
    }

    /// Build the final theme
    #[inline(always)]
    pub fn build(self) -> CyrupTheme {
        self.theme
    }
}

/// Semantic color names for theme consistency
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemanticColor {
    Primary,
    Secondary,
    Accent,
    Success,
    Warning,
    Error,
    Info,
    Debug,
    TextPrimary,
    TextSecondary,
    TextMuted,
    BgHighlight,
    BgCode,
}

impl SemanticColor {
    /// Parse semantic color from string name (for macros)
    #[inline(always)]
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "primary" => Some(SemanticColor::Primary),
            "secondary" => Some(SemanticColor::Secondary),
            "accent" => Some(SemanticColor::Accent),
            "success" => Some(SemanticColor::Success),
            "warning" => Some(SemanticColor::Warning),
            "error" => Some(SemanticColor::Error),
            "info" => Some(SemanticColor::Info),
            "debug" => Some(SemanticColor::Debug),
            "text_primary" => Some(SemanticColor::TextPrimary),
            "text_secondary" => Some(SemanticColor::TextSecondary),
            "text_muted" => Some(SemanticColor::TextMuted),
            "bg_highlight" => Some(SemanticColor::BgHighlight),
            "bg_code" => Some(SemanticColor::BgCode),
            _ => None,
        }
    }
}

/// Theme configuration options
#[derive(Debug, Clone, Default)]
pub enum ThemeConfig {
    /// Use default Cyrup.ai theme
    #[default]
    Default,
    /// No colors, plain text only
    None,
    /// Monochrome with formatting but no colors
    Monochrome,
    /// Custom theme
    Custom(CyrupTheme),
    /// Auto-detect terminal capabilities
    Auto,
    /// High contrast theme for development
    Development,
    /// Subtle professional theme for production
    Production,
}

impl ThemeConfig {
    /// Get the actual theme for this configuration
    #[inline(always)]
    pub fn resolve(&self) -> Option<CyrupTheme> {
        match self {
            ThemeConfig::Default => Some(CyrupTheme::default()),
            ThemeConfig::None => None,
            ThemeConfig::Monochrome => Some(monochrome_theme()),
            ThemeConfig::Custom(theme) => Some(theme.clone()),
            ThemeConfig::Auto => Some(auto_detect_theme()),
            ThemeConfig::Development => Some(development_theme()),
            ThemeConfig::Production => Some(production_theme()),
        }
    }
}

/// Monochrome theme (no colors, but formatting)
fn monochrome_theme() -> CyrupTheme {
    CyrupTheme {
        primary: Color::White,
        secondary: Color::White,
        accent: Color::White,
        success: Color::White,
        warning: Color::White,
        error: Color::White,
        info: Color::White,
        debug: Color::White,
        text_primary: Color::White,
        text_secondary: Color::White,
        text_muted: Color::White,
        bg_highlight: Color::Black,
        bg_code: Color::Black,
    }
}

/// Auto-detect theme based on terminal capabilities
fn auto_detect_theme() -> CyrupTheme {
    let capabilities = detect_terminal_capabilities();

    match capabilities {
        TerminalCapabilities::TrueColor => truecolor_theme(),
        TerminalCapabilities::Color256 => color256_theme(),
        TerminalCapabilities::Basic => basic_ansi_theme(),
    }
}

#[derive(Debug, Clone, Copy)]
enum TerminalCapabilities {
    TrueColor, // 24-bit RGB support
    Color256,  // 256 color palette
    Basic,     // 8/16 ANSI colors
}

fn detect_terminal_capabilities() -> TerminalCapabilities {
    // Check COLORTERM environment variable
    if let Ok(colorterm) = std::env::var("COLORTERM") {
        if colorterm == "truecolor" || colorterm == "24bit" {
            return TerminalCapabilities::TrueColor;
        }
    }

    // Check TERM environment variable
    if let Ok(term) = std::env::var("TERM") {
        if term.contains("256")
            || term.contains("xterm")
            || term.contains("screen")
        {
            return TerminalCapabilities::Color256;
        }
    }

    // Check terminal program specific variables
    if std::env::var("TERM_PROGRAM")
        .map(|v| v == "iTerm.app" || v == "vscode")
        .unwrap_or(false)
    {
        return TerminalCapabilities::TrueColor;
    }

    TerminalCapabilities::Basic
}

fn truecolor_theme() -> CyrupTheme {
    CyrupTheme {
        primary: Color::Rgb(0, 150, 255), // Bright blue
        secondary: Color::Rgb(255, 100, 0), // Orange
        accent: Color::Rgb(100, 255, 150), // Bright green
        success: Color::Rgb(0, 200, 100), // Success green
        warning: Color::Rgb(255, 200, 0), // Warning yellow
        error: Color::Rgb(255, 50, 50),   // Error red
        info: Color::Rgb(150, 150, 255),  // Info blue
        debug: Color::Rgb(255, 100, 255), // Debug magenta
        text_primary: Color::Rgb(240, 240, 240), // Light text
        text_secondary: Color::Rgb(200, 200, 200), // Secondary text
        text_muted: Color::Rgb(150, 150, 150), // Muted text
        bg_highlight: Color::Rgb(40, 40, 60), // Highlight background
        bg_code: Color::Rgb(30, 30, 40),  // Code background
    }
}

fn color256_theme() -> CyrupTheme {
    CyrupTheme {
        primary: Color::Ansi256(39), // Bright blue (256-color)
        secondary: Color::Ansi256(208), // Orange (256-color)
        accent: Color::Ansi256(82),  // Bright green (256-color)
        success: Color::Ansi256(46), // Success green (256-color)
        warning: Color::Ansi256(226), // Warning yellow (256-color)
        error: Color::Ansi256(196),  // Error red (256-color)
        info: Color::Ansi256(117),   // Info blue (256-color)
        debug: Color::Ansi256(201),  // Debug magenta (256-color)
        text_primary: Color::Ansi256(252), // Light text (256-color)
        text_secondary: Color::Ansi256(249), // Secondary text (256-color)
        text_muted: Color::Ansi256(240), // Muted text (256-color)
        bg_highlight: Color::Ansi256(237), // Highlight background (256-color)
        bg_code: Color::Ansi256(236), // Code background (256-color)
    }
}

fn basic_ansi_theme() -> CyrupTheme {
    CyrupTheme {
        primary: Color::Blue,
        secondary: Color::Magenta,
        accent: Color::Green,
        success: Color::Green,
        warning: Color::Yellow,
        error: Color::Red,
        info: Color::Cyan,
        debug: Color::Magenta,
        text_primary: Color::White,
        text_secondary: Color::White,
        text_muted: Color::White,
        bg_highlight: Color::Black,
        bg_code: Color::Black,
    }
}

/// High contrast development theme
fn development_theme() -> CyrupTheme {
    CyrupTheme {
        primary: Color::Rgb(100, 200, 255), // Bright blue
        secondary: Color::Rgb(100, 255, 200), // Bright cyan
        accent: Color::Rgb(255, 200, 100),  // Bright orange
        success: Color::Rgb(100, 255, 100), // Bright green
        warning: Color::Rgb(255, 255, 100), // Bright yellow
        error: Color::Rgb(255, 100, 100),   // Bright red
        info: Color::Rgb(150, 150, 255),    // Light blue
        debug: Color::Rgb(255, 150, 255),   // Light magenta
        text_primary: Color::White,
        text_secondary: Color::Rgb(220, 220, 220),
        text_muted: Color::Rgb(150, 150, 150),
        bg_highlight: Color::Rgb(50, 50, 50),
        bg_code: Color::Rgb(20, 20, 20),
    }
}

/// Subtle professional production theme
fn production_theme() -> CyrupTheme {
    CyrupTheme {
        primary: Color::Rgb(70, 130, 200), // Muted blue
        secondary: Color::Rgb(70, 200, 180), // Muted teal
        accent: Color::Rgb(200, 140, 70),  // Muted orange
        success: Color::Rgb(70, 180, 70),  // Muted green
        warning: Color::Rgb(200, 180, 70), // Muted yellow
        error: Color::Rgb(180, 70, 70),    // Muted red
        info: Color::Rgb(100, 140, 180),   // Muted cyan
        debug: Color::Rgb(140, 100, 180),  // Muted magenta
        text_primary: Color::Rgb(240, 240, 240),
        text_secondary: Color::Rgb(180, 180, 180),
        text_muted: Color::Rgb(120, 120, 120),
        bg_highlight: Color::Rgb(25, 25, 25),
        bg_code: Color::Rgb(35, 35, 35),
    }
}

/// Set global theme configuration
pub fn set_global_theme(config: ThemeConfig) {
    // Since OnceLock can only be set once, we use a workaround with thread_local for runtime changes
    THEME_CONFIG.with(|theme| {
        *theme.borrow_mut() = config;
    });
}

/// Get current global theme configuration
pub fn get_global_theme() -> ThemeConfig {
    THEME_CONFIG.with(|theme| theme.borrow().clone())
}

/// Get current resolved theme (actual CyrupTheme)
pub fn get_current_theme() -> Option<CyrupTheme> {
    get_global_theme().resolve()
}

/// Thread-local storage for theme configuration (allows runtime changes)
use std::cell::RefCell;
thread_local! {
    static THEME_CONFIG: RefCell<ThemeConfig> = const { RefCell::new(ThemeConfig::Default) };
}

/// Execute code with temporary theme override
pub fn with_temporary_theme<F, R>(config: ThemeConfig, f: F) -> R
where
    F: FnOnce() -> R,
{
    let original = get_global_theme();
    set_global_theme(config);
    let result = f();
    set_global_theme(original);
    result
}

/// Write text with semantic color to any WriteColor implementation
#[inline(always)]
pub fn write_colored<W: WriteColor>(
    writer: &mut W,
    semantic: SemanticColor,
    text: &str,
) -> io::Result<()> {
    // Note: std::io::Write is available via WriteColor supertrait bound
    if let Some(theme) = get_current_theme() {
        let spec = theme.spec(semantic);
        writer.set_color(&spec)?;
        write!(writer, "{text}")?;
        writer.reset()?;
    } else {
        write!(writer, "{text}")?;
    }
    Ok(())
}

/// Write bold text with semantic color
#[inline(always)]
pub fn write_colored_bold<W: WriteColor>(
    writer: &mut W,
    semantic: SemanticColor,
    text: &str,
) -> io::Result<()> {
    // Note: std::io::Write is available via WriteColor supertrait bound
    if let Some(theme) = get_current_theme() {
        let spec = theme.bold_spec(semantic);
        writer.set_color(&spec)?;
        write!(writer, "{text}")?;
        writer.reset()?;
    } else {
        write!(writer, "{text}")?;
    }
    Ok(())
}

/// Write italic text with semantic color
#[inline(always)]
pub fn write_colored_italic<W: WriteColor>(
    writer: &mut W,
    semantic: SemanticColor,
    text: &str,
) -> io::Result<()> {
    // Note: std::io::Write is available via WriteColor supertrait bound
    if let Some(theme) = get_current_theme() {
        let spec = theme.italic_spec(semantic);
        writer.set_color(&spec)?;
        write!(writer, "{text}")?;
        writer.reset()?;
    } else {
        write!(writer, "{text}")?;
    }
    Ok(())
}
