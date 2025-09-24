//! Zero-allocation colored output macros with perfect ergonomics
//!
//! Provides println!-style macros with semantic coloring for beautiful terminal output.

/// Print colored text to stdout with semantic color
#[macro_export]
macro_rules! colored_print {
    // Simple semantic color
    ($semantic:ident: $($arg:tt)*) => {{
        use std::io::Write;
        let text = format!($($arg)*);
        let semantic = $crate::theme::SemanticColor::from_name(stringify!($semantic));
        if let Some(semantic) = semantic {
            let mut stdout = $crate::BufferedStandardStream::stdout($crate::ColorChoice::Auto);
            let _ = $crate::theme::write_colored(&mut stdout, semantic, &text);
            let _ = stdout.flush();
        } else {
            print!($($arg)*);
        }
    }};

    // No color specified, use text_primary
    ($($arg:tt)*) => {{
        use std::io::Write;
        let text = format!($($arg)*);
        let mut stdout = $crate::BufferedStandardStream::stdout($crate::ColorChoice::Auto);
        let _ = $crate::theme::write_colored(&mut stdout, $crate::theme::SemanticColor::TextPrimary, &text);
        let _ = stdout.flush();
    }};
}

/// Print colored text to stdout with semantic color and newline
#[macro_export]
macro_rules! colored_println {
    // Simple semantic color
    ($semantic:ident: $($arg:tt)*) => {{
        use std::io::Write;
        let text = format!($($arg)*);
        let semantic = $crate::theme::SemanticColor::from_name(stringify!($semantic));
        if let Some(semantic) = semantic {
            let mut stdout = $crate::BufferedStandardStream::stdout($crate::ColorChoice::Auto);
            let _ = $crate::theme::write_colored(&mut stdout, semantic, &text);
            let _ = writeln!(stdout);
            let _ = stdout.flush();
        } else {
            println!($($arg)*);
        }
    }};

    // No color specified, use text_primary
    ($($arg:tt)*) => {{
        use std::io::Write;
        let text = format!($($arg)*);
        let mut stdout = $crate::BufferedStandardStream::stdout($crate::ColorChoice::Auto);
        let _ = $crate::theme::write_colored(&mut stdout, $crate::theme::SemanticColor::TextPrimary, &text);
        let _ = writeln!(stdout);
        let _ = stdout.flush();
    }};
}

/// Print colored text to stderr with semantic color
#[macro_export]
macro_rules! colored_eprint {
    // Simple semantic color
    ($semantic:ident: $($arg:tt)*) => {{
        use std::io::Write;
        let text = format!($($arg)*);
        let semantic = $crate::theme::SemanticColor::from_name(stringify!($semantic));
        if let Some(semantic) = semantic {
            let mut stderr = $crate::BufferedStandardStream::stderr($crate::ColorChoice::Auto);
            let _ = $crate::theme::write_colored(&mut stderr, semantic, &text);
            let _ = stderr.flush();
        } else {
            eprint!($($arg)*);
        }
    }};

    // No color specified, use text_primary
    ($($arg:tt)*) => {{
        use std::io::Write;
        let text = format!($($arg)*);
        let mut stderr = $crate::BufferedStandardStream::stderr($crate::ColorChoice::Auto);
        let _ = $crate::theme::write_colored(&mut stderr, $crate::theme::SemanticColor::TextPrimary, &text);
        let _ = stderr.flush();
    }};
}

/// Print colored text to stderr with semantic color and newline
#[macro_export]
macro_rules! colored_eprintln {
    // Simple semantic color
    ($semantic:ident: $($arg:tt)*) => {{
        use std::io::Write;
        let text = format!($($arg)*);
        let semantic = $crate::theme::SemanticColor::from_name(stringify!($semantic));
        if let Some(semantic) = semantic {
            let mut stderr = $crate::BufferedStandardStream::stderr($crate::ColorChoice::Auto);
            let _ = $crate::theme::write_colored(&mut stderr, semantic, &text);
            let _ = writeln!(stderr);
            let _ = stderr.flush();
        } else {
            eprintln!($($arg)*);
        }
    }};

    // No color specified, use error color for stderr
    ($($arg:tt)*) => {{
        use std::io::Write;
        let text = format!($($arg)*);
        let mut stderr = $crate::BufferedStandardStream::stderr($crate::ColorChoice::Auto);
        let _ = $crate::theme::write_colored(&mut stderr, $crate::theme::SemanticColor::Error, &text);
        let _ = writeln!(stderr);
        let _ = stderr.flush();
    }};
}

/// Print bold colored text to stdout with semantic color
#[macro_export]
macro_rules! colored_print_bold {
    ($semantic:ident: $($arg:tt)*) => {{
        use std::io::Write;
        let text = format!($($arg)*);
        let semantic = $crate::theme::SemanticColor::from_name(stringify!($semantic));
        if let Some(semantic) = semantic {
            let mut stdout = $crate::BufferedStandardStream::stdout($crate::ColorChoice::Auto);
            let _ = $crate::theme::write_colored_bold(&mut stdout, semantic, &text);
            let _ = stdout.flush();
        } else {
            print!($($arg)*);
        }
    }};
}

/// Print bold colored text to stdout with semantic color and newline
#[macro_export]
macro_rules! colored_println_bold {
    ($semantic:ident: $($arg:tt)*) => {{
        use std::io::Write;
        let text = format!($($arg)*);
        let semantic = $crate::theme::SemanticColor::from_name(stringify!($semantic));
        if let Some(semantic) = semantic {
            let mut stdout = $crate::BufferedStandardStream::stdout($crate::ColorChoice::Auto);
            let _ = $crate::theme::write_colored_bold(&mut stdout, semantic, &text);
            let _ = writeln!(stdout);
            let _ = stdout.flush();
        } else {
            println!($($arg)*);
        }
    }};
}

/// Convenient aliases for common use cases
#[macro_export]
macro_rules! success {
    ($($arg:tt)*) => {
        $crate::colored_println!(success: $($arg)*)
    };
}

#[macro_export]
macro_rules! warning {
    ($($arg:tt)*) => {
        $crate::colored_println!(warning: $($arg)*)
    };
}

#[macro_export]
macro_rules! error {
    ($($arg:tt)*) => {
        $crate::colored_eprintln!(error: $($arg)*)
    };
}

#[macro_export]
macro_rules! info {
    ($($arg:tt)*) => {
        $crate::colored_println!(info: $($arg)*)
    };
}

#[macro_export]
macro_rules! debug_colored {
    ($($arg:tt)*) => {
        $crate::colored_println!(debug: $($arg)*)
    };
}

/// Convenient progress indicator
#[macro_export]
macro_rules! progress {
    ($current:expr, $total:expr, $($arg:tt)*) => {{
        let message = format!($($arg)*);
        let progress_msg = $crate::ColoredMessage::progress($current, $total, message);
        let _ = progress_msg.println();
    }};
}

/// Convenient success checkmark
#[macro_export]
macro_rules! success_check {
    ($($arg:tt)*) => {{
        let message = format!($($arg)*);
        let success_msg = $crate::ColoredMessage::success_check(message);
        let _ = success_msg.println();
    }};
}

/// Convenient warning triangle
#[macro_export]
macro_rules! warning_triangle {
    ($($arg:tt)*) => {{
        let message = format!($($arg)*);
        let warning_msg = $crate::ColoredMessage::warning_triangle(message);
        let _ = warning_msg.println();
    }};
}

/// Convenient error X
#[macro_export]
macro_rules! error_x {
    ($($arg:tt)*) => {{
        let message = format!($($arg)*);
        let error_msg = $crate::ColoredMessage::error_x(message);
        let _ = error_msg.eprintln();
    }};
}

/// Convenient info i
#[macro_export]
macro_rules! info_i {
    ($($arg:tt)*) => {{
        let message = format!($($arg)*);
        let info_msg = $crate::ColoredMessage::info_i(message);
        let _ = info_msg.println();
    }};
}

/// Print Cyrup.ai header
#[macro_export]
macro_rules! cyrup_header {
    () => {{
        let header = $crate::ColoredMessage::cyrup_header();
        let _ = header.println();
    }};
}
