//! Demonstration of Cyrup.ai themed colored output
//!
//! Shows off the beautiful colored terminal output with Cyrup.ai branding.

use termcolor::{
    ColoredMessage, CyrupTheme, ThemeConfig, colored_println, error_x, info_i,
    set_global_theme, success_check, warning_triangle,
};

fn main() {
    // Set Cyrup.ai default theme
    set_global_theme(ThemeConfig::Default);

    // Display Cyrup.ai header
    println!();
    ColoredMessage::cyrup_header().println().unwrap();
    println!();

    // Demonstrate semantic colors
    colored_println!(primary: "🔷 Primary brand color - perfect for headers and important text");
    colored_println!(secondary: "🔹 Secondary brand color - great for accents and highlights");
    colored_println!(accent: "🟠 Accent color - draws attention to key elements");
    println!();

    // Demonstrate status indicators
    success_check!("Model initialization completed successfully");
    info_i!("Processing 1,247 tokens from user input");
    warning_triangle!("Rate limit approaching - requests may be throttled");
    error_x!("Authentication failed - check your API key");
    println!();

    // Demonstrate complex structured output
    ColoredMessage::new()
        .primary("=== Completion Response ===")
        .newline()
        .text_primary("The solution involves three key steps:")
        .newline()
        .accent("  1. ")
        .text_secondary("Initialize the model with your API key")
        .newline()
        .accent("  2. ")
        .text_secondary(
            "Process the user's request through the completion pipeline",
        )
        .newline()
        .accent("  3. ")
        .text_secondary("Stream the response with proper error handling")
        .newline()
        .println()
        .unwrap();

    // Demonstrate progress indicator
    for i in 1..=5 {
        ColoredMessage::progress(i, 5, "Processing completion chunks...")
            .println()
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
    println!();

    // Demonstrate text styles
    ColoredMessage::new()
        .primary("Text Styles: ")
        .text_primary("Normal text, ")
        .text_primary("bold text")
        .bold()
        .text_primary(", ")
        .text_primary("italic text")
        .italic()
        .text_primary(", and ")
        .text_muted("muted text for less important details")
        .println()
        .unwrap();
    println!();

    // Demonstrate custom theme
    let custom_theme = CyrupTheme::builder()
        .primary(termcolor::Color::Rgb(255, 100, 50)) // Custom orange
        .success(termcolor::Color::Rgb(0, 255, 100)) // Custom green
        .build();

    set_global_theme(ThemeConfig::Custom(custom_theme));
    colored_println!(primary: "🎨 Custom theme with orange primary color");
    success_check!("Custom green success color");
    println!();

    // Reset to default
    set_global_theme(ThemeConfig::Default);
    colored_println!(info: "✨ Back to beautiful Cyrup.ai default colors");
    println!();
}
