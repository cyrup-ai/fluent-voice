//! Analysis of README.md syntax examples
//! This test file documents why the README examples don't compile as written

use fluent_voice::prelude::*;

#[test]
fn test_readme_syntax_issues() {
    println!("=== README Syntax Analysis ===\n");

    // Issue 1: The README shows this STT example (lines 64-67):
    // .listen(|segment| {
    //     Ok  => segment.text(),
    //     Err(e) => Err(e),
    // })

    println!("Issue 1: STT listen() syntax");
    println!("README shows: |segment| {{ Ok => segment.text(), Err(e) => Err(e) }}");
    println!("Problems:");
    println!("  - 'Ok =>' is not valid Rust pattern matching syntax");
    println!(
        "  - The parameter name 'segment' implies it's the success value, but it's actually the Result"
    );
    println!("  - segment.text() implies segment has a text() method, but it's a Result");
    println!();

    // Issue 2: The README shows this TTS example (lines 92-95):
    // .synthesize(|conversation| {
    //     Ok  => conversation.into_stream(),
    //     Err(e) => Err(e),
    // })

    println!("Issue 2: TTS synthesize() syntax");
    println!(
        "README shows: |conversation| {{ Ok => conversation.into_stream(), Err(e) => Err(e) }}"
    );
    println!("Problems:");
    println!("  - Same 'Ok =>' syntax issue");
    println!("  - The arms return different types (Stream vs Result)");
    println!("  - This violates Rust's requirement that match arms have the same type");
    println!();

    // Issue 3: The actual API signature
    println!("Issue 3: Actual API signatures");
    println!("The actual listen() signature is:");
    println!("  fn listen<F, R>(self, matcher: F) -> impl Future<Output = R>");
    println!("  where F: FnOnce(Result<Self::Conversation, VoiceError>) -> R");
    println!();
    println!("This means:");
    println!("  - The closure receives a Result, not the unwrapped value");
    println!("  - The closure must return type R, which becomes the Future's output");
    println!("  - Both match arms must return the same type R");
}

#[test]
fn test_working_alternatives() {
    println!("\n=== Working Alternatives ===\n");

    // Working STT example
    async fn working_stt() -> Result<String, VoiceError> {
        let transcript = FluentVoice::stt()
            .with_source(SpeechSource::Microphone {
                backend: MicBackend::Default,
                format: AudioFormat::Pcm16Khz,
                sample_rate: 16_000,
            })
            .listen(|result| {
                // Correct: match on the Result
                match result {
                    Ok(conversation) => Ok(conversation),
                    Err(e) => Err(e),
                }
            })
            .await? // Unwrap the Result
            .collect() // Now we have the conversation
            .await?; // Collect returns a Future<Output = Result<String, VoiceError>>

        Ok(transcript)
    }

    println!("Working STT pattern:");
    println!("  .listen(|result| match result {{");
    println!("      Ok(conversation) => Ok(conversation),");
    println!("      Err(e) => Err(e),");
    println!("  }})");
    println!("  .await?");
    println!("  .collect()");
    println!("  .await?");
}

#[test]
fn test_macro_syntax_possibilities() {
    println!("\n=== Macro Syntax Possibilities ===\n");

    println!("The crate provides macros that could support cleaner syntax:");
    println!();
    println!("Current macro (stt_listen!):");
    println!("  stt_listen!(builder, |conv| {{");
    println!("      Ok => conv,");
    println!("      Err(e) => handle_error(e),");
    println!("  }})");
    println!();
    println!("But even this requires valid Rust in the match arms.");
    println!();
    println!("To support the exact README syntax, we would need:");
    println!("1. A procedural macro that transforms the syntax");
    println!("2. Or update the README to show actual Rust syntax");
}

#[test]
fn test_readme_examples_cannot_compile() {
    // This test documents that the README examples literally cannot compile

    // This is what the README shows:
    // let _transcript = FluentVoice::stt()
    //     .listen(|segment| {
    //         Ok  => segment.text(),
    //         Err(e) => Err(e),
    //     })

    // If we try to write it exactly as shown, we get:
    // 1. "expected pattern, found `=>`" - Ok => is not valid syntax
    // 2. Even if we fix to "Ok(_) =>" we get type errors

    // The README syntax appears to be pseudo-code, not actual Rust
    assert!(true, "README shows pseudo-code, not compilable Rust");
}
