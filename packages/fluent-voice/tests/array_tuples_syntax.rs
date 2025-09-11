//! Test array-tuples syntax in fluent-voice builders

use fluent_voice::prelude::*;

#[tokio::test]
async fn test_array_tuples_syntax() {
    // Test that array-tuples syntax compiles and works
    let _builder = FluentVoice::tts()
        .conversation()
        .additional_params([("beta", "true"), ("debug", "false")])
        .metadata([("key", "val"), ("foo", "bar")]);

    // Test single key-value pair
    let _builder2 = FluentVoice::tts()
        .conversation()
        .additional_params([("single", "value")])
        .metadata([("meta", "data")]);

    // Test empty array
    let _builder3 = FluentVoice::tts()
        .conversation()
        .additional_params([])
        .metadata([]);

    // All syntax variations should compile without errors
    println!("✅ Array-tuples syntax test passed!");
}
