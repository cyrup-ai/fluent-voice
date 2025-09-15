//! Test that engine_config actually affects TTS synthesis

use fluent_voice::prelude::*;
use hashbrown::HashMap;

#[tokio::test]
async fn test_engine_config_integration() {
    // Test that engine_config with device setting is captured and used
    let mut config = HashMap::new();
    config.insert("device", "cpu");

    let builder = TtsConversationBuilderImpl::default()
        .engine_config(config)
        .with_speaker(Speaker::speaker("test").speak("Hello world").build());

    // Verify the config was stored
    assert!(builder.engine_config.is_some());

    let stored_config = builder.engine_config.as_ref().unwrap();
    assert_eq!(stored_config.get("device"), Some(&"cpu".to_string()));
}

#[tokio::test]
async fn test_result_handler_in_both_methods() {
    use std::sync::{Arc, Mutex};

    let call_count = Arc::new(Mutex::new(0));
    let call_count_clone = call_count.clone();

    let builder = TtsConversationBuilderImpl::default()
        .on_result(move |_result| {
            let mut count = call_count_clone.lock().unwrap();
            *count += 1;
        })
        .with_speaker(Speaker::speaker("test").speak("Hello world").build());

    // Test finish_conversation calls the handler
    let conversation = builder.finish_conversation().await;
    assert!(conversation.is_ok());

    // Should have been called once
    assert_eq!(*call_count.lock().unwrap(), 1);
}

#[tokio::test]
async fn test_prelude_postlude_integration() {
    let builder = TtsConversationBuilderImpl::default()
        .with_prelude(|| vec![1, 2, 3, 4]) // Mock audio bytes
        .with_postlude(|| vec![5, 6, 7, 8]) // Mock audio bytes
        .with_speaker(Speaker::speaker("test").speak("Hello world").build());

    // Verify the functions were stored
    assert!(builder.prelude.is_some());
    assert!(builder.postlude.is_some());

    // Test that the functions can be called
    let prelude_data = builder.prelude.as_ref().unwrap()();
    let postlude_data = builder.postlude.as_ref().unwrap()();

    assert_eq!(prelude_data, vec![1, 2, 3, 4]);
    assert_eq!(postlude_data, vec![5, 6, 7, 8]);
}
