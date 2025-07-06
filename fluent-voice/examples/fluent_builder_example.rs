use fluent_voice::prelude::*;
use futures_util::StreamExt;
use std::error::Error;

/// Example showing proper usage of the fluent voice API with elegant builders
#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Example of Fluent Builder API Usage ===");
    println!("\nThis example demonstrates the elegant fluent builder pattern.");
    println!("In a real implementation, you would use a concrete engine.");

    // See examples/api_usage.rs for complete implementations

    Ok(())
}

// Example of how to use the TTS builders with elegant flow
#[allow(dead_code)]
async fn tts_example() -> Result<(), VoiceError> {
    let mut audio_stream = FluentVoice::tts()
        .with_speaker(
            Speaker::speaker("Alice")
                .voice_id(VoiceId::new("voice-uuid"))
                .with_speed_modifier(VocalSpeedMod(0.9))
                .speak("Hello, world!")
                .build(),
        )
        .with_speaker(
            Speaker::speaker("Bob")
                .with_speed_modifier(VocalSpeedMod(1.1))
                .speak("Hi Alice! How are you today?")
                .build(),
        )
        .with_speaker(
            Speaker::speaker("Alice")
                .speak("I'm doing great, thanks for asking!")
                .build(),
        )
        .synthesize(|result| match result {
            Ok(conversation) => conversation.into_stream(), // Returns audio stream
            Err(e) => Err(e),
        })
        .await?; // Single await point

    // Process audio samples
    while let Some(sample) = audio_stream.next().await {
        // Play sample or save to file
        println!("Audio sample: {}", sample);
    }

    Ok(())
}

// Example of how to use the STT builders for microphone input
#[allow(dead_code)]
async fn stt_microphone_example() -> Result<(), VoiceError> {
    let mut transcript_stream = FluentVoice::stt()
        .with_microphone("default")
        .vad_mode(VadMode::Accurate)
        .language_hint(Language::EnglishUs)
        .diarization(Diarization::On)
        .word_timestamps(WordTimestamps::On)
        .punctuation(Punctuation::On)
        .listen(|result| match result {
            Ok(conversation) => conversation.into_stream(), // Returns transcript stream
            Err(e) => Err(e),
        })
        .await?;

    // Process transcript segments
    while let Some(segment) = transcript_stream.next().await {
        println!("Transcript: {}", segment.text);

        if let Some(speaker) = segment.speaker {
            println!("Speaker: {}", speaker);
        }

        if let Some(timestamp) = segment.timestamp {
            println!("Time: {}s", timestamp.start_sec);
        }
    }

    Ok(())
}

// Example of how to use the STT builders for file input
#[allow(dead_code)]
async fn stt_file_example() -> Result<(), VoiceError> {
    let transcript = FluentVoice::stt()
        .transcribe("meeting.wav")
        .with_progress("Processing meeting.wav: {percent}%")
        .language_hint(Language::EnglishUs)
        .diarization(Diarization::On)
        .punctuation(Punctuation::On)
        .emit(|result| match result {
            Ok(transcript) => transcript, // Returns transcript
            Err(e) => Err(e),
        })
        .await?;

    // Process the transcript
    let mut transcript_stream = transcript.stream;

    while let Some(segment) = transcript_stream.next().await {
        println!("Transcript: {}", segment.text);

        if let Some(speaker) = segment.speaker {
            println!("Speaker: {}", speaker);
        }
    }

    Ok(())
}

// Example of engine-specific implementation with the same clean API
#[allow(dead_code)]
mod engine_implementation {
    use super::*;

    // Custom implementation for a specific TTS engine
    async fn elevenlabs_example() -> Result<(), VoiceError> {
        let mut audio_stream = ElevenLabsEngine::conversation()
            .with_speaker(
                Speaker::speaker("Alice")
                    .voice_id(VoiceId::new("eleven-labs-rachel"))
                    .with_speed_modifier(VocalSpeedMod(0.9))
                    .speak("Hello, world!")
                    .build(),
            )
            .with_speaker(
                Speaker::speaker("Bob")
                    .voice_id(VoiceId::new("eleven-labs-josh"))
                    .with_speed_modifier(VocalSpeedMod(1.1))
                    .speak("Hi Alice! How are you today?")
                    .build(),
            )
            .api_key(std::env::var("ELEVENLABS_API_KEY").unwrap_or_default())
            .model("eleven_multilingual_v2")
            .with_stability(0.7)
            .with_similarity_boost(0.3)
            .synthesize(|result| match result {
                Ok(conversation) => conversation.into_stream(),
                Err(e) => Err(e),
            })
            .await?;

        // Process audio samples
        while let Some(sample) = audio_stream.next().await {
            // Play sample or save to file
            println!("Audio sample: {}", sample);
        }

        Ok(())
    }

    // Mock for the example - not part of the actual API
    struct ElevenLabsEngine;

    impl ElevenLabsEngine {
        fn conversation() -> ElevenLabsConversationBuilder {
            ElevenLabsConversationBuilder::default()
        }
    }

    // Mock builder for the example - not part of the actual API
    #[derive(Default)]
    struct ElevenLabsConversationBuilder {
        speakers: Vec<Box<dyn Speaker>>,
        api_key: Option<String>,
        model: Option<String>,
        stability: Option<f32>,
        similarity_boost: Option<f32>,
    }

    impl ElevenLabsConversationBuilder {
        fn with_speaker<S: Speaker + 'static>(mut self, speaker: S) -> Self {
            self.speakers.push(Box::new(speaker));
            self
        }

        fn api_key(mut self, key: String) -> Self {
            self.api_key = Some(key);
            self
        }

        fn model(mut self, model: &str) -> Self {
            self.model = Some(model.to_string());
            self
        }

        fn with_stability(mut self, stability: f32) -> Self {
            self.stability = Some(stability);
            self
        }

        fn with_similarity_boost(mut self, boost: f32) -> Self {
            self.similarity_boost = Some(boost);
            self
        }

        fn synthesize<F, R>(self, matcher: F) -> impl std::future::Future<Output = R>
        where
            F: FnOnce(Result<AudioStream, VoiceError>) -> R,
        {
            async move {
                // In a real implementation, this would connect to the ElevenLabs API
                let stream = AudioStream {
                    samples: vec![0; 100],
                };
                matcher(Ok(stream))
            }
        }
    }

    // Mock stream for the example - not part of the actual API
    struct AudioStream {
        samples: Vec<i16>,
    }

    impl AudioStream {
        async fn next(&mut self) -> Option<i16> {
            self.samples.pop()
        }
    }
}
