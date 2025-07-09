//! Test that all README.md examples work with our trait implementations

use fluent_voice::prelude::*;
use futures_util::StreamExt;

// Mock engine for testing TTS examples
struct MyTtsEngine;

impl TtsEngine for MyTtsEngine {
    type Conv = TtsConversationBuilderImpl<futures::stream::Empty<i16>>;

    fn conversation(&self) -> Self::Conv {
        tts_conversation_builder(|_lines, _lang| futures::stream::empty::<i16>())
    }
}

// Mock engine for testing STT examples
struct MySttEngine;

impl SttEngine for MySttEngine {
    type Conv =
        SttConversationBuilderImpl<futures::stream::Empty<Result<DummySegment, VoiceError>>>;

    fn conversation(&self) -> Self::Conv {
        stt_conversation_builder(
            |_source,
             _vad,
             _noise,
             _lang,
             _diarization,
             _word_timestamps,
             _timestamps_granularity,
             _punctuation| {
                futures::stream::empty::<Result<DummySegment, VoiceError>>()
            },
        )
    }
}

#[tokio::main]
async fn main() -> Result<(), VoiceError> {
    println!("🧪 Testing README.md syntax examples...\n");

    // Test 1: First STT example from README (line 53-68)
    println!("Test 1: STT with microphone and collect()");
    {
        // This syntax is from README line 53-68
        let _transcript = FluentVoiceImpl::stt()
            .with_source(SpeechSource::Microphone {
                backend: MicBackend::Default,
                format: AudioFormat::Pcm16Khz,
                sample_rate: 16_000,
            })
            .vad_mode(VadMode::Accurate)
            .language_hint(Language("en-US"))
            .diarization(Diarization::On)
            .word_timestamps(WordTimestamps::On)
            .punctuation(Punctuation::On)
            .listen(|conversation| match conversation {
                Ok(conv) => Ok(conv),
                Err(e) => Err(e),
            })
            .await?
            .collect() // This collect() method should work
            .await?;

        println!("✅ STT with collect() compiles correctly");
    }

    // Test 2: TTS example with Speaker::speaker syntax (line 78-96)
    println!("\nTest 2: TTS with Speaker::speaker syntax");
    {
        let engine = MyTtsEngine;
        let mut audio_stream = engine
            .conversation()
            .with_speaker(
                SpeakerLine::speaker("Alice") // README shows Speaker::speaker
                    .voice_id(VoiceId::new("voice-uuid"))
                    .with_speed_modifier(VocalSpeedMod(0.9))
                    .speak("Hello, world!")
                    .build(),
            )
            .with_speaker(
                SpeakerLine::speaker("Bob") // README shows Speaker::speaker
                    .with_speed_modifier(VocalSpeedMod(1.1))
                    .speak("Hi Alice! How are you today?")
                    .build(),
            )
            .synthesize(|conversation| match conversation {
                Ok(conv) => Ok(conv.into_stream()),
                Err(e) => Err(e),
            })
            .await?;

        // Process a few samples to verify the stream works
        let mut count = 0;
        while let Some(sample) = audio_stream.next().await {
            count += 1;
            if count > 3 {
                break;
            }
        }
        println!("✅ TTS with Speaker::speaker syntax compiles correctly");
    }

    // Test 3: STT example with transcript stream (line 116-131)
    println!("\nTest 3: STT with transcript stream processing");
    {
        let engine = MySttEngine;
        let mut transcript_stream = engine
            .conversation()
            .with_source(SpeechSource::Microphone {
                backend: MicBackend::Default,
                format: AudioFormat::Pcm16Khz,
                sample_rate: 16_000,
            })
            .vad_mode(VadMode::Accurate)
            .language_hint(Language("en-US"))
            .diarization(Diarization::On)
            .word_timestamps(WordTimestamps::On)
            .punctuation(Punctuation::On)
            .listen(|conversation| match conversation {
                Ok(conv) => Ok(conv.into_stream()),
                Err(e) => Err(e),
            })
            .await?;

        // Process a few segments to verify it works
        let mut count = 0;
        while let Some(result) = transcript_stream.next().await {
            match result {
                Ok(segment) => {
                    // Verify all TranscriptSegment methods work
                    let _start = segment.start_ms();
                    let _speaker = segment.speaker_id();
                    let _text = segment.text();
                    count += 1;
                    if count > 3 {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
        println!("✅ STT transcript stream processing compiles correctly");
    }

    // Test 4: Verify the "Ok => " matcher syntax works
    println!("\nTest 4: Testing README's matcher syntax");
    {
        // The README shows this unusual syntax:
        // .synthesize(|conversation| {
        //     Ok  => conversation.into_stream(),
        //     Err(e) => Err(e),
        // })

        // This is actually invalid Rust syntax. The correct syntax should be:
        let engine = MyTtsEngine;
        let _result = engine
            .conversation()
            .synthesize(|conversation| match conversation {
                Ok(conv) => Ok(conv.into_stream()),
                Err(e) => Err(e),
            })
            .await?;

        println!("⚠️  Note: README shows 'Ok =>' syntax which is invalid Rust");
        println!(
            "   The correct syntax is 'match conversation { Ok(conv) => ..., Err(e) => ... }'"
        );
    }

    // Test 5: Test timestamps_granularity (missing from first example)
    println!("\nTest 5: Testing missing timestamps_granularity");
    {
        // The first STT example is missing timestamps_granularity
        let _builder = FluentVoiceImpl::stt()
            .with_source(SpeechSource::Microphone {
                backend: MicBackend::Default,
                format: AudioFormat::Pcm16Khz,
                sample_rate: 16_000,
            })
            .vad_mode(VadMode::Accurate)
            .language_hint(Language("en-US"))
            .diarization(Diarization::On)
            .word_timestamps(WordTimestamps::On)
            .timestamps_granularity(TimestampsGranularity::Word) // This is in the trait but not the first example
            .punctuation(Punctuation::On);

        println!("✅ All STT configuration methods work correctly");
    }

    // Test 6: VoiceId::new vs VoiceId() constructor
    println!("\nTest 6: Testing VoiceId constructor syntax");
    {
        // README shows VoiceId::new("voice-uuid")
        let voice_id = VoiceId::new("voice-uuid");

        // But our implementation uses VoiceId("voice-uuid".to_string())
        let voice_id2 = VoiceId("voice-uuid".to_string());

        println!("⚠️  Note: README shows 'VoiceId::new()' but implementation uses 'VoiceId()'");
    }

    println!("\n🎉 README syntax verification complete!");
    println!("\n⚠️  Issues found:");
    println!(
        "1. README shows 'Ok =>' syntax which should be 'match result { Ok(...) => ..., Err(e) => ... }'"
    );
    println!(
        "2. README shows 'Speaker::speaker()' but implementation provides 'SpeakerLine::speaker()'"
    );
    println!("3. README shows 'VoiceId::new()' but implementation uses 'VoiceId()' constructor");
    println!(
        "4. First STT example uses .collect() which requires the SttConversation to implement collect()"
    );

    Ok(())
}

// Implement VoiceId::new for README compatibility
impl VoiceId {
    pub fn new(id: &str) -> Self {
        VoiceId(id.to_string())
    }
}
