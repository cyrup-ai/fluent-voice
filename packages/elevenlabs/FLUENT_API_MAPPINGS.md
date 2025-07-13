# Fluent-Voice API Mappings for ElevenLabs

This document shows the complete mapping from ElevenLabs internal API to the fluent-voice builder pattern.

## Engine Entry Point

```rust
// ONLY way to access ElevenLabs functionality
TtsEngine::elevenlabs() -> TtsEngineBuilder
```

## TtsEngineBuilder Methods

| Fluent API Method | Maps To | Description |
|-------------------|---------|-------------|
| `.api_key(key)` | `ElevenLabsClient::new(api_key)` | Set API key directly |
| `.api_key_from_env()` | `ElevenLabsClient::from_env()` | Load from ELEVENLABS_API_KEY, ELEVEN_API_KEY, or ELEVEN_LABS_API_KEY |
| `.http3_enabled(bool)` | `ClientConfig` with HTTP/3 | Enable HTTP/3 QUIC |
| `.http3_config(Http3Config)` | `ClientConfig` fields | Configure HTTP/3 settings |
| `.build()` | `ElevenLabsClient` creation | Build the engine |

### Http3Config Structure

| Field | Maps To | Default |
|-------|---------|---------|
| `enable_early_data` | `ClientConfig::enable_early_data` | true |
| `max_idle_timeout` | `ClientConfig::max_idle_timeout` | 30s |
| `stream_receive_window` | `ClientConfig::stream_receive_window` | 1MB |
| `conn_receive_window` | `ClientConfig::conn_receive_window` | 10MB |
| `send_window` | `ClientConfig::send_window` | 1MB |

## TtsEngine Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `.tts()` | `TtsBuilder` | Start building a TTS request |
| `.voices()` | `Vec<Voice>` | List available voices (hardcoded for now) |
| `.models()` | `Vec<Model>` | List available models |

## TtsBuilder Methods (Complete Mapping)

| Fluent API Method | Maps To ElevenLabs | Type | Required |
|-------------------|-------------------|------|----------|
| `.text(str)` | `TextToSpeechBody::text` | String | ✅ Yes |
| `.voice(str)` | Voice ID lookup + endpoint | String | No (default: Sarah) |
| `.model(str)` | `TextToSpeechBody::model_id` | Model enum | No (default: eleven_multilingual_v2) |
| `.voice_settings(VoiceSettings)` | `TextToSpeechBody::voice_settings` | VoiceSettings | No |
| `.output_format(AudioFormat)` | `TextToSpeechQuery::output_format` | OutputFormat | No (default: Mp3_44100_128) |
| `.optimize_streaming_latency(u8)` | `TextToSpeechQuery::optimize_streaming_latency` | 0-4 | No |
| `.pronunciation_dictionary(str)` | `TextToSpeechBody::pronunciation_dictionary_locators` | Vec<String> | No |
| `.language(str)` | `TextToSpeechBody::language_code` | ISO 639-1 | No |
| `.enable_logging(bool)` | `TextToSpeechBody::enable_logging` | bool | No (default: false) |
| `.seed(i32)` | `TextToSpeechBody::seed` | i32 | No |
| `.previous_text(str)` | `TextToSpeechBody::previous_text` | String | No |
| `.next_text(str)` | `TextToSpeechBody::next_text` | String | No |
| `.use_pvc_as_ivc(bool)` | `TextToSpeechBody::use_pvc_as_ivc` | bool | No (default: false) |
| `.apply_text_normalization(str)` | `TextToSpeechBody::apply_text_normalization` | String | No |

### Terminal Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `.generate()` | `AudioOutput` | Generate audio synchronously |
| `.stream()` | `AudioStream` | Generate audio as stream |
| `.conversation()` | `TtsConversation` | Start conversation mode |

## Voice Mapping

The `.voice()` method accepts either:
1. Voice name (case-insensitive): "Sarah", "Brian", "Rachel", etc.
2. Voice ID directly: "21m00Tcm4TlvDq8ikWAM"

### Complete Voice Name Mappings

| Name | Maps to Voice ID | Type |
|------|------------------|------|
| "rachel" | DefaultVoice::Rachel | Default |
| "domi" | DefaultVoice::Domi | Default |
| "bella" | DefaultVoice::Bella | Default |
| "antoni" | DefaultVoice::Antoni | Default |
| "elli" | DefaultVoice::Elli | Default |
| "josh" | DefaultVoice::Josh | Default |
| "arnold" | DefaultVoice::Arnold | Default |
| "adam" | DefaultVoice::Adam | Default |
| "sam" | DefaultVoice::Sam | Default |
| "daniel" | DefaultVoice::Daniel | Default |
| "charlotte" | DefaultVoice::Charlotte | Default |
| "alice" | DefaultVoice::Alice | Default |
| "matilda" | DefaultVoice::Matilda | Default |
| "bill" | DefaultVoice::Bill | Default |
| "brian" | DefaultVoice::Brian | Default |
| "callum" | DefaultVoice::Callum | Default |
| "charlie" | DefaultVoice::Charlie | Default |
| "carter" | DefaultVoice::Carter | Default |
| "george" | DefaultVoice::George | Default |
| "daphne" | DefaultVoice::Daphne | Default |
| "ellie" | DefaultVoice::Ellie | Default |
| "elijah" | DefaultVoice::Elijah | Default |
| "chris" | DefaultVoice::Chris | Default |
| "jessica" | DefaultVoice::Jessica | Default |
| "gigi" | DefaultVoice::Gigi | Default |
| "glinda" | DefaultVoice::Glinda | Default |
| "grace" | DefaultVoice::Grace | Default |
| "michael" | DefaultVoice::Michael | Default |
| "jessie" | DefaultVoice::Jessie | Default |
| "harry" | DefaultVoice::Harry | Default |
| "liam" | DefaultVoice::Liam | Default |
| "lily" | DefaultVoice::Lily | Default |
| "clyde" | LegacyVoice::Clyde | Legacy |
| "roger" | DefaultVoice::Roger | Default |
| "river" | DefaultVoice::River | Default |
| "ryan" | DefaultVoice::Ryan | Default |
| "will" | DefaultVoice::Will | Default |
| "dave" | DefaultVoice::Dave | Default |
| "max" | DefaultVoice::Max | Default |
| "jeremy" | DefaultVoice::Jeremy | Default |
| "madison" | DefaultVoice::Madison | Default |
| "joanne" | DefaultVoice::Joanne | Default |
| "maya" | DefaultVoice::Maya | Default |
| "ruby" | DefaultVoice::Ruby | Default |
| "laura" | DefaultVoice::Laura | Default |
| "hazel" | DefaultVoice::Hazel | Default |
| "alexis" | DefaultVoice::Alexis | Default |
| "hillary" | DefaultVoice::Hillary | Default |
| "hailey" | DefaultVoice::Hailey | Default |
| "patrick" | DefaultVoice::Patrick | Default |
| "fin" | DefaultVoice::Fin | Default |
| "freya" | DefaultVoice::Freya | Default |
| "aria" | DefaultVoice::Aria | Default |
| "serena" | DefaultVoice::Serena | Default |
| "nicole" | DefaultVoice::Nicole | Default |
| "sky" | DefaultVoice::Sky | Default |
| "andrea" | DefaultVoice::Andrea | Default |
| "hunter" | DefaultVoice::Hunter | Default |
| "india" | DefaultVoice::India | Default |
| "stella" | DefaultVoice::Stella | Default |
| "dorothy" | DefaultVoice::Dorothy | Default |
| "ethan" | DefaultVoice::Ethan | Default |
| "finn" | DefaultVoice::Finn | Default |
| "lawrence" | DefaultVoice::Lawrence | Default |
| "seraphina" | DefaultVoice::Seraphina | Default |
| "ava" | DefaultVoice::Ava | Default |
| "amelia" | DefaultVoice::Amelia | Default |
| "aiden" | DefaultVoice::Aiden | Default |
| "alexander" | DefaultVoice::Alexander | Default |
| "benjamin" | DefaultVoice::Benjamin | Default |
| "isabella" | DefaultVoice::Isabella | Default |
| "james" | DefaultVoice::James | Default |
| "lucas" | DefaultVoice::Lucas | Default |
| "mia" | DefaultVoice::Mia | Default |
| "noah" | DefaultVoice::Noah | Default |
| "oliver" | DefaultVoice::Oliver | Default |
| "william" | DefaultVoice::William | Default |
| "chris - anime" | DefaultVoice::ChrisAnime | Default |
| "jacob" | DefaultVoice::Jacob | Default |
| "matthew" | DefaultVoice::Matthew | Default |
| "victoria" | DefaultVoice::Victoria | Default |
| "sarah" | DefaultVoice::Sarah | Default |

## Model Mapping

| Fluent API String | Maps to Model Enum |
|-------------------|-------------------|
| "eleven_monolingual_v1" | Model::ElevenMonolingualV1 |
| "eleven_multilingual_v1" | Model::ElevenMultilingualV1 |
| "eleven_multilingual_v2" | Model::ElevenMultilingualV2 |
| "eleven_turbo_v2" | Model::ElevenTurboV2 |
| "eleven_turbo_v2_5" | Model::ElevenTurboV2_5 |

## VoiceSettings Structure

| Field | Maps To | Type | Default |
|-------|---------|------|---------|
| `stability` | `VoiceSettings::stability` | f32 | 0.5 |
| `similarity_boost` | `VoiceSettings::similarity_boost` | f32 | 0.75 |
| `style` | `VoiceSettings::style` | Option<f32> | Some(0.0) |
| `use_speaker_boost` | `VoiceSettings::use_speaker_boost` | Option<bool> | Some(true) |

## AudioFormat Enum

| Fluent API | Maps to OutputFormat |
|------------|---------------------|
| `AudioFormat::Mp3_44100_32` | `OutputFormat::Mp3_44100_32` |
| `AudioFormat::Mp3_44100_64` | `OutputFormat::Mp3_44100_64` |
| `AudioFormat::Mp3_44100_96` | `OutputFormat::Mp3_44100_96` |
| `AudioFormat::Mp3_44100_128` | `OutputFormat::Mp3_44100_128` |
| `AudioFormat::Mp3_44100_192` | `OutputFormat::Mp3_44100_192` |
| `AudioFormat::Pcm_16000` | `OutputFormat::Pcm_16000` |
| `AudioFormat::Pcm_22050` | `OutputFormat::Pcm_22050` |
| `AudioFormat::Pcm_24000` | `OutputFormat::Pcm_24000` |
| `AudioFormat::Pcm_44100` | `OutputFormat::Pcm_44100` |
| `AudioFormat::Ulaw_8000` | `OutputFormat::Ulaw_8000` |

## AudioOutput Methods

| Method | Description |
|--------|-------------|
| `.play()` | Play audio using rodio |
| `.bytes()` | Get raw audio bytes |
| `.save(path)` | Save to file |
| `.format()` | Get audio format |
| `.size()` | Get size in bytes |

## AudioStream Methods

| Method | Description |
|--------|-------------|
| `.play()` | Play streaming audio |
| `.save(path)` | Save stream to file |
| Implements `Stream` | Can be used as futures Stream |

## TtsConversation Methods

| Method | Description |
|--------|-------------|
| `.send_text(str)` | Send text and receive audio |

## Complete Usage Example

```rust
use fluent_voice_elevenlabs::{TtsEngine, VoiceSettings, AudioFormat};

// Build engine
let engine = TtsEngine::elevenlabs()
    .api_key_from_env()?
    .http3_enabled(true)
    .build()?;

// Generate audio with ALL options
let audio = engine
    .tts()
    .text("Hello world")
    .voice("Sarah")
    .model("eleven_multilingual_v2")
    .voice_settings(VoiceSettings {
        stability: 0.5,
        similarity_boost: 0.75,
        style: Some(0.5),
        use_speaker_boost: Some(true),
    })
    .output_format(AudioFormat::Mp3_44100_128)
    .optimize_streaming_latency(2)
    .pronunciation_dictionary("dict_id")
    .language("en")
    .enable_logging(true)
    .seed(42)
    .previous_text("Previous sentence.")
    .next_text("Next sentence.")
    .use_pvc_as_ivc(false)
    .apply_text_normalization("auto")
    .generate()
    .await?;

// Use the audio
audio.play().await?;
```

## Internal API Hidden From Users

The following are NOT exposed:
- `ElevenLabsClient`
- `TextToSpeech`, `TextToSpeechBody`, `TextToSpeechQuery`
- All endpoints in `endpoints/`
- Raw `DefaultVoice`, `LegacyVoice` enums
- `shared` module
- `utils` module (except through AudioOutput methods)
- WebSocket functionality (currently disabled)