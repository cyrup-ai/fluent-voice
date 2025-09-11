# Text-to-Speech API

Convert text into natural-sounding speech using ElevenLabs' advanced AI models.

## Base Endpoint

```
POST https://api.elevenlabs.io/v1/text-to-speech/{voice_id}
```

## Convert Text to Speech

### Endpoint

```http
POST /v1/text-to-speech/{voice_id}
```

### Path Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `voice_id` | string | ✓ | ID of the voice to use for speech synthesis |

### Headers

| Header | Type | Required | Description |
|--------|------|----------|-------------|
| `xi-api-key` | string | ✓ | Your ElevenLabs API key |
| `Content-Type` | string | ✓ | `application/json` |

### Request Body

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `text` | string | ✓ | - | Text to convert to speech (max length varies by model) |
| `model_id` | string | ✗ | `eleven_multilingual_v2` | ID of the model to use |
| `output_format` | string | ✗ | `mp3_44100_128` | Audio output format |
| `voice_settings` | object | ✗ | - | Voice parameter overrides |
| `pronunciation_dictionary_locators` | array | ✗ | - | Up to 3 pronunciation dictionaries |
| `seed` | integer | ✗ | - | Deterministic sampling seed |
| `previous_text` | string | ✗ | - | Previous text for context |
| `next_text` | string | ✗ | - | Next text for improved prosody |
| `previous_request_ids` | array | ✗ | - | Previous request IDs for continuity |
| `next_request_ids` | array | ✗ | - | Next request IDs for continuity |

### Output Formats

| Format | Description | Sample Rate | Bitrate |
|--------|-------------|-------------|---------|
| `mp3_22050_32` | MP3 | 22,050 Hz | 32 kbps |
| `mp3_44100_32` | MP3 | 44,100 Hz | 32 kbps |
| `mp3_44100_64` | MP3 | 44,100 Hz | 64 kbps |
| `mp3_44100_96` | MP3 | 44,100 Hz | 96 kbps |
| `mp3_44100_128` | MP3 (default) | 44,100 Hz | 128 kbps |
| `mp3_44100_192` | MP3 | 44,100 Hz | 192 kbps |
| `pcm_16000` | PCM | 16,000 Hz | - |
| `pcm_22050` | PCM | 22,050 Hz | - |
| `pcm_24000` | PCM | 24,000 Hz | - |
| `pcm_44100` | PCM | 44,100 Hz | - |
| `ulaw_8000` | μ-law | 8,000 Hz | - |

### Voice Settings

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `stability` | float | 0.0-1.0 | Voice consistency (higher = more stable) |
| `similarity_boost` | float | 0.0-1.0 | Voice similarity to original (higher = more similar) |
| `style` | float | 0.0-1.0 | Style exaggeration (higher = more expressive) |
| `use_speaker_boost` | boolean | - | Enhance speaker clarity |

### Example Request

```bash
curl -X POST "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM" \
  -H "xi-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello! This is a sample text that will be converted to speech.",
    "model_id": "eleven_multilingual_v2",
    "output_format": "mp3_44100_128",
    "voice_settings": {
      "stability": 0.5,
      "similarity_boost": 0.8,
      "style": 0.2,
      "use_speaker_boost": true
    }
  }' \
  --output "output.mp3"
```

### Response

The response is a binary audio file in the specified format.

**Success Response:**
- **Status:** 200 OK
- **Content-Type:** `audio/mpeg` (or appropriate MIME type)
- **Body:** Binary audio data

**Headers:**
```http
Content-Type: audio/mpeg
Content-Length: 52341
```

## Models

### Available Models

| Model ID | Description | Languages | Max Characters |
|----------|-------------|-----------|----------------|
| `eleven_multilingual_v2` | Latest multilingual model | 32+ | 2,500 |
| `eleven_multilingual_v1` | Previous multilingual model | 28+ | 2,500 |
| `eleven_monolingual_v1` | English-only model | 1 | 5,000 |
| `eleven_flash_v2` | Ultra-low latency | 32+ | 2,000 |
| `eleven_flash_v2_5` | Improved low latency | 32+ | 2,000 |

### Model Capabilities

- **Multilingual Models**: Support 32+ languages with automatic language detection
- **Flash Models**: Optimized for real-time applications with minimal latency
- **Monolingual Models**: Higher quality for English-only use cases

## Language Support

### Supported Languages

The multilingual models support 32+ languages including:

- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Italian (it)
- Portuguese (pt)
- Polish (pl)
- Turkish (tr)
- Russian (ru)
- Dutch (nl)
- Czech (cs)
- Arabic (ar)
- Chinese (zh)
- Japanese (ja)
- Hungarian (hu)
- Korean (ko)
- Hindi (hi)
- Finnish (fi)
- Croatian (hr)
- Slovak (sk)
- Tamil (ta)
- Ukrainian (uk)
- Swedish (sv)
- Bulgarian (bg)
- Romanian (ro)
- Greek (el)
- Norwegian (no)
- Danish (da)
- Lithuanian (lt)
- Latvian (lv)
- Estonian (et)
- Slovenian (sl)

## Advanced Features

### Pronunciation Dictionaries

Custom pronunciation dictionaries can be applied:

```json
{
  "text": "The CEO of OpenAI gave a great speech.",
  "pronunciation_dictionary_locators": [
    {
      "pronunciation_dictionary_id": "dict_123",
      "version_id": "v1.0"
    }
  ]
}
```

### Deterministic Generation

Use a seed for consistent output:

```json
{
  "text": "This will always sound the same.",
  "seed": 42
}
```

### Context Continuity

For multi-segment speech with natural flow:

```json
{
  "text": "This is the second sentence.",
  "previous_text": "This was the first sentence.",
  "previous_request_ids": ["req_123"]
}
```

## Error Handling

### Common Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| 400 | Bad Request | Check request format and parameters |
| 401 | Unauthorized | Verify API key |
| 404 | Voice not found | Check voice_id exists |
| 422 | Unprocessable Entity | Validate request parameters |
| 429 | Rate limit exceeded | Reduce request frequency |
| 500 | Internal server error | Retry request |

### Error Response Format

```json
{
  "detail": {
    "status": "error_type",
    "message": "Detailed error description"
  }
}
```

## SDK Examples

### Python

```python
from elevenlabs import ElevenLabs

client = ElevenLabs(api_key="YOUR_API_KEY")

# Basic usage
audio = client.text_to_speech.convert(
    voice_id="21m00Tcm4TlvDq8ikWAM",
    text="Hello world!"
)

# Advanced usage
audio = client.text_to_speech.convert(
    voice_id="21m00Tcm4TlvDq8ikWAM",
    text="Hello world!",
    model_id="eleven_multilingual_v2",
    output_format="mp3_44100_128",
    voice_settings={
        "stability": 0.5,
        "similarity_boost": 0.8,
        "style": 0.2,
        "use_speaker_boost": True
    }
)

# Save to file
with open("output.mp3", "wb") as f:
    f.write(audio)
```

### Node.js

```javascript
import { ElevenLabs } from '@elevenlabs/elevenlabs-js';

const client = new ElevenLabs({
  apiKey: 'YOUR_API_KEY'
});

// Basic usage
const audio = await client.textToSpeech.convert({
  voice_id: '21m00Tcm4TlvDq8ikWAM',
  text: 'Hello world!'
});

// Advanced usage
const audioAdvanced = await client.textToSpeech.convert({
  voice_id: '21m00Tcm4TlvDq8ikWAM',
  text: 'Hello world!',
  model_id: 'eleven_multilingual_v2',
  output_format: 'mp3_44100_128',
  voice_settings: {
    stability: 0.5,
    similarity_boost: 0.8,
    style: 0.2,
    use_speaker_boost: true
  }
});
```

## Best Practices

1. **Choose the right model**: Use Flash models for real-time applications, multilingual for multiple languages
2. **Optimize voice settings**: Adjust stability and similarity_boost based on your needs
3. **Use context**: Provide previous_text for better continuity in long content
4. **Handle errors gracefully**: Implement retry logic for transient failures
5. **Cache results**: Store generated audio to avoid redundant API calls
6. **Monitor usage**: Track API usage to stay within rate limits