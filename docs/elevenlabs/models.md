# Models API

Retrieve information about available AI voice models and their capabilities.

## Base Endpoint

```
GET https://api.elevenlabs.io/v1/models
```

## List All Models

Get a list of all available AI voice models.

### Endpoint

```http
GET /v1/models
```

### Headers

| Header | Type | Required | Description |
|--------|------|----------|-------------|
| `xi-api-key` | string | ✓ | Your ElevenLabs API key |

### Example Request

```bash
curl -X GET "https://api.elevenlabs.io/v1/models" \
  -H "xi-api-key: YOUR_API_KEY"
```

### Response Format

```json
[
  {
    "model_id": "eleven_multilingual_v2",
    "name": "Eleven Multilingual v2",
    "can_be_finetuned": true,
    "can_do_text_to_speech": true,
    "can_do_voice_conversion": true,
    "can_use_style": true,
    "can_use_speaker_boost": true,
    "serves_pro_voices": true,
    "token_cost_factor": 1.0,
    "description": "Our cutting edge multilingual speech synthesis model, capable of generating life-like speech in 32 languages. Enhanced version of our multilingual model with improved quality and reduced latency.",
    "requires_alpha_access": false,
    "max_characters_request_free_user": 2500,
    "max_characters_request_subscribed_user": 2500,
    "maximum_text_length_per_request": 2500,
    "language": {
      "language_id": "multi",
      "name": "Multi"
    },
    "concurrency_group": "standard"
  },
  {
    "model_id": "eleven_flash_v2_5", 
    "name": "Eleven Flash v2.5",
    "can_be_finetuned": false,
    "can_do_text_to_speech": true,
    "can_do_voice_conversion": false,
    "can_use_style": true,
    "can_use_speaker_boost": true,
    "serves_pro_voices": false,
    "token_cost_factor": 0.3,
    "description": "Our fastest English language model, optimized for real-time conversation with the lowest latency. Supports style but not voice conversion.",
    "requires_alpha_access": false,
    "max_characters_request_free_user": 2000,
    "max_characters_request_subscribed_user": 2000,
    "maximum_text_length_per_request": 2000,
    "language": {
      "language_id": "en",
      "name": "English"
    },
    "concurrency_group": "turbo"
  }
]
```

## Model Properties

### Core Capabilities

| Property | Type | Description |
|----------|------|-------------|
| `model_id` | string | Unique identifier for the model |
| `name` | string | Human-readable model name |
| `can_be_finetuned` | boolean | Whether the model supports fine-tuning |
| `can_do_text_to_speech` | boolean | Text-to-speech capability |
| `can_do_voice_conversion` | boolean | Voice conversion capability |
| `can_use_style` | boolean | Style parameter support |
| `can_use_speaker_boost` | boolean | Speaker boost feature support |
| `serves_pro_voices` | boolean | Professional voice compatibility |

### Performance & Limits

| Property | Type | Description |
|----------|------|-------------|
| `token_cost_factor` | float | Cost multiplier relative to base model |
| `max_characters_request_free_user` | integer | Character limit for free users |
| `max_characters_request_subscribed_user` | integer | Character limit for paid users |
| `maximum_text_length_per_request` | integer | Maximum text length per request |
| `concurrency_group` | string | Performance tier (`standard` or `turbo`) |

### Additional Properties

| Property | Type | Description |
|----------|------|-------------|
| `description` | string | Detailed model description |
| `requires_alpha_access` | boolean | Whether model requires special access |
| `language` | object | Supported language information |

## Available Models

### Multilingual Models

#### Eleven Multilingual v2
- **Model ID**: `eleven_multilingual_v2`
- **Languages**: 32+ languages
- **Max Characters**: 2,500
- **Features**: Text-to-speech, voice conversion, style, speaker boost
- **Use Case**: General purpose multilingual synthesis
- **Cost Factor**: 1.0x

#### Eleven Multilingual v1
- **Model ID**: `eleven_multilingual_v1`
- **Languages**: 28+ languages
- **Max Characters**: 2,500
- **Features**: Text-to-speech, voice conversion
- **Use Case**: Legacy multilingual synthesis
- **Cost Factor**: 1.0x

### English-Only Models

#### Eleven Monolingual v1
- **Model ID**: `eleven_monolingual_v1`
- **Languages**: English only
- **Max Characters**: 5,000
- **Features**: Text-to-speech, voice conversion, enhanced quality
- **Use Case**: High-quality English synthesis
- **Cost Factor**: 1.0x

### Low-Latency Models

#### Eleven Flash v2.5
- **Model ID**: `eleven_flash_v2_5`
- **Languages**: English only
- **Max Characters**: 2,000
- **Features**: Ultra-low latency, style support
- **Use Case**: Real-time conversation, live streaming
- **Cost Factor**: 0.3x
- **Concurrency**: Turbo tier

#### Eleven Flash v2
- **Model ID**: `eleven_flash_v2`
- **Languages**: 32+ languages
- **Max Characters**: 2,000
- **Features**: Low latency, multilingual
- **Use Case**: Real-time multilingual applications
- **Cost Factor**: 0.5x
- **Concurrency**: Turbo tier

### Turbo Models

#### Eleven Turbo v2.5
- **Model ID**: `eleven_turbo_v2_5`
- **Languages**: English only
- **Max Characters**: 2,000
- **Features**: Balanced speed and quality
- **Use Case**: Fast English synthesis
- **Cost Factor**: 0.5x

#### Eleven Turbo v2
- **Model ID**: `eleven_turbo_v2`
- **Languages**: 32+ languages
- **Max Characters**: 2,000
- **Features**: Fast multilingual synthesis
- **Use Case**: Quick multilingual audio generation
- **Cost Factor**: 0.5x

## Model Selection Guide

### Choose Based on Use Case

| Use Case | Recommended Model | Reason |
|----------|-------------------|---------|
| Real-time conversation | `eleven_flash_v2_5` | Lowest latency |
| Live streaming | `eleven_flash_v2` | Low latency + multilingual |
| Audiobook narration | `eleven_monolingual_v1` | Highest quality English |
| Multilingual content | `eleven_multilingual_v2` | Best multilingual quality |
| Bulk generation | `eleven_turbo_v2` | Good balance of speed/cost |
| Budget projects | `eleven_flash_v2_5` | Lowest cost factor |

### Performance Characteristics

| Model | Latency | Quality | Cost | Languages |
|-------|---------|---------|------|-----------|
| Flash v2.5 | Lowest | Good | Lowest | English |
| Flash v2 | Low | Good | Low | 32+ |
| Turbo v2.5 | Medium | High | Medium | English |
| Turbo v2 | Medium | High | Medium | 32+ |
| Multilingual v2 | High | Highest | High | 32+ |
| Monolingual v1 | High | Highest | High | English |

## Feature Compatibility

### Voice Conversion Support

Models that support speech-to-speech voice conversion:
- `eleven_multilingual_v2`
- `eleven_multilingual_v1`
- `eleven_monolingual_v1`
- `eleven_turbo_v2_5`
- `eleven_turbo_v2`

### Style Parameter Support

Models that support style parameter:
- `eleven_multilingual_v2`
- `eleven_flash_v2_5`
- `eleven_flash_v2`
- `eleven_turbo_v2_5`
- `eleven_turbo_v2`

### Fine-Tuning Support

Models that support fine-tuning:
- `eleven_multilingual_v2`
- `eleven_multilingual_v1`
- `eleven_monolingual_v1`

### Professional Voice Support

Models that work with professional voices:
- `eleven_multilingual_v2`
- `eleven_multilingual_v1`
- `eleven_monolingual_v1`

## Concurrency Groups

### Standard Tier
- Higher quality processing
- Standard request throughput
- Models: Multilingual v2, Multilingual v1, Monolingual v1

### Turbo Tier
- Optimized for speed
- Higher request throughput
- Models: Flash v2.5, Flash v2, Turbo v2.5, Turbo v2

## Language Support

### Multilingual Models (32+ Languages)

Supported languages include:
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

## Cost Considerations

### Token Cost Factors

| Model | Cost Factor | Relative Cost |
|-------|-------------|---------------|
| Flash v2.5 | 0.3x | 30% of base |
| Flash v2 | 0.5x | 50% of base |
| Turbo v2.5 | 0.5x | 50% of base |
| Turbo v2 | 0.5x | 50% of base |
| Multilingual v2 | 1.0x | 100% (base) |
| Multilingual v1 | 1.0x | 100% (base) |
| Monolingual v1 | 1.0x | 100% (base) |

## Error Handling

### Common Error Codes

| Code | Description |
|------|-------------|
| 401 | Unauthorized - Invalid API key |
| 429 | Rate limit exceeded |
| 500 | Internal server error |

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

# Get all models
models = client.models.get_all()

# Find specific model
for model in models:
    if model.model_id == "eleven_multilingual_v2":
        print(f"Model: {model.name}")
        print(f"Languages: {model.language.name}")
        print(f"Max chars: {model.maximum_text_length_per_request}")
        break

# Filter models by capability
tts_models = [m for m in models if m.can_do_text_to_speech]
conversion_models = [m for m in models if m.can_do_voice_conversion]
```

### Node.js

```javascript
import { ElevenLabs } from '@elevenlabs/elevenlabs-js';

const client = new ElevenLabs({
  apiKey: 'YOUR_API_KEY'
});

// Get all models
const models = await client.models.getAll();

// Find specific model
const multilingualModel = models.find(
  m => m.model_id === 'eleven_multilingual_v2'
);

console.log(`Model: ${multilingualModel.name}`);
console.log(`Languages: ${multilingualModel.language.name}`);

// Filter models by capability
const ttsModels = models.filter(m => m.can_do_text_to_speech);
const conversionModels = models.filter(m => m.can_do_voice_conversion);
```

## Best Practices

1. **Match model to use case**: Choose Flash for real-time, Multilingual for quality
2. **Consider cost factors**: Flash models are significantly cheaper
3. **Check feature compatibility**: Verify model supports required features
4. **Monitor character limits**: Different models have different limits
5. **Test latency requirements**: Flash models for interactive applications
6. **Language considerations**: Use Monolingual for English-only higher quality