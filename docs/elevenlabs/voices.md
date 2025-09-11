# Voices API

Manage and discover voices available for text-to-speech synthesis.

## Base Endpoints

```
GET https://api.elevenlabs.io/v1/voices
GET https://api.elevenlabs.io/v1/voices/{voice_id}
GET https://api.elevenlabs.io/v2/voices
```

## List All Voices

Get a list of all available voices with optional filtering and pagination.

### Endpoint (v2)

```http
GET /v2/voices
```

### Query Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `next_page_token` | string | ✗ | Token for pagination |
| `page_size` | integer | ✗ | Number of voices per page (default: 10, max: 100) |
| `search` | string | ✗ | Filter by name, description, or labels |
| `sort` | string | ✗ | Sort by `created_at_unix` or `name` |
| `voice_type` | string | ✗ | Filter by voice type |
| `category` | string | ✗ | Filter by voice category |

### Voice Types

| Type | Description |
|------|-------------|
| `personal` | Your custom voices |
| `community` | Community-created voices |
| `default` | ElevenLabs default voices |
| `professional` | Professional voice actors |

### Voice Categories

| Category | Description |
|----------|-------------|
| `premade` | Pre-made voices |
| `cloned` | Cloned voices |
| `generated` | AI-generated voices |

### Example Request

```bash
curl -X GET "https://api.elevenlabs.io/v2/voices?page_size=20&search=female&sort=name" \
  -H "xi-api-key: YOUR_API_KEY"
```

### Response Format

```json
{
  "voices": [
    {
      "voice_id": "21m00Tcm4TlvDq8ikWAM",
      "name": "Rachel",
      "samples": [
        {
          "sample_id": "sample_123",
          "file_name": "rachel_sample.mp3",
          "mime_type": "audio/mpeg",
          "size_bytes": 123456,
          "hash": "abc123def456"
        }
      ],
      "category": "premade",
      "fine_tuning": {
        "model_id": "eleven_multilingual_v2",
        "is_allowed_to_fine_tune": true,
        "finetuning_state": "fine_tuned",
        "verification": {
          "requires_verification": false,
          "is_verified": true,
          "verification_failures": [],
          "verification_attempts_count": 0
        },
        "manual_verification": {
          "extra_text": "",
          "request_time_unix": 1640995200,
          "files": []
        },
        "manual_verification_requested": false
      },
      "labels": {
        "accent": "american",
        "description": "calm",
        "age": "young",
        "gender": "female",
        "use case": "narration"
      },
      "description": "A calm and soothing voice perfect for narration.",
      "preview_url": "https://storage.googleapis.com/eleven-public-prod/premade/voices/21m00Tcm4TlvDq8ikWAM/df6788f9-5c96-470d-8312-aab3b3d8f50a.mp3",
      "available_for_tiers": ["free", "starter", "creator", "pro"],
      "settings": {
        "stability": 0.71,
        "similarity_boost": 0.5,
        "style": 0.0,
        "use_speaker_boost": true
      },
      "sharing": {
        "status": "enabled",
        "history_item_sample_id": "sample_456",
        "original_voice_id": "original_voice_123",
        "public_owner_id": "public_owner_456",
        "liked_by_count": 42,
        "cloned_by_count": 15,
        "whitelisted_emails": []
      },
      "high_quality_base_model_ids": ["eleven_multilingual_v2"],
      "safety_control": null,
      "voice_verification": {
        "requires_verification": false,
        "is_verified": true,
        "verification_failures": [],
        "verification_attempts_count": 0
      },
      "permission_on_resource": null
    }
  ],
  "has_more": true,
  "total_count": 150,
  "next_page_token": "next_page_token_456"
}
```

## Get Specific Voice

Retrieve detailed information about a specific voice.

### Endpoint

```http
GET /v1/voices/{voice_id}
```

### Path Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `voice_id` | string | ✓ | Unique identifier for the voice |

### Example Request

```bash
curl -X GET "https://api.elevenlabs.io/v1/voices/21m00Tcm4TlvDq8ikWAM" \
  -H "xi-api-key: YOUR_API_KEY"
```

### Response Format

```json
{
  "voice_id": "21m00Tcm4TlvDq8ikWAM",
  "name": "Rachel",
  "samples": [
    {
      "sample_id": "sample_123",
      "file_name": "rachel_sample.mp3",
      "mime_type": "audio/mpeg",
      "size_bytes": 123456,
      "hash": "abc123def456"
    }
  ],
  "category": "premade",
  "fine_tuning": {
    "model_id": "eleven_multilingual_v2",
    "is_allowed_to_fine_tune": true,
    "finetuning_state": "fine_tuned",
    "verification": {
      "requires_verification": false,
      "is_verified": true,
      "verification_failures": [],
      "verification_attempts_count": 0
    }
  },
  "labels": {
    "accent": "american",
    "description": "calm",
    "age": "young", 
    "gender": "female",
    "use case": "narration"
  },
  "description": "A calm and soothing voice perfect for narration.",
  "preview_url": "https://storage.googleapis.com/eleven-public-prod/premade/voices/21m00Tcm4TlvDq8ikWAM/df6788f9-5c96-470d-8312-aab3b3d8f50a.mp3",
  "available_for_tiers": ["free", "starter", "creator", "pro"],
  "settings": {
    "stability": 0.71,
    "similarity_boost": 0.5,
    "style": 0.0,
    "use_speaker_boost": true
  },
  "sharing": {
    "status": "enabled",
    "history_item_sample_id": "sample_456",
    "original_voice_id": "original_voice_123",
    "public_owner_id": "public_owner_456",
    "liked_by_count": 42,
    "cloned_by_count": 15,
    "whitelisted_emails": []
  },
  "high_quality_base_model_ids": ["eleven_multilingual_v2"],
  "safety_control": null,
  "voice_verification": {
    "requires_verification": false,
    "is_verified": true,
    "verification_failures": [],
    "verification_attempts_count": 0
  }
}
```

## List Voices (v1)

Legacy endpoint for listing voices.

### Endpoint

```http
GET /v1/voices
```

### Response Format

```json
{
  "voices": [
    {
      "voice_id": "21m00Tcm4TlvDq8ikWAM",
      "name": "Rachel",
      "samples": null,
      "category": "premade",
      "fine_tuning": {
        "model_id": null,
        "is_allowed_to_fine_tune": false,
        "finetuning_state": "not_started",
        "verification": null,
        "manual_verification": null,
        "manual_verification_requested": false
      },
      "labels": {
        "accent": "american",
        "description": "calm",
        "age": "young",
        "gender": "female",
        "use case": "narration"
      },
      "description": null,
      "preview_url": "https://storage.googleapis.com/eleven-public-prod/premade/voices/21m00Tcm4TlvDq8ikWAM/df6788f9-5c96-470d-8312-aab3b3d8f50a.mp3",
      "available_for_tiers": [],
      "settings": null,
      "sharing": null,
      "high_quality_base_model_ids": [],
      "safety_control": null,
      "voice_verification": null,
      "permission_on_resource": null
    }
  ]
}
```

## Voice Settings

Default voice settings can be customized for each voice.

### Settings Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `stability` | float | 0.0-1.0 | 0.71 | Voice consistency |
| `similarity_boost` | float | 0.0-1.0 | 0.5 | Voice similarity to original |
| `style` | float | 0.0-1.0 | 0.0 | Style exaggeration |
| `use_speaker_boost` | boolean | - | true | Enhance speaker clarity |

### Stability
- **Low (0.0-0.3)**: More variable, creative output
- **Medium (0.4-0.7)**: Balanced stability and variability  
- **High (0.8-1.0)**: Very consistent, less variation

### Similarity Boost
- **Low (0.0-0.3)**: More creative interpretation
- **Medium (0.4-0.7)**: Balanced similarity
- **High (0.8-1.0)**: Very close to original voice

## Default Voices

ElevenLabs provides several high-quality default voices:

### English Voices

| Voice ID | Name | Gender | Accent | Description |
|----------|------|--------|--------|-------------|
| `21m00Tcm4TlvDq8ikWAM` | Rachel | Female | American | Calm, young narrator |
| `AZnzlk1XvdvUeBnXmlld` | Domi | Female | American | Strong, confident |
| `EXAVITQu4vr4xnSDxMaL` | Bella | Female | American | Soft, pleasant |
| `ErXwobaYiN019PkySvjV` | Antoni | Male | American | Well-rounded, versatile |
| `MF3mGyEYCl7XYWbV9V6O` | Elli | Female | American | Emotional, young |
| `TxGEqnHWrfWFTfGW9XjX` | Josh | Male | American | Deep, serious |
| `VR6AewLTigWG4xSOukaG` | Arnold | Male | American | Crisp, authoritative |
| `pNInz6obpgDQGcFmaJgB` | Adam | Male | American | Deep, natural |
| `yoZ06aMxZJJ28mfd3POQ` | Sam | Male | American | Raspy, masculine |

### Multilingual Voices

| Voice ID | Name | Languages | Description |
|----------|------|-----------|-------------|
| `Xb7hH8MSUJpSbSDYk0k2` | Alice | 28+ | Confident, news anchor |
| `IKne3meq5aSn9XLyUdCD` | Charlie | 28+ | Casual, conversational |
| `onwK4e9ZLuTAKqWW03F9` | George | 28+ | Warm, friendly |

## Voice Labels

Voices are categorized with descriptive labels:

### Accent Labels
- `american` - American English
- `british` - British English  
- `australian` - Australian English
- `irish` - Irish English

### Age Labels
- `young` - Youthful voice
- `middle aged` - Mature voice
- `old` - Elderly voice

### Gender Labels
- `male` - Male voice
- `female` - Female voice

### Use Case Labels
- `narration` - Story telling, audio books
- `news` - News reading, announcements
- `conversational` - Dialogue, chat bots
- `characters` - Character voices, gaming

### Description Labels
- `calm` - Peaceful, soothing
- `energetic` - High energy, excited  
- `authoritative` - Commanding, serious
- `friendly` - Warm, approachable
- `dramatic` - Theatrical, expressive

## Error Handling

### Common Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid parameters |
| 401 | Unauthorized - Invalid API key |
| 404 | Voice not found |
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

# List all voices
voices = client.voices.get_all()

# Get specific voice
voice = client.voices.get(voice_id="21m00Tcm4TlvDq8ikWAM")

# Search voices
voices = client.voices.get_all(
    search="female",
    category="premade",
    page_size=20
)
```

### Node.js

```javascript
import { ElevenLabs } from '@elevenlabs/elevenlabs-js';

const client = new ElevenLabs({
  apiKey: 'YOUR_API_KEY'
});

// List all voices
const voices = await client.voices.getAll();

// Get specific voice  
const voice = await client.voices.get({
  voice_id: '21m00Tcm4TlvDq8ikWAM'
});

// Search voices
const searchResults = await client.voices.getAll({
  search: 'female',
  category: 'premade', 
  page_size: 20
});
```

## Best Practices

1. **Cache voice data**: Voice metadata changes infrequently
2. **Use appropriate filters**: Narrow down results with search and category filters
3. **Preview voices**: Listen to preview URLs before selection
4. **Test voice settings**: Experiment with stability and similarity_boost
5. **Consider use case**: Match voice characteristics to your application
6. **Monitor voice availability**: Check available_for_tiers for subscription compatibility