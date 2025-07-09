# Speech-to-Text API

Transcribe audio files into text with advanced features like speaker diarization and multilingual support.

## Base Endpoint

```
POST https://api.elevenlabs.io/v1/speech-to-text
```

## Transcribe Audio

Convert audio files to text with support for 99 languages and advanced features.

### Endpoint

```http
POST /v1/speech-to-text
```

### Headers

| Header | Type | Required | Description |
|--------|------|----------|-------------|
| `xi-api-key` | string | ✓ | Your ElevenLabs API key |
| `Content-Type` | string | ✓ | `multipart/form-data` |

### Form Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `audio` | file | ✓ | Audio file to transcribe |
| `model_id` | string | ✗ | Model to use (default: `scribe_v1`) |
| `language_code` | string | ✗ | Language hint for transcription |
| `prompt` | string | ✗ | Context prompt for better accuracy |
| `response_format` | string | ✗ | Output format (`json`, `text`, `srt`, `vtt`) |
| `temperature` | float | ✗ | Randomness control (0.0-1.0) |
| `timestamp_granularities` | array | ✗ | Timestamp detail level |

### Supported Audio Formats

| Format | Extension | Max Size | Max Duration |
|--------|-----------|----------|--------------|
| MP3 | `.mp3` | 1 GB | 4.5 hours |
| MP4 | `.mp4` | 1 GB | 4.5 hours |
| MPEG | `.mpeg` | 1 GB | 4.5 hours |
| MPGA | `.mpga` | 1 GB | 4.5 hours |
| M4A | `.m4a` | 1 GB | 4.5 hours |
| WAV | `.wav` | 1 GB | 4.5 hours |
| WEBM | `.webm` | 1 GB | 4.5 hours |

### Response Formats

#### JSON (default)
```json
{
  "text": "Full transcription text",
  "language": "en",
  "duration": 125.5,
  "segments": [
    {
      "id": 0,
      "seek": 0,
      "start": 0.0,
      "end": 3.5,
      "text": "Hello, welcome to our presentation.",
      "tokens": [1234, 5678, 9012],
      "temperature": 0.0,
      "avg_logprob": -0.45,
      "compression_ratio": 1.2,
      "no_speech_prob": 0.01,
      "speaker": "SPEAKER_00"
    }
  ]
}
```

#### Text
```
Hello, welcome to our presentation. Today we'll be discussing the latest developments in artificial intelligence.
```

#### SRT (SubRip)
```
1
00:00:00,000 --> 00:00:03,500
Hello, welcome to our presentation.

2  
00:00:03,500 --> 00:00:07,200
Today we'll be discussing the latest developments.
```

#### VTT (WebVTT)
```
WEBVTT

00:00:00.000 --> 00:00:03.500
Hello, welcome to our presentation.

00:00:03.500 --> 00:00:07.200
Today we'll be discussing the latest developments.
```

### Example Request

```bash
curl -X POST "https://api.elevenlabs.io/v1/speech-to-text" \
  -H "xi-api-key: YOUR_API_KEY" \
  -F "audio=@meeting.mp3" \
  -F "model_id=scribe_v1" \
  -F "language_code=en" \
  -F "response_format=json" \
  -F "timestamp_granularities[]=word" \
  -F "timestamp_granularities[]=segment"
```

## Models

### Scribe v1

The primary speech-to-text model with the following capabilities:

- **Languages**: 99 languages supported
- **Speaker Diarization**: Up to 32 speakers
- **Timestamps**: Word and segment level
- **Audio Events**: Detects laughter, applause, etc.
- **Accuracy**: Varies by language (see language support section)

## Language Support

ElevenLabs supports transcription in 99 languages with varying accuracy levels:

### Excellent Accuracy (<5% WER)
- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Italian (it)
- Portuguese (pt)
- Russian (ru)
- Japanese (ja)
- Korean (ko)
- Chinese (zh)

### High Accuracy (5-10% WER)
- Dutch (nl)
- Turkish (tr)
- Polish (pl)
- Arabic (ar)
- Hindi (hi)
- Thai (th)
- Vietnamese (vi)
- Hebrew (he)
- Czech (cs)
- Hungarian (hu)

### Good Accuracy (10-15% WER)
- Swedish (sv)
- Norwegian (no)
- Finnish (fi)
- Danish (da)
- Greek (el)
- Bulgarian (bg)
- Romanian (ro)
- Croatian (hr)
- Slovak (sk)
- Lithuanian (lt)

### Moderate Accuracy (15-25% WER)
- Ukrainian (uk)
- Slovenian (sl)
- Estonian (et)
- Latvian (lv)
- Serbian (sr)
- Macedonian (mk)
- Albanian (sq)
- Bosnian (bs)
- Catalan (ca)
- Basque (eu)

*Note: WER = Word Error Rate*

## Advanced Features

### Speaker Diarization

Automatically identify and separate different speakers:

```json
{
  "segments": [
    {
      "speaker": "SPEAKER_00",
      "text": "Good morning everyone."
    },
    {
      "speaker": "SPEAKER_01", 
      "text": "Thank you for joining us today."
    }
  ]
}
```

### Word-Level Timestamps

Get precise timing for individual words:

```json
{
  "segments": [
    {
      "words": [
        {
          "word": "Hello",
          "start": 0.0,
          "end": 0.5
        },
        {
          "word": "world",
          "start": 0.6,
          "end": 1.1
        }
      ]
    }
  ]
}
```

### Audio Event Detection

Detect non-speech audio events:

```json
{
  "segments": [
    {
      "text": "[LAUGHTER]",
      "start": 15.2,
      "end": 17.8
    },
    {
      "text": "[APPLAUSE]", 
      "start": 45.1,
      "end": 48.3
    }
  ]
}
```

## Pricing

Speech-to-text pricing is based on audio duration:

| Tier | Monthly Allowance | Price |
|------|-------------------|-------|
| Free | 30 minutes | $0 |
| Starter | 3 hours | $5/month |
| Creator | 24 hours | $22/month |
| Pro | 100 hours | $99/month |
| Scale | 500 hours | $330/month |
| Business | 2000 hours | $1,320/month |

Additional usage beyond monthly allowance is charged per minute.

## Performance

### Processing Speed

- **Files under 8 minutes**: Sequential processing
- **Files over 8 minutes**: Parallel processing for faster results
- **Concurrency**: Depends on audio duration and subscription tier

### Accuracy Optimization

1. **Use language hints**: Specify `language_code` for better accuracy
2. **Provide context**: Use `prompt` parameter for domain-specific content
3. **Quality audio**: Higher quality input improves transcription accuracy
4. **Appropriate format**: Use uncompressed formats when possible

## Error Handling

### Common Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| 400 | Bad Request | Check file format and parameters |
| 401 | Unauthorized | Verify API key |
| 413 | File too large | Reduce file size or duration |
| 415 | Unsupported format | Use supported audio format |
| 429 | Rate limit exceeded | Reduce request frequency |
| 500 | Internal server error | Retry request |

### Error Response Format

```json
{
  "detail": {
    "status": "error_type",
    "message": "Detailed error description",
    "code": "ERROR_CODE"
  }
}
```

## SDK Examples

### Python

```python
from elevenlabs import ElevenLabs

client = ElevenLabs(api_key="YOUR_API_KEY")

# Basic transcription
with open("audio.mp3", "rb") as audio_file:
    transcript = client.speech_to_text.transcribe(
        audio=audio_file,
        model_id="scribe_v1"
    )

print(transcript.text)

# Advanced transcription with options
with open("meeting.wav", "rb") as audio_file:
    transcript = client.speech_to_text.transcribe(
        audio=audio_file,
        model_id="scribe_v1",
        language_code="en",
        response_format="json",
        timestamp_granularities=["word", "segment"]
    )

# Process segments with speaker information
for segment in transcript.segments:
    print(f"[{segment.start:.1f}s] {segment.speaker}: {segment.text}")
```

### Node.js

```javascript
import { ElevenLabs } from '@elevenlabs/elevenlabs-js';
import fs from 'fs';

const client = new ElevenLabs({
  apiKey: 'YOUR_API_KEY'
});

// Basic transcription
const audioFile = fs.readFileSync('audio.mp3');
const transcript = await client.speechToText.transcribe({
  audio: audioFile,
  model_id: 'scribe_v1'
});

console.log(transcript.text);

// Advanced transcription
const advancedTranscript = await client.speechToText.transcribe({
  audio: audioFile,
  model_id: 'scribe_v1',
  language_code: 'en',
  response_format: 'json',
  timestamp_granularities: ['word', 'segment']
});

// Process segments
advancedTranscript.segments.forEach(segment => {
  console.log(`[${segment.start.toFixed(1)}s] ${segment.speaker}: ${segment.text}`);
});
```

## Best Practices

1. **Optimize audio quality**: Use clear, high-quality audio for best results
2. **Specify language**: Use `language_code` parameter when language is known
3. **Use appropriate format**: Choose the right response format for your use case
4. **Handle large files**: Consider splitting very long audio files
5. **Context prompts**: Use relevant prompts for domain-specific terminology
6. **Monitor usage**: Track transcription minutes to manage costs
7. **Error handling**: Implement robust error handling for file processing
8. **Batch processing**: Process multiple files efficiently within rate limits