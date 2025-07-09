# ElevenLabs API Documentation

This directory contains comprehensive documentation for all ElevenLabs API endpoints and capabilities.

## API Overview

ElevenLabs provides a powerful AI audio API with the following core capabilities:

- **Text-to-Speech**: Convert text to natural-sounding speech in 32+ languages
- **Speech-to-Text**: Transcribe audio in 99 languages with speaker diarization
- **Voice Cloning**: Create custom voices from audio samples
- **Speech-to-Speech**: Convert speech while preserving vocal characteristics
- **Audio Isolation**: Separate voices from background audio
- **Sound Effects**: Generate audio effects from text descriptions
- **Voice Changer**: Real-time voice modification
- **Dubbing**: Multi-language audio dubbing

## Base URL

```
https://api.elevenlabs.io
```

## Authentication

All API requests require authentication via the `xi-api-key` header:

```http
xi-api-key: YOUR_API_KEY
```

## Documentation Index

- [Authentication](./authentication.md) - API keys and authentication methods
- [Text-to-Speech](./text-to-speech.md) - Convert text to speech
- [Speech-to-Text](./speech-to-text.md) - Transcribe audio to text
- [Voices](./voices.md) - Voice management and discovery
- [Models](./models.md) - Available AI models and capabilities
- [Speech-to-Speech](./speech-to-speech.md) - Voice conversion
- [Voice Cloning](./voice-cloning.md) - Custom voice creation
- [Audio Isolation](./audio-isolation.md) - Audio separation
- [User Management](./user-management.md) - Account and usage management
- [Error Handling](./error-handling.md) - Error codes and troubleshooting

## Rate Limits

Rate limits vary by subscription tier and endpoint. Check your account dashboard for specific limits.

## SDKs

Official SDKs are available for:

- **Python**: `pip install elevenlabs`
- **Node.js**: `npm install @elevenlabs/elevenlabs-js`

## Support

- [Official Documentation](https://elevenlabs.io/docs)
- [API Reference](https://elevenlabs.io/docs/api-reference)
- [Community Support](https://help.elevenlabs.io)