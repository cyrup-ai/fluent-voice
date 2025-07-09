# Authentication

ElevenLabs API uses API key authentication for all requests.

## API Key Authentication

All API requests must include your API key in the `xi-api-key` header:

```http
xi-api-key: YOUR_API_KEY
```

## Getting Your API Key

1. Sign up for an ElevenLabs account at [elevenlabs.io](https://elevenlabs.io)
2. Navigate to your Profile Settings
3. Find your API key in the account section
4. Copy the key for use in your applications

## Security Best Practices

- **Never expose your API key in client-side code**
- **Store API keys as environment variables**
- **Use different keys for development and production**
- **Regenerate keys if compromised**

## Environment Variables

Set your API key as an environment variable:

```bash
export ELEVENLABS_API_KEY="your_api_key_here"
```

## SDK Authentication

### Python SDK

```python
from elevenlabs import ElevenLabs

# Method 1: Pass API key directly
client = ElevenLabs(api_key="your_api_key_here")

# Method 2: Use environment variable ELEVENLABS_API_KEY
client = ElevenLabs()
```

### Node.js SDK

```javascript
import { ElevenLabs } from '@elevenlabs/elevenlabs-js';

// Method 1: Pass API key directly
const client = new ElevenLabs({
  apiKey: 'your_api_key_here'
});

// Method 2: Use environment variable ELEVENLABS_API_KEY
const client = new ElevenLabs();
```

## HTTP Request Examples

### cURL

```bash
curl -X GET "https://api.elevenlabs.io/v1/voices" \
  -H "xi-api-key: YOUR_API_KEY"
```

### HTTP Headers

```http
GET /v1/voices HTTP/1.1
Host: api.elevenlabs.io
xi-api-key: YOUR_API_KEY
Content-Type: application/json
```

## Error Responses

### 401 Unauthorized

Returned when API key is missing or invalid:

```json
{
  "detail": {
    "status": "unauthorized",
    "message": "Invalid or missing API key"
  }
}
```

### 403 Forbidden

Returned when API key doesn't have required permissions:

```json
{
  "detail": {
    "status": "forbidden", 
    "message": "Insufficient permissions for this operation"
  }
}
```

## Rate Limiting

Rate limits are enforced per API key and vary by subscription tier:

- **Free Tier**: Limited requests per month
- **Starter**: Higher monthly limits
- **Creator**: Even higher limits
- **Pro**: Maximum limits with priority processing

Rate limit headers are included in responses:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640995200
```

## Subscription Tiers

Different API capabilities are available based on your subscription:

| Feature | Free | Starter | Creator | Pro |
|---------|------|---------|---------|-----|
| Text-to-Speech | ✓ | ✓ | ✓ | ✓ |
| Voice Cloning | Limited | ✓ | ✓ | ✓ |
| Commercial Use | ✗ | ✓ | ✓ | ✓ |
| API Access | Limited | ✓ | ✓ | ✓ |
| Priority Support | ✗ | ✗ | ✓ | ✓ |