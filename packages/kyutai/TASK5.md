# TASK5: Wire Streaming TTS Functionality

## Status: INTEGRATION - Connect Real-time Streaming to Fluent-Voice

### Objective
Wire the existing `tts_streaming.rs` real-time streaming implementation into the fluent-voice API for low-latency TTS.

### THE OTHER END OF THE WIRE: Streaming Extension to TtsConversationBuilder

**Extension Method**: Add streaming support to the existing `TtsConversationBuilder` implementation

```rust
impl KyutaiTtsConversationBuilder {
    // Add streaming method to existing builder
    pub fn streaming(mut self) -> Self {
        self.streaming_enabled = true;
        self
    }
    
    // Alternative terminal method for streaming
    pub async fn stream(self) -> Result<impl Stream<Item = AudioChunk>, VoiceError> {
        // Use StreamingModel instead of regular Model
        // Return audio chunk stream
    }
}
```

**Stream Trait**: Implement standard Rust `Stream` trait for audio chunks

```rust
use futures_core::Stream;
use std::pin::Pin;
use std::task::{Context, Poll};

pub struct KyutaiAudioStream {
    streaming_model: StreamingModel,
    // ... state fields
}

impl Stream for KyutaiAudioStream {
    type Item = AudioChunk;
    
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Use existing StreamingModel.step() method
    }
}
```

**Integration Point**: The streaming functionality extends the existing `TtsConversationBuilder` from TASK2, providing an alternative to the standard `.synthesize()` method for real-time use cases.

### Existing Working Implementation
**File: `src/tts_streaming.rs`**
- Complete `StreamingModel` wrapper (lines 12-184)
- Real-time `step()` method for incremental generation (lines 42-98)
- `flush()` method for completion (lines 100-109)
- `StreamingModule` trait implementation for standard interface

### Integration Points

#### 1. Streaming Model Creation
**Use existing**: `StreamingModel::new()`
```rust
let streaming_model = StreamingModel::new(Arc::new(Mutex::new(tts_model)));
```

#### 2. Incremental Generation
**Use existing**: `step()` method
```rust
let audio_chunk = streaming_model.step(
    Some(text_token),
    &conditions_map,
)?;
```

#### 3. Stream Completion
**Use existing**: `flush()` method
```rust
let final_audio = streaming_model.flush()?;
```

### Implementation Strategy

#### 1. Streaming Builder Extension
Add streaming support to `KyutaiTtsConversationBuilder`:
```rust
impl KyutaiTtsConversationBuilder {
    pub fn streaming(mut self) -> Self {
        self.streaming_enabled = true;
        self
    }
    
    pub async fn stream(self) -> Result<AudioChunkStream, MoshiError> {
        // Use StreamingModel instead of regular Model
        let tts_model = tts::Model::load(/*...*/)?;
        let streaming_model = StreamingModel::new(Arc::new(Mutex::new(tts_model)));
        
        // Return streaming interface
        Ok(AudioChunkStream {
            streaming_model,
            text_queue: VecDeque::new(),
            conditions: self.build_conditions()?,
        })
    }
}
```

#### 2. Audio Chunk Stream
Create streaming interface that wraps existing functionality:
```rust
pub struct AudioChunkStream {
    streaming_model: StreamingModel,
    text_queue: VecDeque<String>,
    conditions: HashMap<String, Condition>,
}

impl AudioChunkStream {
    pub async fn add_text(&mut self, text: &str) -> Result<(), MoshiError> {
        self.text_queue.push_back(text.to_string());
        Ok(())
    }
    
    pub async fn next_chunk(&mut self) -> Result<Option<Vec<f32>>, MoshiError> {
        if let Some(text) = self.text_queue.pop_front() {
            // Use existing step() method
            let chunk = self.streaming_model.step(
                Some(&text),
                &self.conditions,
            )?;
            Ok(Some(chunk))
        } else {
            Ok(None)
        }
    }
    
    pub async fn finish(&mut self) -> Result<Vec<f32>, MoshiError> {
        // Use existing flush() method
        self.streaming_model.flush()
    }
}
```

#### 3. StreamingModule Integration
Leverage existing `StreamingModule` trait:
```rust
impl StreamingModule for AudioChunkStream {
    fn forward_streaming(&mut self, input: &Tensor) -> Result<Tensor> {
        self.streaming_model.forward_streaming(input)
    }
    
    fn reset_streaming(&mut self) {
        self.streaming_model.reset_streaming();
    }
    
    fn streaming_state_size(&self) -> usize {
        self.streaming_model.streaming_state_size()
    }
}
```

### Implementation Steps

1. **Extend TTS Builder**:
   - Add `.streaming()` method to enable streaming mode
   - Add `.stream()` method that returns `AudioChunkStream`
   - Modify existing `.await?` to support both modes

2. **Create Streaming Interface**:
   - Implement `AudioChunkStream` wrapper
   - Provide methods for incremental text input
   - Return audio chunks as they're generated

3. **State Management**:
   - Handle streaming state between chunks
   - Support reset/restart functionality
   - Manage memory and buffer sizes

4. **Integration with Conditions**:
   - Apply speaker conditioning to streaming
   - Support dynamic condition changes
   - Handle condition updates between chunks

### Key Integration Code

```rust
// In engine.rs - extend TTS builder
impl KyutaiTtsConversationBuilder {
    pub async fn stream(self) -> Result<impl Stream<Item = Vec<f32>>, MoshiError> {
        let tts_model = tts::Model::load(
            &self.model_paths.lm_model_path,
            &self.model_paths.mimi_model_path,
            DType::F32,
            &Device::Cpu,
        )?;
        
        let streaming_model = StreamingModel::new(Arc::new(Mutex::new(tts_model)));
        
        // Create async stream that yields audio chunks
        let stream = async_stream::stream! {
            for text_chunk in self.text_chunks {
                let audio = streaming_model.step(Some(&text_chunk), &conditions)?;
                yield audio;
            }
            
            // Final flush
            let final_audio = streaming_model.flush()?;
            if !final_audio.is_empty() {
                yield final_audio;
            }
        };
        
        Ok(stream)
    }
}
```

### Files to Modify
- `src/engine.rs` (add streaming support to builders)

### Files to Use (DO NOT MODIFY)
- `src/tts_streaming.rs` (complete streaming implementation)
- `src/streaming.rs` (StreamingModule trait)

### Testing
- Test incremental text-to-speech generation
- Verify low-latency audio output
- Test streaming state management
- Confirm proper cleanup and flushing

### Acceptance Criteria
- ✅ Supports real-time TTS streaming via fluent-voice API
- ✅ Uses existing `StreamingModel` implementation
- ✅ Provides incremental audio chunk output
- ✅ Maintains proper streaming state between chunks
- ✅ Integrates with existing condition system