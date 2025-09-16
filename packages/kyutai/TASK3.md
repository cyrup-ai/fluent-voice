# TASK3: Wire Existing STT Implementation

## Status: INTEGRATION - Connect Working ASR to Fluent-Voice API

### Objective
Wire the existing `asr::State::step_pcm()` method into `KyutaiSttConversationBuilder` to provide speech recognition through the fluent-voice interface.

### THE OTHER END OF THE WIRE: Fluent-Voice Traits to Implement

**Primary Trait**: `SttConversationBuilder` from `/Volumes/samsung_t9/fluent-voice/packages/fluent-voice/src/stt_conversation.rs`

```rust
pub trait SttConversationBuilder: Sized + Send {
    type Conversation: SttConversation;
    
    // Required methods to implement:
    fn with_source(self, src: SpeechSource) -> Self;
    fn vad_mode(self, mode: VadMode) -> Self;
    fn noise_reduction(self, level: NoiseReduction) -> Self;
    fn language_hint(self, lang: Language) -> Self;
    fn diarization(self, d: Diarization) -> Self;
    fn word_timestamps(self, w: WordTimestamps) -> Self;
    fn timestamps_granularity(self, g: TimestampsGranularity) -> Self;
    fn punctuation(self, p: Punctuation) -> Self;
    
    // THE CRITICAL METHOD FROM TASK1:
    fn on_prediction<F>(self, f: F) -> Self
    where F: FnMut(String, String) + Send + 'static;
    
    // Result and event handlers:
    fn on_result<F>(self, f: F) -> Self
    where F: FnMut(VoiceError) -> String + Send + 'static;
    fn on_wake<F>(self, f: F) -> Self
    where F: FnMut(String) + Send + 'static;
    fn on_turn_detected<F>(self, f: F) -> Self
    where F: FnMut(Option<String>, String) + Send + 'static;
}
```

**Supporting Traits**: 
- `SttConversation` - Session object with `into_stream()` method
- `MicrophoneBuilder` - For live microphone input 
- `TranscriptionBuilder` - For file transcription

**Domain Types**: From `/Volumes/samsung_t9/fluent-voice/packages/domain/src/`
- `TranscriptionSegment` - Individual transcript segments with timing
- `SpeechSource` - Audio input source (microphone or file)
- `VadMode`, `NoiseReduction`, etc. - Configuration enums
- `VoiceError` - Error handling

### Existing Working Implementation
**File: `src/asr.rs`**
- Complete `State::step_pcm()` method (lines 87-97)
- Processes PCM audio → `Vec<Word>` with timestamps
- `Word` struct contains tokens, start_time, stop_time
- Integrates with `LmModel` and `Mimi` for audio processing
- `StateBuilder` for configuration

### Integration Points

#### 1. ASR State Initialization
**Use existing**: `asr::StateBuilder`
```rust
let asr_state = StateBuilder::new()
    .asr_delay_in_tokens(6)
    .build(audio_tokenizer, lm_model)?;
```

#### 2. Audio Processing
**Use existing**: `State::step_pcm()`
```rust
let words = asr_state.step_pcm(pcm_tensor, |text_token, audio_token| {
    // Handle intermediate predictions
    Ok(())
})?;
```

#### 3. Word Extraction
**Use existing**: `Word` struct with timing
```rust
for word in words {
    println!("Word: {:?} [{:.2}s - {:.2}s]", 
             word.tokens, word.start_time, word.stop_time);
}
```

### Implementation Steps

1. **Modify `KyutaiSttConversationBuilder.await?`**:
   - Load models using existing `KyutaiModelManager`
   - Initialize `asr::State` with `LmModel` and `Mimi`
   - Set up audio processing pipeline

2. **Audio Stream Processing**:
   - Convert incoming audio to `Tensor` format expected by `step_pcm()`
   - Handle continuous audio streaming
   - Process audio chunks with appropriate buffer sizes

3. **Event Handler Integration**:
   - Wire `step_pcm()` callback to existing event handlers
   - Call stored callbacks from TASK1 fix
   - Convert `Word` timestamps to expected format

4. **Output Conversion**:
   - Convert `Word.tokens` to text using tokenizer
   - Format timing information for fluent-voice API
   - Handle word-level vs sentence-level outputs

### Key Integration Code

```rust
// In KyutaiSttConversationBuilder.await?
let manager = KyutaiModelManager::new();
let paths = manager.download_models().await?;

// Load models (reuse TTS loading pattern)
let lm_model = LmModel::load(paths.lm_model_path)?;
let mimi = Mimi::load(paths.mimi_model_path)?;

// Create ASR state
let mut asr_state = StateBuilder::new()
    .asr_delay_in_tokens(6)
    .build(mimi, lm_model)?;

// Process audio stream
loop {
    let pcm_chunk = audio_stream.next().await?;
    let words = asr_state.step_pcm(pcm_chunk, |text_token, audio_token| {
        // Call prediction callbacks from TASK1
        if let Some(callback) = &mut self.prediction_callback {
            callback(format!("{}", text_token), "intermediate".to_string());
        }
        Ok(())
    })?;
    
    // Handle completed words
    for word in words {
        // Convert tokens to text and trigger word callbacks
        let text = tokenizer.decode(&word.tokens)?;
        // Call word completion handlers
    }
}
```

### Files to Modify
- `src/engine.rs` (KyutaiSttConversationBuilder implementation)

### Files to Use (DO NOT MODIFY)
- `src/asr.rs` (complete working ASR implementation)
- `src/models.rs` (model downloading)

### Testing
- Process live microphone input
- Verify word-level timestamps
- Test prediction callbacks from TASK1
- Confirm text accuracy and timing

### Acceptance Criteria
- ✅ Audio input produces text output via fluent-voice API
- ✅ Uses existing `asr::State::step_pcm()` implementation
- ✅ Provides word-level timing information
- ✅ Integrates with prediction callbacks from TASK1
- ✅ No code duplication or reimplementation