# DIA-VOICE TODO LIST

## Ambient Agent + Deterministic Pipeline Architecture

### 1. Create Ambient Agent Interface
```rust
trait AmbientAgent: Send + Sync {
    async fn analyze(&self, context: ConversationContext) -> VoiceDirectives;
}

struct VoiceDirectives {
    tag_insertions: Vec<(usize, &'static str)>, // position, tag
    filter_params: FilterParameters,
    timing_adjustments: Vec<TimingCurve>,
}
```
- [ ] Fast inference model that analyzes Speaker config + text
- [ ] Returns concrete directives for deterministic execution
- [ ] Sub-10ms inference time for real-time operation

### 2. Deterministic Filter Pipeline
```rust
AudioPipeline::from_directives(directives: VoiceDirectives)
    .build() // No decisions, just execution
```
- [ ] Pipeline is purely deterministic - no choices at runtime
- [ ] All parameters come from ambient agent
- [ ] Reproducible given same directives

### 3. Agent Training Data Structure
```rust
struct TrainingExample {
    speaker_config: SpeakerConfig,
    text: String,
    optimal_tags: Vec<(usize, &'static str)>,
    optimal_filters: FilterParameters,
}
```
- [ ] Capture what makes "natural" tag placement
- [ ] Learn persona → emotion mapping patterns
- [ ] Train on voice actor performances

### 4. Two-Stage Processing
```rust
let directives = ambient_agent.analyze(
    ConversationContext {
        speaker: Speaker::named("Joey").with_persona_trait(VoicePersona::Nervous),
        text: "Oh, hey there! I wasn't expecting you so soon.",
        history: prev_utterances,
    }
).await?;

let pipeline = AudioPipeline::from_directives(directives);
let audio = pipeline.process(generated_audio);
```
- [ ] Stage 1: Agent makes all creative decisions (async, ML)
- [ ] Stage 2: Pipeline executes decisions (sync, deterministic)

### 5. Agent Context Window
```rust
struct ConversationContext {
    speaker: Box<dyn Speaker>,
    text: String,
    history: RingBuffer<Utterance>, // Last N exchanges
    ambient_profile: AmbientProfile, // Learned user preferences
}
```
- [ ] Agent sees conversation history for continuity
- [ ] Learns user's preferred style over time
- [ ] Adjusts tag frequency based on context

### 6. Caching for Common Patterns
```rust
DirectiveCache::new()
    .key_by(speaker_hash, text_hash)
    .ttl(Duration::from_secs(300))
```
- [ ] Cache agent decisions for repeated phrases
- [ ] "Hello!", "Goodbye", etc. don't need re-analysis
- [ ] Instant response for common utterances