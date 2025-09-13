# Replace Animator Mock Implementations

## Description
Replace mock implementations in animator examples with real LiveKit integration to provide production-quality examples for users.

## Current Violations
- `packages/animator/src/examples/amplitude_waveform_example.rs:12,117,128` - Uses `create_mock_audio_track()`
- `packages/animator/src/examples/video_rendering_example.rs:13,128,135` - Uses `create_mock_video_track()`

## Technical Resolution
Replace mock implementations with real LiveKit connections:

```rust
impl AudioTrackExample {
    pub async fn create_livekit_audio_track(
        room_url: &str,
        access_token: &str,
    ) -> Result<RemoteAudioTrack, LiveKitError> {
        let room_options = RoomOptions::default();
        let room = Room::connect(room_url, access_token, room_options).await?;
        
        let audio_track = room
            .remote_participants()
            .iter()
            .find_map(|participant| participant.audio_tracks().next())
            .ok_or(LiveKitError::NoAudioTrackAvailable)?;
        
        audio_track.set_enabled(true).await?;
        
        tracing::info!(
            track_id = %audio_track.sid(),
            participant = %audio_track.participant().identity(),
            "Connected to real LiveKit audio track"
        );
        
        Ok(audio_track)
    }
}
```

## Success Criteria
- [ ] Remove all `create_mock_audio_track()` usage
- [ ] Remove all `create_mock_video_track()` usage
- [ ] Implement real LiveKit room connections
- [ ] Add proper error handling for network issues
- [ ] Include authentication and connection management
- [ ] Provide working examples that connect to real LiveKit rooms
- [ ] Add comprehensive documentation for setup

## Dependencies
- Milestone 0: Async Architecture Compliance
- Milestone 1: Configuration Management
- Milestone 2: Audio Processing Enhancement

## Architecture Impact
LOW - Improves example quality but doesn't affect core functionality