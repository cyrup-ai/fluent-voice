#[cfg(all(
    not(feature = "microphone"),
    not(feature = "encodec"),
    not(feature = "mimi"),
    not(feature = "snac")
))]
compile_error!("At least one audio feature must be enabled: microphone, encodec, mimi, or snac");

use std::collections::HashMap;

pub mod client_util;
mod livekit_client;
pub mod playback;
mod remote_video_track_view;
pub mod util;

#[cfg(any(test, feature = "test", all(target_os = "windows", target_env = "gnu")))]
pub mod test {
    // Mock functionality will be implemented here when needed
}

pub use livekit_client::*;
pub use playback::AudioStream;
pub use remote_video_track_view::{RemoteVideoTrackView, RemoteVideoTrackViewEvent};

// Re-export raw_window_handle for consumers
pub use raw_window_handle;
// pub use wgpu; // Available through ratagpu if needed

#[derive(Debug, Clone)]
pub enum Participant {
    Local(livekit_client::LocalParticipant),
    Remote(livekit_client::RemoteParticipant),
}

#[derive(Debug, Clone)]
pub enum TrackPublication {
    Local(livekit_client::LocalTrackPublication),
    Remote(livekit_client::RemoteTrackPublication),
}

impl TrackPublication {
    pub fn sid(&self) -> livekit_client::TrackSid {
        match self {
            TrackPublication::Local(local) => local.sid(),
            TrackPublication::Remote(remote) => remote.sid(),
        }
    }

    pub fn is_muted(&self) -> bool {
        match self {
            TrackPublication::Local(local) => local.is_muted(),
            TrackPublication::Remote(remote) => remote.is_muted(),
        }
    }
}

#[derive(Clone, Debug)]
pub enum RemoteTrack {
    Audio(livekit_client::RemoteAudioTrack),
    Video(livekit_client::RemoteVideoTrack),
}

impl RemoteTrack {
    pub fn sid(&self) -> livekit_client::TrackSid {
        match self {
            RemoteTrack::Audio(remote_audio_track) => remote_audio_track.sid(),
            RemoteTrack::Video(remote_video_track) => remote_video_track.sid(),
        }
    }
}

#[derive(Clone, Debug)]
pub enum LocalTrack {
    Audio(livekit_client::LocalAudioTrack),
    Video(livekit_client::LocalVideoTrack),
}

#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum RoomEvent {
    ParticipantConnected(livekit_client::RemoteParticipant),
    ParticipantDisconnected(livekit_client::RemoteParticipant),
    LocalTrackPublished {
        publication: livekit_client::LocalTrackPublication,
        track: LocalTrack,
        participant: livekit_client::LocalParticipant,
    },
    LocalTrackUnpublished {
        publication: livekit_client::LocalTrackPublication,
        participant: livekit_client::LocalParticipant,
    },
    LocalTrackSubscribed {
        track: LocalTrack,
    },
    TrackSubscribed {
        track: RemoteTrack,
        publication: livekit_client::RemoteTrackPublication,
        participant: livekit_client::RemoteParticipant,
    },
    TrackUnsubscribed {
        track: RemoteTrack,
        publication: livekit_client::RemoteTrackPublication,
        participant: livekit_client::RemoteParticipant,
    },
    TrackSubscriptionFailed {
        participant: livekit_client::RemoteParticipant,
        // error: livekit::track::TrackError,
        track_sid: livekit_client::TrackSid,
    },
    TrackPublished {
        publication: livekit_client::RemoteTrackPublication,
        participant: livekit_client::RemoteParticipant,
    },
    TrackUnpublished {
        publication: livekit_client::RemoteTrackPublication,
        participant: livekit_client::RemoteParticipant,
    },
    TrackMuted {
        participant: Participant,
        publication: TrackPublication,
    },
    TrackUnmuted {
        participant: Participant,
        publication: TrackPublication,
    },
    RoomMetadataChanged {
        old_metadata: String,
        metadata: String,
    },
    ParticipantMetadataChanged {
        participant: Participant,
        old_metadata: String,
        metadata: String,
    },
    ParticipantNameChanged {
        participant: Participant,
        old_name: String,
        name: String,
    },
    ParticipantAttributesChanged {
        participant: Participant,
        changed_attributes: HashMap<String, String>,
    },
    ActiveSpeakersChanged {
        speakers: Vec<Participant>,
    },
    ConnectionStateChanged(livekit_client::ConnectionState),
    Connected {
        participants_with_tracks: Vec<(
            livekit_client::RemoteParticipant,
            Vec<livekit_client::RemoteTrackPublication>,
        )>,
    },
    Disconnected {
        reason: &'static str,
    },
    Reconnecting,
    Reconnected,
}
