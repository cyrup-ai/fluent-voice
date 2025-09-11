use anyhow::Result;
use futures::{SinkExt, channel::mpsc};
use std::collections::HashMap;
use tokio::task::JoinHandle;

use crate::playback;

pub use crate::playback::{RemoteVideoFrame, play_remote_video_track};
use crate::{LocalTrack, Participant, RemoteTrack, RoomEvent, TrackPublication};

#[derive(Clone, Debug)]
pub struct RemoteVideoTrack(pub livekit::track::RemoteVideoTrack);
#[derive(Clone, Debug)]
pub struct RemoteAudioTrack(pub livekit::track::RemoteAudioTrack);
#[derive(Clone, Debug)]
pub struct RemoteTrackPublication(pub livekit::publication::RemoteTrackPublication);
#[derive(Clone, Debug)]
pub struct RemoteParticipant(pub livekit::participant::RemoteParticipant);

#[derive(Clone, Debug)]
pub struct LocalVideoTrack(pub livekit::track::LocalVideoTrack);
#[derive(Clone, Debug)]
pub struct LocalAudioTrack(pub livekit::track::LocalAudioTrack);
#[derive(Clone, Debug)]
pub struct LocalTrackPublication(pub livekit::publication::LocalTrackPublication);
#[derive(Clone, Debug)]
pub struct LocalParticipant(pub livekit::participant::LocalParticipant);

pub struct Room {
    room: livekit::Room,
    _task: JoinHandle<()>,
    playback: playback::AudioStack,
}

pub type TrackSid = livekit::id::TrackSid;
pub type ConnectionState = livekit::ConnectionState;
#[derive(Clone, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
pub struct ParticipantIdentity(pub String);

impl Room {
    pub async fn connect(
        url: String,
        token: String,
    ) -> Result<(Self, mpsc::UnboundedReceiver<RoomEvent>)> {
        // Use default room configuration for simplicity and reliability
        let config = livekit::RoomOptions::default();
        let (room, mut events) = livekit::Room::connect(&url, &token, config).await?;

        let (mut tx, rx) = mpsc::unbounded();
        let task = tokio::spawn(async move {
            while let Some(event) = events.recv().await {
                if let Some(event) = room_event_from_livekit(event) {
                    tx.send(event).await.ok();
                }
            }
        });

        Ok((
            Self {
                room,
                _task: task,
                playback: playback::AudioStack::new(tokio::runtime::Handle::current())?,
            },
            rx,
        ))
    }

    pub fn local_participant(&self) -> LocalParticipant {
        LocalParticipant(self.room.local_participant())
    }

    pub fn remote_participants(&self) -> HashMap<ParticipantIdentity, RemoteParticipant> {
        self.room
            .remote_participants()
            .into_iter()
            .map(|(k, v)| (ParticipantIdentity(k.0), RemoteParticipant(v)))
            .collect()
    }

    pub fn connection_state(&self) -> ConnectionState {
        self.room.connection_state()
    }

    pub async fn publish_local_microphone_track(
        &self,
    ) -> Result<(LocalTrackPublication, playback::AudioStream)> {
        let (track, stream) = self.playback.capture_local_microphone_track()?;
        let publication = self
            .local_participant()
            .publish_track(
                livekit::track::LocalTrack::Audio(track.0),
                livekit::options::TrackPublishOptions {
                    source: livekit::track::TrackSource::Microphone,
                    ..Default::default()
                },
            )
            .await?;

        Ok((publication, stream))
    }

    pub async fn unpublish_local_track(&self, sid: TrackSid) -> Result<LocalTrackPublication> {
        self.local_participant().unpublish_track(sid).await
    }

    pub fn play_remote_audio_track(
        &self,
        track: &RemoteAudioTrack,
    ) -> Result<playback::AudioStream> {
        Ok(self.playback.play_remote_audio_track(track))
    }
}

impl LocalParticipant {
    // Placeholder for publishing screenshare track
    // This will need to be reimplemented with a custom screen capture solution
    // that doesn't rely on gpui

    async fn publish_track(
        &self,
        track: livekit::track::LocalTrack,
        options: livekit::options::TrackPublishOptions,
    ) -> Result<LocalTrackPublication> {
        let participant = self.0.clone();
        participant
            .publish_track(track, options)
            .await
            .map(LocalTrackPublication)
            .map_err(|error| anyhow::anyhow!("failed to publish track: {error}"))
    }

    pub async fn unpublish_track(&self, sid: TrackSid) -> Result<LocalTrackPublication> {
        let participant = self.0.clone();
        participant
            .unpublish_track(&sid)
            .await
            .map(LocalTrackPublication)
            .map_err(|error| anyhow::anyhow!("failed to unpublish track: {error}"))
    }
}

impl LocalTrackPublication {
    pub fn mute(&self) {
        let track = self.0.clone();
        tokio::spawn(async move {
            track.mute();
        });
    }

    pub fn unmute(&self) {
        let track = self.0.clone();
        tokio::spawn(async move {
            track.unmute();
        });
    }

    pub fn sid(&self) -> TrackSid {
        self.0.sid()
    }

    pub fn is_muted(&self) -> bool {
        self.0.is_muted()
    }
}

impl RemoteParticipant {
    pub fn identity(&self) -> ParticipantIdentity {
        ParticipantIdentity(self.0.identity().0)
    }

    pub fn track_publications(&self) -> HashMap<TrackSid, RemoteTrackPublication> {
        self.0
            .track_publications()
            .into_iter()
            .map(|(sid, publication)| (sid, RemoteTrackPublication(publication)))
            .collect()
    }
}

impl RemoteAudioTrack {
    pub fn sid(&self) -> TrackSid {
        self.0.sid()
    }
}

impl RemoteVideoTrack {
    pub fn sid(&self) -> TrackSid {
        self.0.sid()
    }
}

impl RemoteTrackPublication {
    pub fn is_muted(&self) -> bool {
        self.0.is_muted()
    }

    pub fn is_enabled(&self) -> bool {
        self.0.is_enabled()
    }

    pub fn track(&self) -> Option<RemoteTrack> {
        self.0.track().map(remote_track_from_livekit)
    }

    pub fn is_audio(&self) -> bool {
        self.0.kind() == livekit::track::TrackKind::Audio
    }

    pub fn set_enabled(&self, enabled: bool) {
        let track = self.0.clone();
        tokio::spawn(async move { track.set_enabled(enabled) });
    }

    pub fn sid(&self) -> TrackSid {
        self.0.sid()
    }
}

impl Participant {
    pub fn identity(&self) -> ParticipantIdentity {
        match self {
            Participant::Local(local_participant) => {
                ParticipantIdentity(local_participant.0.identity().0)
            }
            Participant::Remote(remote_participant) => {
                ParticipantIdentity(remote_participant.0.identity().0)
            }
        }
    }
}

fn participant_from_livekit(participant: livekit::participant::Participant) -> Participant {
    match participant {
        livekit::participant::Participant::Local(local) => {
            Participant::Local(LocalParticipant(local))
        }
        livekit::participant::Participant::Remote(remote) => {
            Participant::Remote(RemoteParticipant(remote))
        }
    }
}

fn publication_from_livekit(
    publication: livekit::publication::TrackPublication,
) -> TrackPublication {
    match publication {
        livekit::publication::TrackPublication::Local(local) => {
            TrackPublication::Local(LocalTrackPublication(local))
        }
        livekit::publication::TrackPublication::Remote(remote) => {
            TrackPublication::Remote(RemoteTrackPublication(remote))
        }
    }
}

fn remote_track_from_livekit(track: livekit::track::RemoteTrack) -> RemoteTrack {
    match track {
        livekit::track::RemoteTrack::Audio(audio) => RemoteTrack::Audio(RemoteAudioTrack(audio)),
        livekit::track::RemoteTrack::Video(video) => RemoteTrack::Video(RemoteVideoTrack(video)),
    }
}

fn local_track_from_livekit(track: livekit::track::LocalTrack) -> LocalTrack {
    match track {
        livekit::track::LocalTrack::Audio(audio) => LocalTrack::Audio(LocalAudioTrack(audio)),
        livekit::track::LocalTrack::Video(video) => LocalTrack::Video(LocalVideoTrack(video)),
    }
}
fn room_event_from_livekit(event: livekit::RoomEvent) -> Option<RoomEvent> {
    let event = match event {
        livekit::RoomEvent::ParticipantConnected(remote_participant) => {
            RoomEvent::ParticipantConnected(RemoteParticipant(remote_participant))
        }
        livekit::RoomEvent::ParticipantDisconnected(remote_participant) => {
            RoomEvent::ParticipantDisconnected(RemoteParticipant(remote_participant))
        }
        livekit::RoomEvent::LocalTrackPublished {
            publication,
            track,
            participant,
        } => RoomEvent::LocalTrackPublished {
            publication: LocalTrackPublication(publication),
            track: local_track_from_livekit(track),
            participant: LocalParticipant(participant),
        },
        livekit::RoomEvent::LocalTrackUnpublished {
            publication,
            participant,
        } => RoomEvent::LocalTrackUnpublished {
            publication: LocalTrackPublication(publication),
            participant: LocalParticipant(participant),
        },
        livekit::RoomEvent::LocalTrackSubscribed { track } => RoomEvent::LocalTrackSubscribed {
            track: local_track_from_livekit(track),
        },
        livekit::RoomEvent::TrackSubscribed {
            track,
            publication,
            participant,
        } => RoomEvent::TrackSubscribed {
            track: remote_track_from_livekit(track),
            publication: RemoteTrackPublication(publication),
            participant: RemoteParticipant(participant),
        },
        livekit::RoomEvent::TrackUnsubscribed {
            track,
            publication,
            participant,
        } => RoomEvent::TrackUnsubscribed {
            track: remote_track_from_livekit(track),
            publication: RemoteTrackPublication(publication),
            participant: RemoteParticipant(participant),
        },
        livekit::RoomEvent::TrackSubscriptionFailed {
            participant,
            error: _,
            track_sid,
        } => RoomEvent::TrackSubscriptionFailed {
            participant: RemoteParticipant(participant),
            track_sid,
        },
        livekit::RoomEvent::TrackPublished {
            publication,
            participant,
        } => RoomEvent::TrackPublished {
            publication: RemoteTrackPublication(publication),
            participant: RemoteParticipant(participant),
        },
        livekit::RoomEvent::TrackUnpublished {
            publication,
            participant,
        } => RoomEvent::TrackUnpublished {
            publication: RemoteTrackPublication(publication),
            participant: RemoteParticipant(participant),
        },
        livekit::RoomEvent::TrackMuted {
            participant,
            publication,
        } => RoomEvent::TrackMuted {
            publication: publication_from_livekit(publication),
            participant: participant_from_livekit(participant),
        },
        livekit::RoomEvent::TrackUnmuted {
            participant,
            publication,
        } => RoomEvent::TrackUnmuted {
            publication: publication_from_livekit(publication),
            participant: participant_from_livekit(participant),
        },
        livekit::RoomEvent::RoomMetadataChanged {
            old_metadata,
            metadata,
        } => RoomEvent::RoomMetadataChanged {
            old_metadata,
            metadata,
        },
        livekit::RoomEvent::ParticipantMetadataChanged {
            participant,
            old_metadata,
            metadata,
        } => RoomEvent::ParticipantMetadataChanged {
            participant: participant_from_livekit(participant),
            old_metadata,
            metadata,
        },
        livekit::RoomEvent::ParticipantNameChanged {
            participant,
            old_name,
            name,
        } => RoomEvent::ParticipantNameChanged {
            participant: participant_from_livekit(participant),
            old_name,
            name,
        },
        livekit::RoomEvent::ParticipantAttributesChanged {
            participant,
            changed_attributes,
        } => RoomEvent::ParticipantAttributesChanged {
            participant: participant_from_livekit(participant),
            changed_attributes: changed_attributes.into_iter().collect(),
        },
        livekit::RoomEvent::ActiveSpeakersChanged { speakers } => {
            RoomEvent::ActiveSpeakersChanged {
                speakers: speakers.into_iter().map(participant_from_livekit).collect(),
            }
        }
        livekit::RoomEvent::Connected {
            participants_with_tracks,
        } => RoomEvent::Connected {
            participants_with_tracks: participants_with_tracks
                .into_iter()
                .map({
                    |(p, t)| {
                        (
                            RemoteParticipant(p),
                            t.into_iter().map(RemoteTrackPublication).collect(),
                        )
                    }
                })
                .collect(),
        },
        livekit::RoomEvent::Disconnected { reason } => RoomEvent::Disconnected {
            reason: reason.as_str_name(),
        },
        livekit::RoomEvent::Reconnecting => RoomEvent::Reconnecting,
        livekit::RoomEvent::Reconnected => RoomEvent::Reconnected,
        _ => {
            log::trace!("dropping livekit event: {event:?}");
            return None;
        }
    };

    Some(event)
}
