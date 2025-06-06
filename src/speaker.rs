//! fluent_voice/src/speaker.rs
//! ---------------------------
//! Speaker trait definition

pub trait Speaker: Send + Sync {
    fn id(&self) -> &str;
}