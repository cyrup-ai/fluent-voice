//! BCP-47 language tag (e.g. "en-US").
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Language(pub &'static str);
