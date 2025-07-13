mod wakeword_comp;
mod wakeword_ref_build;
pub(crate) use wakeword_comp::WakewordComparator;
pub use wakeword_ref_build::{
    build_from_files as WakewordRefBuildFromFiles, build_from_iter as WakewordRefBuildFromBuffers,
};
