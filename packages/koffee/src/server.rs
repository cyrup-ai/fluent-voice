//! TCP server functionality for Koffee

use crate::{KoffeeCandle, Result};

/// Run a TCP server that accepts audio input and returns wake-word detections.
pub fn run_tcp(_detector: &mut KoffeeCandle, port: u16) -> Result<()> {
    // TODO: Implement TCP server functionality
    eprintln!("TCP server on port {port} not yet implemented");
    Ok(())
}
