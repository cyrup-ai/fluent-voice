//! Test to check available rodio methods

use rodio::*;

pub fn test_rodio_api() {
    // Try to use the methods we expect to exist
    let _stream_result = OutputStream::try_default(); // This should compile if method exists
    // let _sink_result = Sink::try_new(&handle); // This should compile if method exists
}
