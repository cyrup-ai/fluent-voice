/// Input module for matrix data structures and data sources

/// Matrix data structure for storing multi-dimensional audio data
pub type Matrix<T> = Vec<Vec<T>>;

/// Data source enumeration for various input types
#[derive(Debug, Clone)]
pub enum DataSource {
    /// Microphone input
    Microphone,
    /// File input
    File(String),
    /// Network stream
    Network(String),
    /// Test signal generator
    TestSignal,
}

impl Default for DataSource {
    fn default() -> Self {
        DataSource::TestSignal
    }
}

impl DataSource {
    /// Get a human-readable name for the data source
    pub fn name(&self) -> &str {
        match self {
            DataSource::Microphone => "Microphone",
            DataSource::File(_) => "File",
            DataSource::Network(_) => "Network",
            DataSource::TestSignal => "Test Signal",
        }
    }
}
