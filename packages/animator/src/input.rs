/// Input module for matrix data structures and data sources
/// Matrix data structure for storing multi-dimensional audio data
pub type Matrix<T> = Vec<Vec<T>>;

/// Data source trait for various input types
pub trait DataSource<T> {
    /// Receive data from the source
    fn recv(&mut self) -> Option<Matrix<T>>;
}

/// Data source enumeration for various input types
#[derive(Debug, Clone)]
pub enum DataSourceType {
    /// Microphone input
    Microphone,
    /// File input
    File(String),
    /// Network stream
    Network(String),
    /// Test signal generator
    TestSignal,
}

impl Default for DataSourceType {
    fn default() -> Self {
        DataSourceType::TestSignal
    }
}

impl DataSourceType {
    /// Get a human-readable name for the data source
    pub fn name(&self) -> &str {
        match self {
            DataSourceType::Microphone => "Microphone",
            DataSourceType::File(_) => "File",
            DataSourceType::Network(_) => "Network",
            DataSourceType::TestSignal => "Test Signal",
        }
    }
}
