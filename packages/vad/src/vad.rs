#[cfg(feature = "ort")]
use ort::{session::Session, session::builder::GraphOptimizationLevel, value::TensorRef};
use std::sync::Arc;

use crate::{Sample, error::Error};

/// A voice activity detector session.
#[cfg(feature = "ort")]
#[derive(Debug)]
pub struct VoiceActivityDetector {
    session: Session,
    chunk_size: usize,
    sample_rate: i64,
    h: ndarray::Array3<f32>,
    c: ndarray::Array3<f32>,
}

/// The silero ONNX model as bytes.
const MODEL: &[u8] = include_bytes!("../onnx/silero_vad.onnx");

/// Creates an optimized ONNX session for VAD inference.
/// Returns Result to handle any initialization errors gracefully.
#[cfg(feature = "ort")]
fn create_optimized_session() -> Result<Session, Error> {
    Session::builder()
        .map_err(|e| Error::PredictionFailed(format!("Failed to create session builder: {e}")))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| Error::PredictionFailed(format!("Failed to set optimization level: {e}")))?
        .with_intra_threads(1)
        .map_err(|e| Error::PredictionFailed(format!("Failed to set intra threads: {e}")))?
        .with_inter_threads(1)
        .map_err(|e| Error::PredictionFailed(format!("Failed to set inter threads: {e}")))?
        .commit_from_memory(MODEL)
        .map_err(|e| Error::PredictionFailed(format!("Failed to load model: {e}")))
}

#[cfg(feature = "ort")]
impl VoiceActivityDetector {
    /// Create a new [VoiceActivityDetectorBuilder].
    pub fn builder() -> VoiceActivityDetectorBuilder {
        VoiceActivityDetectorConfig::builder()
    }

    /// Gets the chunks size
    pub(crate) fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Resets the state of the voice activity detector session.
    pub fn reset(&mut self) {
        self.h.fill(0f32);
        self.c.fill(0f32);
    }

    /// Predicts the existence of speech in a single iterable of audio.
    ///
    /// The samples iterator will be padded if it is too short, or truncated if it is
    /// too long.
    pub fn predict<S, I>(&mut self, samples: I) -> Result<f32, Error>
    where
        S: Sample,
        I: IntoIterator<Item = S>,
    {
        let mut input = ndarray::Array2::<f32>::zeros((1, self.chunk_size));
        for (i, sample) in samples.into_iter().take(self.chunk_size).enumerate() {
            input[[0, i]] = sample.to_f32();
        }

        let sample_rate = ndarray::arr1::<i64>(&[self.sample_rate]);

        // Create tensor references for zero-allocation input
        let input_tensor = TensorRef::from_array_view(input.view())
            .map_err(|e| Error::PredictionFailed(format!("Failed to create input tensor: {e}")))?;
        let sr_tensor = TensorRef::from_array_view(sample_rate.view()).map_err(|e| {
            Error::PredictionFailed(format!("Failed to create sample rate tensor: {e}"))
        })?;
        let h_tensor = TensorRef::from_array_view(self.h.view())
            .map_err(|e| Error::PredictionFailed(format!("Failed to create h tensor: {e}")))?;
        let c_tensor = TensorRef::from_array_view(self.c.view())
            .map_err(|e| Error::PredictionFailed(format!("Failed to create c tensor: {e}")))?;

        let inputs = ort::inputs![
            "input" => input_tensor,
            "sr" => sr_tensor,
            "h" => h_tensor,
            "c" => c_tensor,
        ];

        let outputs = self
            .session
            .run(inputs)
            .map_err(|e| Error::PredictionFailed(e.to_string()))?;

        // Extract h and c state updates with zero-allocation array extraction
        let hn_array = outputs
            .get("hn")
            .ok_or_else(|| Error::PredictionFailed("Missing 'hn' output".to_string()))?
            .try_extract_array::<f32>()
            .map_err(|e| Error::PredictionFailed(format!("Failed to extract hn array: {e}")))?
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|e| Error::PredictionFailed(format!("Failed to reshape hn array: {e}")))?;

        let cn_array = outputs
            .get("cn")
            .ok_or_else(|| Error::PredictionFailed("Missing 'cn' output".to_string()))?
            .try_extract_array::<f32>()
            .map_err(|e| Error::PredictionFailed(format!("Failed to extract cn array: {e}")))?
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|e| Error::PredictionFailed(format!("Failed to reshape cn array: {e}")))?;

        // Update internal state efficiently
        self.h.assign(&hn_array);
        self.c.assign(&cn_array);

        // Extract probability with zero-allocation
        let output_array = outputs
            .get("output")
            .ok_or_else(|| Error::PredictionFailed("Missing 'output' output".to_string()))?
            .try_extract_array::<f32>()
            .map_err(|e| Error::PredictionFailed(format!("Failed to extract output array: {e}")))?;

        let probability = output_array[[0]];

        Ok(probability)
    }
}

/// The configuration for the [VoiceActivityDetector]. Used to create
/// a [VoiceActivityDetectorBuilder] that performs runtime validation on build.
#[cfg(feature = "ort")]
#[derive(Debug, typed_builder::TypedBuilder)]
#[builder(
    builder_method(vis = ""),
    builder_type(name = VoiceActivityDetectorBuilder, vis = "pub"),
    build_method(into = Result<VoiceActivityDetector, Error>, vis = "pub"))
]
struct VoiceActivityDetectorConfig {
    #[builder(setter(into))]
    chunk_size: usize,
    #[builder(setter(into))]
    sample_rate: i64,
    #[builder(default, setter(strip_option))]
    session: Option<Arc<Session>>,
}

#[cfg(feature = "ort")]
impl From<VoiceActivityDetectorConfig> for Result<VoiceActivityDetector, Error> {
    fn from(value: VoiceActivityDetectorConfig) -> Self {
        if (value.sample_rate as f32) / (value.chunk_size as f32) > 31.25 {
            return Err(Error::VadConfigError {
                sample_rate: value.sample_rate,
                chunk_size: value.chunk_size,
            });
        }

        let session = if let Some(session) = value.session {
            // Extract the session from the Arc if provided
            Arc::try_unwrap(session)
                .map_err(|_| Error::PredictionFailed("Cannot unwrap shared session".to_string()))?
        } else {
            // Create a new optimized session for blazing-fast performance
            create_optimized_session()?
        };

        Ok(VoiceActivityDetector {
            session,
            chunk_size: value.chunk_size,
            sample_rate: value.sample_rate,
            h: ndarray::Array3::<f32>::zeros((2, 1, 64)),
            c: ndarray::Array3::<f32>::zeros((2, 1, 64)),
        })
    }
}
