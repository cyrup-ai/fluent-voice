use ort::{session::Session, session::builder::GraphOptimizationLevel};
use std::sync::{Arc, LazyLock};

use crate::asr::{Sample, error::Error};

/// A voice activity detector session.
#[derive(Debug)]
pub struct VoiceActivityDetector {
    session: Arc<Session>,
    chunk_size: usize,
    sample_rate: i64,
    h: std::sync::Arc<std::sync::Mutex<ndarray::Array3<f32>>>,
    c: std::sync::Arc<std::sync::Mutex<ndarray::Array3<f32>>>,
}

/// The silero ONNX model as bytes.
const MODEL: &[u8] = include_bytes!("onnx/silero_vad.onnx");

static DEFAULT_SESSION: LazyLock<Arc<Session>> = LazyLock::new(|| {
    Arc::new({
        Session::builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap()
            .with_intra_threads(1)
            .unwrap()
            .with_inter_threads(1)
            .unwrap()
            .commit_from_memory(MODEL)
            .unwrap()
    })
});

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
        self.h.lock().unwrap().fill(0f32);
        self.c.lock().unwrap().fill(0f32);
    }

    /// Predicts the existence of speech in a single iterable of audio.
    ///
    /// The samples iterator will be padded if it is too short, or truncated if it is
    /// too long.
    pub fn predict<S, I>(&mut self, samples: I) -> Result<f32, Box<dyn std::error::Error>>
    where
        S: Sample,
        I: IntoIterator<Item = S>,
    {
        let mut input = ndarray::Array2::<f32>::zeros((1, self.chunk_size));
        for (i, sample) in samples.into_iter().take(self.chunk_size).enumerate() {
            input[[0, i]] = sample.to_f32();
        }

        let sample_rate = ndarray::arr1::<i64>(&[self.sample_rate]);

        let h_guard = self.h.lock().unwrap();
        let c_guard = self.c.lock().unwrap();

        let input_tensor = ort::value::TensorRef::from_array_view(input.view())?;
        let sr_tensor = ort::value::TensorRef::from_array_view(sample_rate.view())?;
        let h_tensor = ort::value::TensorRef::from_array_view(h_guard.view())?;
        let c_tensor = ort::value::TensorRef::from_array_view(c_guard.view())?;

        let outputs = self.session.run(ort::inputs![
            "input" => input_tensor,
            "sr" => sr_tensor,
            "h" => h_tensor,
            "c" => c_tensor,
        ])?;

        // Update h and c recursively.
        let hn = outputs[0].try_extract_array::<f32>()?;
        let cn = outputs[1].try_extract_array::<f32>()?;

        drop(h_guard);
        drop(c_guard);

        {
            let mut h_guard = self.h.lock().unwrap();
            h_guard.assign(&hn);
        }
        {
            let mut c_guard = self.c.lock().unwrap();
            c_guard.assign(&cn);
        }

        // Get the probability of speech.
        let output = outputs[2].try_extract_array::<f32>()?;
        let probability = output[[0, 0]];

        Ok(probability)
    }
}

/// The configuration for the [VoiceActivityDetector]. Used to create
/// a [VoiceActivityDetectorBuilder] that performs runtime validation on build.
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

impl From<VoiceActivityDetectorConfig> for Result<VoiceActivityDetector, Error> {
    fn from(value: VoiceActivityDetectorConfig) -> Self {
        if (value.sample_rate as f32) / (value.chunk_size as f32) > 31.25 {
            return Err(Error::VadConfigError {
                sample_rate: value.sample_rate,
                chunk_size: value.chunk_size,
            });
        }

        let session = value.session.unwrap_or_else(|| DEFAULT_SESSION.clone());

        Ok(VoiceActivityDetector {
            session,
            chunk_size: value.chunk_size,
            sample_rate: value.sample_rate,
            h: std::sync::Arc::new(std::sync::Mutex::new(ndarray::Array3::<f32>::zeros((
                2, 1, 64,
            )))),
            c: std::sync::Arc::new(std::sync::Mutex::new(ndarray::Array3::<f32>::zeros((
                2, 1, 64,
            )))),
        })
    }
}
