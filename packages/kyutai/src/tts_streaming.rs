// src/tts_streaming.rs

use super::tts::Model;
use crate::conditioner::Condition;
use crate::error::MoshiError;
use crate::streaming::StreamingModule;
use candle_core::{Result, Tensor};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone)]
pub struct StreamingModel {
    inner: Arc<Mutex<Model>>,
    state: StreamingState,
}

#[derive(Debug, Clone)]
struct StreamingState {
    text_token: u32,
    audio_codes: Vec<u32>,
    pcm_buffer: Vec<f32>,
}

impl StreamingModel {
    pub fn new(model: Arc<Mutex<crate::tts::Model>>) -> Self {
        let model_ref = model.lock().unwrap();
        let text_start_token = model_ref.config().tts.text_start_token;
        let audio_pad_token = model_ref.lm().audio_pad_token();
        let audio_codebooks = model_ref.config().lm.audio_codebooks;
        drop(model_ref);

        Self {
            inner: model,
            state: StreamingState {
                text_token: text_start_token,
                audio_codes: vec![audio_pad_token; audio_codebooks],
                pcm_buffer: vec![],
            },
        }
    }

    pub fn step(
        &mut self,
        text: Option<&str>,
        conditions: &HashMap<String, Condition>,
    ) -> Result<Vec<f32>> {
        let mut model = self
            .inner
            .lock()
            .map_err(|_| MoshiError::Custom("Lock failed".into()))?;

        if let Some(text) = text {
            // Tokenize and set new text input
            // Assuming tokenizer is part of model or config
            // For simplicity, assume text is already tokenized to single token for streaming
            self.state.text_token = text
                .parse::<u32>()
                .map_err(|e| MoshiError::Custom(e.to_string()))?;
        }

        // Convert conditions from Condition enum to Tensor
        let tensor_conditions: HashMap<String, Tensor> = conditions
            .iter()
            .map(|(k, v)| {
                let tensor = match v {
                    crate::conditioner::Condition::Tensor(t) => t.clone(),
                    crate::conditioner::Condition::AddToInput(t) => t.clone(),
                };
                (k.clone(), tensor)
            })
            .collect();

        model.lm_mut().step_without_ca_src(
            self.state.text_token,
            &self.state.audio_codes,
            None,
            Some(&tensor_conditions),
        )?;

        if let Some(codes) = model.lm().last_audio_tokens() {
            let codes_tensor = Tensor::from_vec(
                codes.clone(),
                (1, 1, model.config().mimi_num_codebooks),
                model.lm().device(),
            )?;
            let pcm_step = model.mimi_mut().decode_step(&codes_tensor)?;
            if let Some(pcm) = pcm_step {
                let pcm_vec = pcm.to_vec1::<f32>()?;
                self.state.pcm_buffer.extend_from_slice(&pcm_vec);
                self.state.audio_codes = codes;
            }
        }

        let output = self.state.pcm_buffer.clone();
        self.state.pcm_buffer.clear();

        Ok(output)
    }

    pub fn flush(&mut self) -> Result<Vec<f32>> {
        let mut model = self
            .inner
            .lock()
            .map_err(|_| MoshiError::Custom("Lock failed".into()))?;
        model.mimi_mut().flush()?;
        let output = self.state.pcm_buffer.clone();
        self.state.pcm_buffer.clear();
        Ok(output)
    }

    fn step_internal(
        &mut self,
        conditions: &HashMap<String, crate::conditioner::Condition>,
    ) -> Result<Vec<f32>> {
        let mut model = self
            .inner
            .lock()
            .map_err(|_| MoshiError::Custom("Lock failed".into()))?;

        // Convert conditions from Condition enum to Tensor
        let tensor_conditions: HashMap<String, Tensor> = conditions
            .iter()
            .map(|(k, v)| {
                let tensor = match v {
                    crate::conditioner::Condition::Tensor(t) => t.clone(),
                    crate::conditioner::Condition::AddToInput(t) => t.clone(),
                };
                (k.clone(), tensor)
            })
            .collect();

        model.lm_mut().step_without_ca_src(
            self.state.text_token,
            &self.state.audio_codes,
            None,
            Some(&tensor_conditions),
        )?;

        if let Some(codes) = model.lm().last_audio_tokens() {
            let codes_tensor = Tensor::from_vec(
                codes.clone(),
                (1, 1, model.config().mimi_num_codebooks),
                model.lm().device(),
            )?;
            let pcm_step = model.mimi_mut().decode_step(&codes_tensor)?;
            if let Some(pcm) = pcm_step {
                let pcm_vec = pcm.to_vec1::<f32>()?;
                self.state.audio_codes = codes;
                return Ok(pcm_vec);
            }
        }

        Ok(vec![])
    }
}

impl StreamingModule for StreamingModel {
    fn forward_streaming(&mut self, input: &Tensor) -> Result<Tensor> {
        // For TTS streaming, input is text tokens, output is audio chunk
        let text_tokens = input.to_vec1::<u32>()?;
        let conditions = HashMap::new(); // Placeholder - would need actual conditioning

        // Process single token from input
        if let Some(&token) = text_tokens.first() {
            self.state.text_token = token;
        }

        let pcm = self.step_internal(&conditions)?;
        let pcm_len = pcm.len();
        Ok(Tensor::from_vec(pcm, (1, pcm_len), input.device())?)
    }

    fn reset_streaming(&mut self) {
        let model = self.inner.lock().unwrap();
        self.state.text_token = model.config().tts.text_start_token;
        self.state.audio_codes =
            vec![model.lm().audio_pad_token(); model.config().lm.audio_codebooks];
        self.state.pcm_buffer.clear();
    }

    fn streaming_state_size(&self) -> usize {
        self.state.pcm_buffer.len() + self.state.audio_codes.len()
    }
}
