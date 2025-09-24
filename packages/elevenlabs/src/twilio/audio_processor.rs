use crate::audio_decoders::{AudioFormatDecoder, KnownFormatDecoder};
use crate::audio_encoders::{AudioFormatEncoder, MulawEncoder};
use crate::shared::query_params::OutputFormat;
use crate::error::Error;
use crate::client::ElevenLabsClient;

/// High-level audio processor for Twilio WebSocket integration
pub struct TwilioAudioProcessor {
    mulaw_decoder: KnownFormatDecoder,
    mulaw_encoder: MulawEncoder,
    sample_rate: u32,
}

impl TwilioAudioProcessor {
    pub fn new() -> Self {
        Self {
            mulaw_decoder: KnownFormatDecoder::new(OutputFormat::MuLaw8000Hz),
            mulaw_encoder: MulawEncoder::new(),
            sample_rate: 8000,
        }
    }
    
    /// Convert Twilio μ-law audio to PCM for ElevenLabs processing
    pub fn twilio_to_pcm(&self, mulaw_data: &[u8]) -> Result<Vec<i16>, Error> {
        self.mulaw_decoder.decode_to_pcm(mulaw_data)
    }
    
    /// Convert ElevenLabs PCM output back to Twilio μ-law format
    pub fn pcm_to_twilio(&self, pcm_data: &[i16]) -> Result<Vec<u8>, Error> {
        self.mulaw_encoder.encode_from_pcm(pcm_data)
    }
    
    /// Process audio through complete Twilio → ElevenLabs → Twilio pipeline
    pub async fn process_with_elevenlabs(
        &self,
        mulaw_input: &[u8],
        elevenlabs_client: &ElevenLabsClient,
        processing_fn: impl FnOnce(Vec<i16>) -> Result<Vec<i16>, Error>,
    ) -> Result<Vec<u8>, Error> {
        // Step 1: Twilio μ-law → PCM
        let pcm_data = self.twilio_to_pcm(mulaw_input)?;
        
        // Step 2: Process with ElevenLabs (TTS, voice effects, etc.)
        let processed_pcm = processing_fn(pcm_data)?;
        
        // Step 3: PCM → Twilio μ-law
        self.pcm_to_twilio(&processed_pcm)
    }
    
    /// Real-time streaming processor for WebSocket audio
    pub fn process_streaming_chunk(
        &self,
        mulaw_chunk: &[u8],
        pcm_buffer: &mut Vec<i16>,
    ) -> Result<(), Error> {
        let pcm_samples = self.twilio_to_pcm(mulaw_chunk)?;
        pcm_buffer.extend(pcm_samples);
        Ok(())
    }
}

impl Default for TwilioAudioProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_twilio_audio_processor() {
        let processor = TwilioAudioProcessor::new();
        
        // Generate test μ-law data (sine wave)
        let mulaw_data: Vec<u8> = (0..160).map(|i| {
            let sample = (i as f32 * 2.0 * std::f32::consts::PI / 160.0).sin() * 1000.0;
            let pcm_samples = vec![sample as i16];
            let mulaw_bytes = processor.mulaw_encoder.encode_from_pcm(&pcm_samples).unwrap();
            mulaw_bytes[0]
        }).collect();
        
        // Test conversion pipeline
        let pcm_result = processor.twilio_to_pcm(&mulaw_data).unwrap();
        assert_eq!(pcm_result.len(), 160);
        
        let mulaw_result = processor.pcm_to_twilio(&pcm_result).unwrap();
        assert_eq!(mulaw_result.len(), 160);
    }
}
