use crate::asr::decoder::token_id;
use crate::asr::model::WhisperModel;
use candle_core::{D, IndexOp, Result, Tensor};
use candle_transformers::models::whisper as m;
use tokenizers::Tokenizer;

/// Language codes supported by tiny-whisper.  Order must match the model.
pub const LANGUAGES: &[&str] = &[
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it",
    "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur",
    "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si",
    "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln",
    "ha", "ba", "jw", "su",
];

// -- LANGUAGES array exactly as you posted earlier --
// (omitted here to save space; keep it identical)

/// Detects the language of the first few seconds of audio and returns the
/// tokenizer-id of the `<|xx|>` token.
pub fn detect_language(
    model: &mut WhisperModel,
    tokenizer: &Tokenizer,
    mel: &Tensor,
) -> Result<u32> {
    let (_b, _c, seq_len) = mel.dims3()?;
    let mel = mel.narrow(
        2,
        0,
        usize::min(seq_len, model.config().max_source_positions),
    )?;
    let device = mel.device();

    let language_token_ids: Vec<_> = LANGUAGES
        .iter()
        .map(|t| {
            token_id(tokenizer, &format!("<|{t}|>"))
                .map_err(|e| candle_core::Error::Msg(e.to_string()))
        })
        .collect::<Result<Vec<_>>>()?;

    let sot =
        token_id(tokenizer, m::SOT_TOKEN).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let audio_features = model.encoder_forward(&mel, true)?;
    let tokens = Tensor::new(&[[sot]], device)?;
    let lang_ids = Tensor::new(language_token_ids.as_slice(), device)?;

    let ys = model.decoder_forward(&tokens, &audio_features, true)?;
    let logits = model.decoder_final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
    let logits = logits.index_select(&lang_ids, 0)?;
    let probs = candle_nn::ops::softmax(&logits, D::Minus1)?.to_vec1::<f32>()?;

    let (best, _) = probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap();
    Ok(language_token_ids[best])
}
