//! Bridges ASR final sentences → LLM suggestions.

use crossbeam_channel::{select, Receiver, Sender};

use crate::asr::whisper_stream::PartialTranscript;

/// Listens on `asr_rx`, pushes every finalised sentence to the LLM,
/// and forwards the rolling suggestion back to the UI.
pub fn spawn_predictive(
    asr_rx: Receiver<PartialTranscript>,
    llm_prompt: Sender<String>,
    llm_stream: Receiver<String>,
    ui_tx: Sender<String>,
) {
    std::thread::spawn(move || {
        let mut current_line = String::new();
        loop {
            select! {
                recv(asr_rx) -> msg => match msg {
                    Ok(PartialTranscript::Final(s)) => {
                        current_line = s.clone();
                        let _ = llm_prompt.send(current_line.clone());
                    }
                    _ => {}
                },
                recv(llm_stream) -> tok => {
                    if let Ok(t) = tok {
                        let combined = format!("{current_line}▍{t}");
                        let _ = ui_tx.send(combined);
                    }
                }
            }
        }
    });
}
