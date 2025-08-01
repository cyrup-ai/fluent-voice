use crate::asr::label::{LabelState, LabeledAudio};
use crate::asr::{PredictIterator, Sample};

/// Labels an iterator of speech samples as either speech or non-speech according
/// to the provided speech sensitity.
pub struct LabelIterator<'a, T, I>
where
    I: Iterator,
{
    pub(super) iter: PredictIterator<'a, T, I>,
    pub(super) state: LabelState<T>,
}

impl<T, I> Iterator for LabelIterator<'_, T, I>
where
    T: Sample,
    I: Iterator<Item = T>,
{
    type Item = LabeledAudio<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(buffered) = self.state.try_buffer() {
            return Some(buffered);
        }

        for (chunk, probability) in self.iter.by_ref() {
            if let Some(audio) = self.state.try_next(chunk, probability) {
                return Some(audio);
            }
        }

        self.state.flush()
    }
}
