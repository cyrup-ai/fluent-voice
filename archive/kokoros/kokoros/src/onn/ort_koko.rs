use ndarray::{ArrayBase, IxDyn, OwnedRepr};
use ort::{
    session::{Session, SessionInputValue, SessionOutputs},
    value::{Tensor, Value},
};
use std::cell::RefCell;

use super::ort_base;
use ort_base::OrtBase;

pub struct OrtKoko {
    sess: Option<RefCell<Session>>,
}
impl ort_base::OrtBase for OrtKoko {
    fn set_sess(&mut self, sess: Session) {
        self.sess = Some(RefCell::new(sess));
    }

    fn sess(&self) -> Option<&Session> {
        // Cannot return a direct reference to Session from RefCell
        // This method may need redesign or removal for RefCell compatibility
        None
    }
}
impl OrtKoko {
    pub fn new(model_path: String) -> Result<Self, String> {
        let mut instance = OrtKoko { sess: None };
        instance.load_model(model_path)?;
        Ok(instance)
    }

    pub fn infer(
        &self,
        tokens: Vec<Vec<i64>>,
        styles: Vec<Vec<f32>>,
        speed: f32,
    ) -> Result<ArrayBase<OwnedRepr<f32>, IxDyn>, Box<dyn std::error::Error>> {
        // inference koko
        // token, styles, speed
        // 1,N 1,256
        // [[0, 56, 51, 142, 156, 69, 63, 3, 16, 61, 4, 16, 156, 51, 4, 16, 62, 77, 156, 51, 86, 5, 0]]

        // Prepend 3 tokens to the first entry, to workaround initial silence issue
        // Make sure the first token is 0, I think it might be important?
        // let mut tokens = tokens;
        // let mut first_entry = tokens[0].clone();
        // let initial_pause = vec![0, 30, 30, 30];
        // first_entry.splice(0..1, initial_pause);
        // tokens[0] = first_entry;

        let shape = [tokens.len(), tokens[0].len()];
        let tokens_flat: Vec<i64> = tokens.into_iter().flatten().collect();
        let tokens = Tensor::from_array((shape, tokens_flat))?;
        let tokens_value: SessionInputValue = SessionInputValue::Owned(Value::from(tokens));

        let shape_style = [styles.len(), styles[0].len()];
        eprintln!("shape_style: {:?}", shape_style);
        let style_flat: Vec<f32> = styles.into_iter().flatten().collect();
        let style = Tensor::from_array((shape_style, style_flat))?;
        let style_value: SessionInputValue = SessionInputValue::Owned(Value::from(style));

        let speed = vec![speed; 1];
        let speed = Tensor::from_array(([1], speed))?;
        let speed_value: SessionInputValue = SessionInputValue::Owned(Value::from(speed));

        if let Some(sess_cell) = self.sess.as_ref() {
            let inputs = ort::inputs![
                "tokens" => tokens_value,
                "style" => style_value,
                "speed" => speed_value,
            ];
            let mut sess = sess_cell.borrow_mut();
            let outputs: SessionOutputs = sess.run(inputs)?;
            let (shape, data) = outputs["audio"]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("Failed to extract tensor: {}", e))?;
            let shape_dims: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
            let output = ArrayBase::from_shape_vec(shape_dims, data.iter().copied().collect())
                .map_err(|e| format!("Failed to create array from tensor: {}", e))?;
            Ok(output)
        } else {
            Err("Session is not initialized.".into())
        }
    }
}
