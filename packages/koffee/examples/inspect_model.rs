use anyhow::Result;
use koffee::wakewords::WakewordLoad;
use koffee::wakewords::wakeword_model::WakewordModel;

fn main() -> Result<()> {
    let model_path = "syrup.rpw";
    println!("Loading model from: {}", model_path);

    // Load the model
    let model = WakewordModel::load_from_file(model_path)?;

    // Print basic model info
    println!("✅ Model loaded successfully!");
    println!("   - Number of labels: {}", model.labels.len());
    println!("   - Train size: {}", model.train_size);
    println!("   - KFC size: {:?}", model.kfc_size);
    println!("   - Model type: {:?}", model.m_type);
    println!("   - RMS level: {}", model.rms_level);

    // Print weight information
    match &model.weights {
        koffee::wakewords::wakeword_model::ModelWeights::Map(weights) => {
            println!("✅ Found {} weight tensors:", weights.len());
            for (name, tensor) in weights {
                println!(
                    "   - {}: {:?} ({} bytes)",
                    name,
                    tensor.dims,
                    tensor.bytes.len()
                );
            }

            // Check for expected tensor keys based on model type
            let expected_tensors = match model.m_type {
                koffee::ModelType::Tiny => vec!["ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias"],
                _ => vec![
                    "ln1.weight",
                    "ln1.bias",
                    "ln2.weight",
                    "ln2.bias",
                    "ln3.weight",
                    "ln3.bias",
                ],
            };

            println!("\n🔍 Checking for expected tensors:");
            for tensor_name in &expected_tensors {
                if weights.contains_key(*tensor_name) {
                    println!("   ✅ Found expected tensor: {}", tensor_name);
                } else {
                    println!("   ❌ Missing expected tensor: {}", tensor_name);
                }
            }
        }
        koffee::wakewords::wakeword_model::ModelWeights::Raw(bytes) => {
            println!(
                "❌ Model contains raw weights ({} bytes) which is not supported for loading",
                bytes.len()
            );
            return Ok(());
        }
    }

    Ok(())
}
