mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;

use std::path::PathBuf;
use tokenizers::Tokenizer;

fn main() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let input = "<|im_start|>user
What are some potential applications for quantum computing?<|im_end|>
<|im_start|>assistant";
    let binding = tokenizer.encode(input, true).unwrap();
    let input_ids = binding.get_ids();
    println!("\n{} \n{:?}", input, input_ids);
    let output_ids = llama.generate(
        input_ids,
        256,
        0.55,
        35,
        0.65,
    );
    println!("{}", tokenizer.decode(&output_ids, true).unwrap());
}
