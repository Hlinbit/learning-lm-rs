mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;
mod user;

use std::{borrow::{Borrow, BorrowMut}, fmt::format, path::PathBuf};
use tokenizers::Tokenizer;
use std::io::{self, Write};

fn main() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let mut user_name = String::from("default");
    loop {
        print!(">>> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).expect("Failed to read line");

        // 去掉末尾的换行符
        let input = input.trim();
        match input {
            "login" => {user_name = user::login(); continue;},
            "logout" => {user::logout(&user_name); user_name = String::from("default"); continue;},
            "exit" => {break;},
            "quit" => {break;},
            _ => {}
        };
        let prompt = format!("<|im_start|>{} {}<|im_end|><|im_start|>assistant", user_name, input);

        let binding = tokenizer.encode(prompt, true).unwrap();
        let input_ids = binding.get_ids();
        // println!("\n{} \n{:?}", input, input_ids);
        let output_ids = llama.generate(
            input_ids,
            32,
            0.55,
            35,
            0.65,
        );
        println!("{}", tokenizer.decode(&output_ids, true).unwrap());
    }
}
