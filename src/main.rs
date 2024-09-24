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
use crate::config::LlamaConfigJson;
use std::fs::File;
use half::f16;

enum LlamaEnum {
    F32(model::Llama<f32>),
    F16(model::Llama<f16>),
}

fn main() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat16");
    let config = File::open(model_dir.join("config.json")).unwrap();
    let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
    let llama: LlamaEnum;
    if config.torch_dtype == "float32" {
        llama = LlamaEnum::F32(model::Llama::<f32>::from_safetensors(&model_dir));
    } else {
        llama = LlamaEnum::F16(model::Llama::<f16>::from_safetensors(&model_dir));
    }
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
        let mut output_ids = vec![0u32,0];
        match llama {
            LlamaEnum::F32(ref llama_f32) => {
                // 使用 Llama<f32> 的逻辑
                output_ids = llama_f32.generate(
                    input_ids,
                    32,
                    0.55,
                    35,
                    0.65,
                );
            },
            LlamaEnum::F16(ref llama_f16) => {
                output_ids = llama_f16.generate(
                    input_ids,
                    128,
                    0.55,
                    35,
                    0.65,
                );
            }
        }
        
        println!("{}", tokenizer.decode(&output_ids, true).unwrap());
    }
}
