use std::io::{self, Write};
use crate::{LlamaEnum, kvcache::CacheManager, model::Llama};

pub fn login(llama: &mut LlamaEnum) -> String {
    // 提示用户输入用户名
    print!("Please enter your username: ");
    io::stdout().flush().unwrap(); // 确保提示符立即显示

    // 读取用户输入
    let mut username = String::new();
    io::stdin().read_line(&mut username).expect("Failed to read line");

    // 去掉用户名末尾的换行符
    let name = username.trim();
    match llama {
        LlamaEnum::F32(ref mut llama_f32) => {
            llama_f32.load_cache(name);
        },
        LlamaEnum::F16(ref mut llama_f16) => {
            llama_f16.load_cache(name);
        }
    }

    // 打印欢迎信息
    println!("Welcome, {}!", name);
    return format!("{}", name);
}

pub fn logout(user_name: &str, llama: &mut LlamaEnum) -> bool {
    // 打印欢迎信息

    match llama {
        LlamaEnum::F32(ref mut llama_f32) => {
            llama_f32.store_cache();
        },
        LlamaEnum::F16(ref mut llama_f16) => {
            llama_f16.store_cache();
        }
    }

    println!("GoodBye, {}!", user_name);
    return true;
}
