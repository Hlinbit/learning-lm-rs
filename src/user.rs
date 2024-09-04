use std::io::{self, Write};

pub fn login() -> String {
    // 提示用户输入用户名
    print!("Please enter your username: ");
    io::stdout().flush().unwrap(); // 确保提示符立即显示

    // 读取用户输入
    let mut username = String::new();
    io::stdin().read_line(&mut username).expect("Failed to read line");

    // 去掉用户名末尾的换行符
    let name = username.trim();

    // 打印欢迎信息
    println!("Welcome, {}!", name);
    return format!("{}", name);
}

pub fn logout(user_name: &str) -> bool {
    // 打印欢迎信息
    println!("GoodBye, {}!", user_name);
    return true;
}
