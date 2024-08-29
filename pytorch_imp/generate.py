from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 指定模型目录
model_directory = "models/story"

# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_directory)

# 打印模型配置（可选）
print(model.config)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_directory)

# 输入文本
text = "Once upon a time"

# 将输入文本编码为模型的输入格式
inputs = tokenizer(text, return_tensors="pt")

# 生成文本
# 设置 `max_length` 为生成文本的最大长度
output_sequences = model.generate(
    input_ids=inputs["input_ids"],        # 输入的 token IDs
    max_length=50,                        # 生成文本的最大长度
    num_return_sequences=1,               # 生成的序列数量
    no_repeat_ngram_size=2,               # 防止重复的 n-gram
    repetition_penalty=1.5,               # 重复惩罚
    do_sample=True,                       # 是否使用抽样（如果为 False 则使用贪心搜索或束搜索）
    top_k=50,                             # Top-K 采样
    top_p=0.95,                           # Top-p (nucleus) 采样
    temperature=0.7                       # 采样温度
)

# 解码生成的 token IDs 为文本
generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

# 打印生成的文本
print(f"Generated text: {generated_text}")