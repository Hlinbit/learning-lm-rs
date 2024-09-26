import torch
from safetensors.torch import load_file, save_file
import os

# 定义模型文件路径
input_model_path = "models/chat/model.safetensors"  # 输入的 f32 safetensor 文件
output_model_path = "models/chat16/model.safetensors"  # 输出的 f16 safetensor 文件

# 1. 加载 f32 模型
model_f32 = load_file(input_model_path)

# 2. 将模型权重从 f32 转换为 f16
model_f16 = {key: tensor.half() for key, tensor in model_f32.items()}
metadata = {
    "description": "FP16 version of the model",
    "author": "hlinbit",
    'format': 'pt'
}


# 3. 保存转换后的 f16 模型为 safetensors 格式
save_file(model_f16, output_model_path, metadata=metadata)

print(f"Model has been successfully converted to {output_model_path}")