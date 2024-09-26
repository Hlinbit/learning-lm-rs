from safetensors import safe_open

# 指定 Safetensor 文件路径
file_path = "models/chat16/model.safetensors"

# 使用 safe_open 打开文件
with safe_open(file_path, framework="numpy") as f:
    # 打印文件的元信息（包含张量的名字和形状）
    metadata = f.metadata()
    print("Meta Information:", metadata)
    # for key in f.keys():
        # print(f"Tensor Name: {key}")
        # print("Tensor:", tensor_data.shape)
        # print()