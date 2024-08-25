from transformers import AutoModel, AutoTokenizer
import torch
# excute from project directory.
model_directory = "models/story"

model = AutoModel.from_pretrained(model_directory)

print(model.config)

for name, param in model.named_parameters():
    print(f"Name: {name}, Size: {param.size()}")

tokenizer = AutoTokenizer.from_pretrained(model_directory)
text = "Once upon a time"
inputs = tokenizer(text, return_tensors="pt")
outputs_dict = {}

def hook_fn(layer_name):
    def hook(module, input, output):
        outputs_dict[layer_name] = {
            "input": input,
            "output": output
        }
    return hook

    

# 注册钩子
for name, layer in model.named_modules():
    layer_name = f"transformer_layer_{name}"
    layer.register_forward_hook(hook_fn(layer_name))

# 执行推理
with torch.no_grad():
    print(model(**inputs))



for layer_name, data in outputs_dict.items():
    print(f"Layer: {layer_name}")
    if isinstance(data['input'], tuple):
        for t in data['input']:
            if isinstance(t , torch.Tensor):
                print(f"Input shape: {t.shape}")
    else:
        print(f"Input shape: {data['input'].shape}")

    if isinstance(data['output'], tuple):
        for t in data['output']:
            if isinstance(t , torch.Tensor):
                print(f"Output shape: {t.shape}")
    elif isinstance(data['output'], torch.Tensor):
        print(f"Output shape: {data['output'].shape}")
    else:
        print(f"Output type: {type(t)}")
    print(f"Input: {data['input']}")
    print(f"Output: {data['output']}")
    print()