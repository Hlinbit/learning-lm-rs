from transformers import pipeline, AutoTokenizer 

generate = pipeline("text-generation", model="models/chat", tokenizer="models/chat")
tokenizer = AutoTokenizer.from_pretrained("models/chat")
messages = [
    {
        "role": "user",
        "content": "What are some potential applications for quantum computing?",
    },
]

prompt = "<|im_start|>user\nWhat are some potential applications for quantum computing?<|im_end|>\n<|im_start|>assistant"
inputs = tokenizer(prompt, return_tensors="pt")
print(inputs)
output = generate(
    prompt,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.65,
    top_k=35,
    top_p=0.55,
    repetition_penalty=1,
)

print(output[0]["generated_text"])


[32001,  2188,           13, 3195, 460, 741, 4628, 8429, 354, 10915, 21263, 28804, 32000, 28705, 13, 32001, 13892]
[32001,  2188,    28705, 13, 3195, 460, 741, 4628, 8429, 354, 10915, 21263, 28804, 32000, 28705, 13, 32001, 13892]