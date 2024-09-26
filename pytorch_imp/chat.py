from transformers import pipeline, AutoTokenizer 

generate = pipeline("text-generation", model="models/chat16", tokenizer="models/chat16")
tokenizer = AutoTokenizer.from_pretrained("models/chat16")
messages = [
    {
        "role": "user",
        "content": "What are some potential applications for quantum computing?",
    },
]

prompt = "<|im_start|>user\nWhat are some potential applications for quantum computing?<|im_end|>\n<|im_start|>assistant"
inputs = tokenizer(prompt, return_tensors="pt")
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
