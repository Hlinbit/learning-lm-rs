from transformers import pipeline

generate = pipeline("text-generation", "Felladrin/Minueza-32M-UltraChat")

messages = [
    {
        "role": "system",
        "content": "You are a highly knowledgeable and friendly assistant. Your goal is to understand and respond to user inquiries with clarity. Your interactions are always respectful, helpful, and focused on delivering the most accurate information to the user.",
    },
    {
        "role": "user",
        "content": "Hey! Got a question for you!",
    },
    {
        "role": "assistant",
        "content": "Sure! What's it?",
    },
    {
        "role": "user",
        "content": "What are some potential applications for quantum computing?",
    },
]

prompt = generate.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

output = generate(
    prompt,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.65,
    top_k=35,
    top_p=0.55,
    repetition_penalty=1.176,
)

print(output[0]["generated_text"])
