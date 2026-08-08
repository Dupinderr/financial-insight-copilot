import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

client = InferenceClient(token=os.getenv("HF_TOKEN"))

response = client.chat_completion(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    messages=[{"role": "user", "content": "Say hello in one word."}],
    max_tokens=20,
)

print(response.choices[0].message.content)