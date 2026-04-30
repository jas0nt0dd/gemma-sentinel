# test_model.py — paste in terminal as one block
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
    base_url=os.getenv("API_BASE_URL"),
    api_key=os.getenv("HF_TOKEN")
)

model_name = os.getenv("MODEL_NAME") or "gpt-5.4-mini"

resp = client.chat.completions.create(
    model=model_name,
    messages=[{"role": "user", "content": "Reply with: OK"}],
    max_tokens=10
)
print("✅ Model works:", resp.choices[0].message.content)