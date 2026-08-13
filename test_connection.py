import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError(
        "OPENROUTER_API_KEY not found. "
        "Copy .env.example to .env and add your real key from openrouter.ai/keys"
    )

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

response = client.chat.completions.create(
    model="anthropic/claude-haiku-4.5",
    max_tokens=300,
    messages=[
        {"role": "user", "content": "In one sentence, what is pharmacovigilance?"}
    ],
)

print("Claude's response (via OpenRouter):")
print(response.choices[0].message.content)

print("\nIf you see a real answer above, your OpenRouter key and connection are working.")
