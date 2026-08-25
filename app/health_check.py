
import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

MODELS_TO_CHECK = ["openrouter/free", "meta-llama/llama-3.3-70b-instruct:free"]


def check_model(model_name: str) -> bool:
    print(f"Checking {model_name} ...", end=" ", flush=True)
    start = time.time()
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Reply with just the word: OK"}],
            max_tokens=10,
        )
        elapsed = time.time() - start
        content = response.choices[0].message.content if response.choices else None
        if content:
            print(f"WORKING ({elapsed:.1f}s) — replied: {content.strip()!r}")
            return True
        print(f"EMPTY RESPONSE ({elapsed:.1f}s)")
        return False
    except Exception as e:
        elapsed = time.time() - start
        print(f"FAILED ({elapsed:.1f}s) — {e}")
        return False


def main():
    print("=" * 60)
    print("Field Notes — pre-demo health check")
    print("=" * 60)

    results = {model: check_model(model) for model in MODELS_TO_CHECK}

    print("=" * 60)
    if any(results.values()):
        print("At least one model is working — you're good to demo.")
        if not results.get(MODELS_TO_CHECK[0]):
            print(f"NOTE: primary model ({MODELS_TO_CHECK[0]}) failed — "
                  f"the agent will still work but every answer will be "
                  f"slower, since it'll fail the primary before falling back.")
        sys.exit(0)
    else:
        print("ALL MODELS FAILED. The agent will not be able to answer "
              "questions right now. Check your OPENROUTER_API_KEY, your "
              "internet connection, or OpenRouter's status page before "
              "presenting.")
        sys.exit(1)


if __name__ == "__main__":
    main()