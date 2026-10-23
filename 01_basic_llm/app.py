
import os
import time

from dotenv import load_dotenv
from langfuse import get_client
from langfuse.openai import OpenAI

load_dotenv()

# OpenRouter-compatible OpenAI client.
# Langfuse automatically instruments supported calls.
client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.getenv(
        "OPENAI_BASE_URL",
        "https://openrouter.ai/api/v1",
    ),
)

langfuse = get_client()

def main():
    question = "Explain Kubernetes in simple terms with one example."

    print("\nSending question to Qwen3-8B...")
    start_time = time.perf_counter()

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "qwen/qwen3-8b"),
        messages=[
            {
                "role": "system",
                "content": "You are a helpful technical tutor. Be concise.",
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )

    elapsed = time.perf_counter() - start_time
    answer = response.choices[0].message.content

    print("\nQuestion:", question)
    print("\nAnswer:\n", answer)
    print(f"\nApplication latency: {elapsed:.2f} seconds")

    if response.usage:
        print("\nToken usage:")
        print("Input tokens:", response.usage.prompt_tokens)
        print("Output tokens:", response.usage.completion_tokens)
        print("Total tokens:", response.usage.total_tokens)

    # Ensure queued trace data is sent before the script exits.
    langfuse.flush()
    print("\nLangfuse trace flush requested.")

if __name__ == "__main__":
    main()
