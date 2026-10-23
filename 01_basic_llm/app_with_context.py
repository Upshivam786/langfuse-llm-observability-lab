
import os
import time

from dotenv import load_dotenv
from langfuse import get_client, observe, propagate_attributes
from langfuse.openai import OpenAI

load_dotenv()

# OpenRouter client using the OpenAI-compatible API.
client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.getenv(
        "OPENAI_BASE_URL",
        "https://openrouter.ai/api/v1",
    ),
)

langfuse = get_client()


@observe(name="context-aware-llm-call")
def ask_question(
    question: str,
    user_id: str,
    session_id: str,
) -> str:
    """Ask an LLM question and attach context to its Langfuse trace."""

    print("\nSending question to Qwen3-8B...")
    start = time.perf_counter()

    # Propagate context to the active trace and child observations.
    with propagate_attributes(
        user_id=user_id,
        session_id=session_id,
        tags=["learning-lab", "openrouter", "qwen"],
        metadata={
            "application": "basic-llm-lab",
            "environment": "development",
            "provider": "openrouter",
            "experiment": "02-context",
        },
    ):
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "qwen/qwen3-8b"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful technical tutor. Be concise."
                    ),
                },
                {
                    "role": "user",
                    "content": question,
                },
            ],
        )

    elapsed = time.perf_counter() - start
    answer = response.choices[0].message.content or ""

    print("\nQuestion:", question)
    print("\nAnswer:\n", answer)
    print(f"\nLLM latency: {elapsed:.2f} seconds")

    if response.usage:
        print("\nToken usage:")
        print("Input:", response.usage.prompt_tokens)
        print("Output:", response.usage.completion_tokens)
        print("Total:", response.usage.total_tokens)

    return answer


def main():
    ask_question(
        question="What is the difference between Docker and Kubernetes?",
        user_id="learner_001",
        session_id="devops-learning-session-001",
    )

    # Flush queued Langfuse events before the script exits.
    langfuse.flush()
    print("\nLangfuse flush requested.")


if __name__ == "__main__":
    main()
