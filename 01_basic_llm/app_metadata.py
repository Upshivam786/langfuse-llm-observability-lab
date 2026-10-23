
import os
import time
import uuid

from dotenv import load_dotenv
from langfuse import get_client, observe, propagate_attributes
from langfuse.openai import OpenAI

load_dotenv()

MODEL_NAME = "qwen/qwen3.7-plus"
APP_VERSION = "1.0.0"
ENVIRONMENT = "learning-lab"

llm = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.getenv(
        "OPENAI_BASE_URL",
        "https://openrouter.ai/api/v1",
    ),
)

langfuse = get_client()


@observe(name="metadata-llm-workflow")
def run_workflow(question: str) -> str:
    request_id = str(uuid.uuid4())

    with propagate_attributes(
        user_id="learner_001",
        session_id="metadata-experiment-session",
        tags=["learning-lab", "metadata", "experiment-05"],
        metadata={
            "application": "langfuse-masterclass",
            "app_version": APP_VERSION,
            "environment": ENVIRONMENT,
            "provider": "openrouter",
            "request_id": request_id,
        },
    ):
        start_time = time.perf_counter()

        response = llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a concise DevOps tutor. "
                        "Explain concepts clearly with one example."
                    ),
                },
                {"role": "user", "content": question},
            ],
            temperature=0.3,
        )

        latency = time.perf_counter() - start_time
        answer = response.choices[0].message.content or ""

        print("\nRequest ID:", request_id)
        print("Model:", MODEL_NAME)
        print("Application version:", APP_VERSION)
        print("Environment:", ENVIRONMENT)
        print(f"LLM latency: {latency:.2f} seconds")

        if response.usage:
            print("Input tokens:", response.usage.prompt_tokens)
            print("Output tokens:", response.usage.completion_tokens)
            print("Total tokens:", response.usage.total_tokens)

        print("\nAnswer:\n", answer)
        return answer


def main():
    question = "What is a Kubernetes Deployment?"

    try:
        run_workflow(question)
    except Exception as exc:
        print(f"\nWorkflow failed: {type(exc).__name__}: {exc}")
    finally:
        langfuse.flush()
        print("\nLangfuse flush requested.")


if __name__ == "__main__":
    main()
