
import os

from dotenv import load_dotenv
from langfuse import get_client, observe, propagate_attributes
from langfuse.openai import OpenAI

load_dotenv()

langfuse = get_client()

llm = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.getenv(
        "OPENAI_BASE_URL",
        "https://openrouter.ai/api/v1",
    ),
)

MODEL_NAME = "qwen/qwen3.7-plus"


@observe(name="validate-reliable-input")
def validate_input(question: str) -> str:
    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    return question


@observe(name="reliable-tracing-workflow")
def run_workflow(question: str) -> str:
    with propagate_attributes(
        user_id="learner_001",
        session_id="reliable-tracing-session-001",
        tags=["learning-lab", "reliability", "experiment-10"],
        metadata={
            "experiment": "10-reliable-tracing",
            "provider": "openrouter",
        },
    ):
        cleaned_question = validate_input(question)

        response = llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a concise DevOps tutor. "
                        "Explain technical concepts clearly for beginners."
                    ),
                },
                {"role": "user", "content": cleaned_question},
            ],
            temperature=0.3,
        )

        return response.choices[0].message.content or ""


def run_test(label: str, question: str) -> None:
    print(f"\n--- {label} ---")

    try:
        answer = run_workflow(question)
        print("\nAnswer:\n", answer)

    except Exception as exc:
        # Record that the application caught the exception.
        # Re-raise it so Langfuse can mark the observed workflow as failed.
        print(f"\nCaught exception: {type(exc).__name__}: {exc}")
        raise

    finally:
        # Ensure buffered Langfuse events are sent after each test.
        langfuse.flush()
        print("Langfuse flush requested.")


def main():
    # Test 1: a successful request
    try:
        run_test(
            "SUCCESS TEST",
            "Explain what a Kubernetes Pod is in simple terms.",
        )
    except Exception:
        print("Success test unexpectedly failed.")

    # Test 2: intentionally invalid input
    try:
        run_test("ERROR TEST", "   ")
    except ValueError:
        print("Expected validation failure confirmed.")

    finally:
        langfuse.flush()
        print("\nFinal Langfuse flush requested.")


if __name__ == "__main__":
    main()
