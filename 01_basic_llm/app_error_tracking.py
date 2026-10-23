import os

from dotenv import load_dotenv
from langfuse import get_client, observe, propagate_attributes
from langfuse.openai import OpenAI

load_dotenv()

# OpenRouter-compatible client with Langfuse instrumentation
llm = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.getenv(
        "OPENAI_BASE_URL",
        "https://openrouter.ai/api/v1",
    ),
)

langfuse = get_client()


@observe(name="validate-input")
def validate_input(question: str) -> str:
    """Validate input and deliberately raise an error for empty input."""
    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    return question


@observe(name="generate-answer")
def generate_answer(question: str) -> str:
    """Call the LLM after input validation."""
    response = llm.chat.completions.create(
        model="qwen/qwen3.7-plus",
        messages=[
            {
                "role": "system",
                "content": "You are a concise and helpful DevOps tutor.",
            },
            {"role": "user", "content": question},
        ],
    )

    return response.choices[0].message.content or ""


@observe(name="error-tracking-workflow")
def run_workflow(question: str) -> str:
    with propagate_attributes(
        user_id="learner_001",
        session_id="error-tracking-session-001",
        tags=["learning-lab", "error-tracking"],
        metadata={
            "experiment": "04-error-tracking",
            "provider": "openrouter",
        },
    ):
        cleaned_question = validate_input(question)
        return generate_answer(cleaned_question)


def main():
    # Change this to a normal question after testing the error.
    question = "Explain what Kubernetes pods are."

    try:
        answer = run_workflow(question)
        print("\nAnswer:\n", answer)

    except Exception as exc:
        print(f"\nWorkflow failed: {type(exc).__name__}: {exc}")

    finally:
        langfuse.flush()
        print("\nLangfuse flush requested.")


if __name__ == "__main__":
    main()
