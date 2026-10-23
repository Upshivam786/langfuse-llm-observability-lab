
import os
import time

from dotenv import load_dotenv
from langfuse import get_client, observe, propagate_attributes
from langfuse.openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.getenv(
        "OPENAI_BASE_URL",
        "https://openrouter.ai/api/v1",
    ),
)

langfuse = get_client()


@observe(name="validate-question")
def validate_question(question: str) -> str:
    """Validate the incoming question."""
    cleaned_question = question.strip()

    if not cleaned_question:
        raise ValueError("Question cannot be empty.")

    if len(cleaned_question) > 1000:
        raise ValueError("Question must be 1000 characters or fewer.")

    return cleaned_question


@observe(name="prepare-prompt")
def prepare_prompt(question: str) -> list:
    """Prepare the messages sent to the LLM."""
    return [
        {
            "role": "system",
            "content": (
                "You are a helpful DevOps tutor. "
                "Explain technical concepts clearly and concisely."
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]


@observe(name="devops-question-workflow")
def run_workflow(question: str) -> str:
    """Run the complete, instrumented question-answer workflow."""

    with propagate_attributes(
        user_id="learner_001",
        session_id="custom-spans-session-001",
        tags=["learning-lab", "custom-spans"],
        metadata={
            "application": "basic-llm-lab",
            "experiment": "03-custom-spans",
            "provider": "openrouter",
        },
    ):
        # Step 1: Validate the question.
        cleaned_question = validate_question(question)

        # Step 2: Prepare the prompt.
        messages = prepare_prompt(cleaned_question)

        # Step 3: Generate the answer.
        # The Langfuse OpenAI integration captures this generation.
        start = time.perf_counter()

        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "qwen/qwen3.7-plus"),
            messages=messages,
        )

        elapsed = time.perf_counter() - start
        answer = response.choices[0].message.content or ""

        print("\nQuestion:", cleaned_question)
        print("\nAnswer:\n", answer)
        print(f"\nLLM latency: {elapsed:.2f} seconds")

        if response.usage:
            print("\nToken usage:")
            print("Input:", response.usage.prompt_tokens)
            print("Output:", response.usage.completion_tokens)
            print("Total:", response.usage.total_tokens)

        return answer


def main():
    question = "Explain the difference between Docker images and containers."

    try:
        run_workflow(question)
    finally:
        # Flush even when the workflow raises an exception.
        langfuse.flush()
        print("\nLangfuse flush requested.")


if __name__ == "__main__":
    main()
