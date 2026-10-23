
import os
import time

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
PROMPT_NAME = "devops-tutor-prompt"


@observe(name="prompt-management-workflow")
def run_workflow(topic: str) -> str:
    # Fetch the production version of our centrally managed prompt.
    prompt = langfuse.get_prompt(
        PROMPT_NAME,
        label="production",
    )

    # Replace {{topic}} with the value supplied at runtime.
    compiled_prompt = prompt.compile(topic=topic)

    print("Prompt name:", PROMPT_NAME)
    print("Prompt version:", prompt.version)
    print("Topic:", topic)
    print("\nCompiled prompt:\n", compiled_prompt)

    with propagate_attributes(
        user_id="learner_001",
        session_id="prompt-management-session-001",
        tags=["learning-lab", "prompt-management", "experiment-06"],
        metadata={
            "experiment": "06-prompt-management",
            "provider": "openrouter",
        },
    ):
        start = time.perf_counter()

        response = llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": compiled_prompt},
                {
                    "role": "user",
                    "content": f"Explain {topic}.",
                },
            ],
            temperature=0.3,
            langfuse_prompt=prompt,
        )

        latency = time.perf_counter() - start
        answer = response.choices[0].message.content or ""

        print("\nAnswer:\n", answer)
        print(f"\nLLM latency: {latency:.2f} seconds")

        if response.usage:
            print("\nToken usage:")
            print("Input:", response.usage.prompt_tokens)
            print("Output:", response.usage.completion_tokens)
            print("Total:", response.usage.total_tokens)

        return answer


def main():
    try:
        run_workflow("Kubernetes Deployments")
    except Exception as exc:
        print(f"\nWorkflow failed: {type(exc).__name__}: {exc}")
    finally:
        langfuse.flush()
        print("\nLangfuse flush requested.")


if __name__ == "__main__":
    main()
