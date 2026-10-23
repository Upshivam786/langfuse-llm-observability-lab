
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
TOPIC = "Kubernetes Deployments"


@observe(name="prompt-version-comparison")
def run_comparison(version: int) -> str:
    print(f"\n{'=' * 55}")
    print(f"Testing prompt version {version}")
    print("=" * 55)

    # Fetch the exact version, not whichever version is latest.
    prompt = langfuse.get_prompt(
        PROMPT_NAME,
        version=version,
    )

    compiled_prompt = prompt.compile(topic=TOPIC)

    print("Prompt:", PROMPT_NAME)
    print("Version:", prompt.version)

    with propagate_attributes(
        user_id="learner_001",
        session_id=f"prompt-version-{version}-session",
        tags=[
            "learning-lab",
            "prompt-versioning",
            f"prompt-v{version}",
        ],
        metadata={
            "experiment": "07-prompt-versioning",
            "prompt_name": PROMPT_NAME,
            "prompt_version": version,
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
                    "content": f"Explain {TOPIC}.",
                },
            ],
            temperature=0.3,
            langfuse_prompt=prompt,
        )

        latency = time.perf_counter() - start
        answer = response.choices[0].message.content or ""

        print(f"\nLatency: {latency:.2f} seconds")

        if response.usage:
            print("Input tokens:", response.usage.prompt_tokens)
            print("Output tokens:", response.usage.completion_tokens)
            print("Total tokens:", response.usage.total_tokens)

        print("\nAnswer:\n", answer)
        return answer


def main():
    try:
        # Run both versions against the same topic and model.
        answer_v1 = run_comparison(version=1)
        answer_v2 = run_comparison(version=2)

        print("\n\nCOMPARISON COMPLETE")
        print("Version 1 answer length:", len(answer_v1), "characters")
        print("Version 2 answer length:", len(answer_v2), "characters")

    except Exception as exc:
        print(f"\nExperiment failed: {type(exc).__name__}: {exc}")

    finally:
        langfuse.flush()
        print("\nLangfuse flush requested.")


if __name__ == "__main__":
    main()
