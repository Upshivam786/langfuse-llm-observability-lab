
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


@observe(name="prepare-question")
def prepare_question(question: str) -> str:
    return question.strip()


@observe(name="evaluate-response")
def evaluate_response(answer: str) -> dict:
    required = ["deployment", "pod", "replica", "update"]
    answer_lower = answer.lower()
    matched = [term for term in required if term in answer_lower]

    return {
        "score": len(matched) / len(required),
        "matched": matched,
        "missing": [term for term in required if term not in matched],
    }


@observe(name="scoring-hierarchy-workflow")
def run_workflow(question: str) -> str:
    with propagate_attributes(
        user_id="learner_001",
        session_id="scoring-hierarchy-session-001",
        tags=["learning-lab", "hierarchy", "scoring"],
        metadata={
            "experiment": "09-scoring-hierarchy",
            "provider": "openrouter",
        },
    ):
        cleaned_question = prepare_question(question)

        response = llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a DevOps tutor. Explain Kubernetes "
                        "Deployments clearly. Include pods, replicas, "
                        "and how updates work."
                    ),
                },
                {"role": "user", "content": cleaned_question},
            ],
            temperature=0.3,
        )

        answer = response.choices[0].message.content or ""

        result = evaluate_response(answer)

        # Attach a numeric score to this workflow's current trace.
        langfuse.score_current_trace(
            name="keyword-coverage",
            value=result["score"],
            data_type="NUMERIC",
            comment=(
                f"Matched {len(result['matched'])} of 4 required concepts."
            ),
            metadata={
                "matched_concepts": result["matched"],
                "missing_concepts": result["missing"],
            },
        )

        print("\nQuestion:", cleaned_question)
        print("\nAnswer:\n", answer)
        print("\nKeyword coverage:", f"{result['score']:.2f}")
        print("Matched:", result["matched"])
        print("Missing:", result["missing"])

        if response.usage:
            print("\nInput tokens:", response.usage.prompt_tokens)
            print("Output tokens:", response.usage.completion_tokens)
            print("Total tokens:", response.usage.total_tokens)

        return answer


def main():
    try:
        run_workflow(
            "Explain Kubernetes Deployments, including pods, replicas, "
            "and how application updates work."
        )
    except Exception as exc:
        print(f"\nWorkflow failed: {type(exc).__name__}: {exc}")
    finally:
        langfuse.flush()
        print("\nLangfuse flush requested.")


if __name__ == "__main__":
    main()
