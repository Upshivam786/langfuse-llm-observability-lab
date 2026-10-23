
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

# Concepts required for a good answer to this question.
REQUIRED_CONCEPTS = [
    "deployment",
    "pod",
    "replica",
    "update",
]


def evaluate_answer(answer: str) -> dict:
    """Simple rule-based evaluation; not a semantic correctness judge."""
    answer_lower = answer.lower()

    matched = [
        concept
        for concept in REQUIRED_CONCEPTS
        if concept in answer_lower
    ]

    score = len(matched) / len(REQUIRED_CONCEPTS)

    return {
        "score": score,
        "matched": matched,
        "missing": [
            concept
            for concept in REQUIRED_CONCEPTS
            if concept not in matched
        ],
    }


@observe(name="llm-answer-evaluation")
def run_evaluation(question: str) -> str:
    with propagate_attributes(
        user_id="learner_001",
        session_id="evaluation-session-001",
        tags=["learning-lab", "evaluation", "experiment-08"],
        metadata={
            "experiment": "08-llm-evaluation",
            "provider": "openrouter",
            "evaluation_method": "keyword-coverage",
        },
    ):
        response = llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a DevOps tutor. Explain concepts accurately "
                        "for beginners. For Kubernetes Deployments, explain "
                        "pods, replicas, and how updates work."
                    ),
                },
                {"role": "user", "content": question},
            ],
            temperature=0.3,
        )

        answer = response.choices[0].message.content or ""

        result = evaluate_answer(answer)

        print("\nQuestion:", question)
        print("\nAnswer:\n", answer)

        print("\nEvaluation score:", f"{result['score']:.2f}")
        print("Matched concepts:", result["matched"])
        print("Missing concepts:", result["missing"])

        # Attach the numeric score to the active Langfuse trace.
        langfuse.score_current_trace(
            name="keyword-coverage",
            value=result["score"],
            data_type="NUMERIC",
            comment=(
                f"Matched {len(result['matched'])} of "
                f"{len(REQUIRED_CONCEPTS)} required concepts."
            ),
            metadata={
                "matched_concepts": result["matched"],
                "missing_concepts": result["missing"],
                "evaluation_method": "keyword-coverage",
            },
        )

        if response.usage:
            print("\nInput tokens:", response.usage.prompt_tokens)
            print("Output tokens:", response.usage.completion_tokens)
            print("Total tokens:", response.usage.total_tokens)

        return answer


def main():
    try:
        run_evaluation(
            "Explain Kubernetes Deployments and how they manage replicas "
            "and application updates."
        )
    except Exception as exc:
        print(f"\nEvaluation failed: {type(exc).__name__}: {exc}")
    finally:
        langfuse.flush()
        print("\nLangfuse flush requested.")


if __name__ == "__main__":
    main()
