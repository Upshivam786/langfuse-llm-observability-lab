
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langfuse import get_client, observe, propagate_attributes
from langfuse.openai import OpenAI

# Reuse the credentials from Module 1.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / "01_basic_llm" / ".env")

MODEL = os.getenv("OPENAI_MODEL", "qwen/qwen3.7-plus")

llm = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.getenv(
        "OPENAI_BASE_URL",
        "https://openrouter.ai/api/v1",
    ),
)

langfuse = get_client()

# Our tiny local knowledge base.
DOCUMENTS = [
    {
        "id": "doc-001",
        "title": "Kubernetes Pods",
        "text": (
            "A Pod is the smallest deployable unit in Kubernetes. "
            "It contains one or more containers that share networking "
            "and can share storage."
        ),
        "keywords": ["pod", "pods", "container", "networking", "storage"],
    },
    {
        "id": "doc-002",
        "title": "Kubernetes Deployments",
        "text": (
            "A Deployment manages replicated application Pods. "
            "It supports scaling and rolling updates, and helps maintain "
            "the desired number of replicas."
        ),
        "keywords": ["deployment", "deployments", "replicas", "scaling", "updates"],
    },
    {
        "id": "doc-003",
        "title": "Kubernetes Services",
        "text": (
            "A Service provides a stable network endpoint for accessing "
            "a group of Pods. It helps applications communicate even "
            "when individual Pods are replaced."
        ),
        "keywords": ["service", "services", "network", "endpoint", "communication"],
    },
]


@observe(name="retrieve-documents")
def retrieve_documents(question: str, top_k: int = 2) -> list:
    """Retrieve documents using a simple keyword-overlap baseline."""
    question_words = set(question.lower().replace("?", "").split())

    ranked = []
    for document in DOCUMENTS:
        searchable_words = set(
            (document["title"] + " " + " ".join(document["keywords"]))
            .lower()
            .split()
        )
        score = len(question_words & searchable_words)

        if score > 0:
            ranked.append((score, document))

    ranked.sort(key=lambda item: item[0], reverse=True)
    results = [document for _, document in ranked[:top_k]]

    print(f"\nRetrieved {len(results)} document(s):")
    for document in results:
        print(f"- {document['id']}: {document['title']}")

    return results


@observe(name="generate-rag-answer")
def generate_answer(question: str, documents: list) -> str:
    """Generate an answer using only the retrieved context."""
    context = "\n\n".join(
        f"Source: {doc['title']}\n{doc['text']}"
        for doc in documents
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful Kubernetes tutor. Answer using the "
                "provided context. If the context does not contain the "
                "answer, say that the knowledge base does not provide "
                "enough information. Do not invent source facts."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Retrieved context:\n{context}"
            ),
        },
    ]

    response = llm.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=800,
    )

    answer = response.choices[0].message.content or ""

    print("\nAnswer:\n")
    print(answer)

    if response.usage:
        print("\nToken usage:")
        print("Input:", response.usage.prompt_tokens)
        print("Output:", response.usage.completion_tokens)
        print("Total:", response.usage.total_tokens)

    return answer


@observe(name="rag-workflow")
def run_rag(question: str) -> str:
    """Parent observation for the entire RAG workflow."""
    with propagate_attributes(
        user_id="learner_001",
        session_id="rag-learning-session-001",
        tags=["learning-lab", "rag", "keyword-retrieval"],
        metadata={
            "experiment": "rag-01-basic",
            "provider": "openrouter",
            "retrieval_method": "keyword-overlap",
            "application": "langfuse-masterclass",
        },
    ):
        start = time.perf_counter()

        documents = retrieve_documents(question)

        if not documents:
            answer = (
                "I could not find relevant documents in the knowledge base."
            )
            print("\n" + answer)
            return answer

        answer = generate_answer(question, documents)

        print(f"\nTotal RAG workflow time: {time.perf_counter() - start:.2f}s")
        return answer


def main():
    question = "What is a Kubernetes Pod, and how does it use containers?"

    try:
        run_rag(question)
    except Exception as exc:
        print(f"\nRAG workflow failed: {type(exc).__name__}: {exc}")
        raise
    finally:
        langfuse.flush()
        print("\nLangfuse flush requested.")


if __name__ == "__main__":
    main()
