
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from langfuse import get_client, observe, propagate_attributes


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / "01_basic_llm" / ".env")

MODEL = os.getenv("OPENAI_MODEL", "qwen/qwen3.7-plus")
MAX_TOKENS = 200
TOP_K = 2

llm = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
)

langfuse = get_client()


# --------------------------------------------------
# 2. Sample knowledge base
# --------------------------------------------------

DOCUMENTS = [
    {
        "id": "doc-001",
        "title": "Kubernetes Pods",
        "text": (
            "A Pod is the smallest deployable unit in Kubernetes. "
            "It contains one or more containers that share networking "
            "and storage resources."
        ),
        "keywords": ["pod", "pods", "container", "containers",
                     "networking", "storage"],
    },
    {
        "id": "doc-002",
        "title": "Kubernetes Deployments",
        "text": (
            "A Deployment manages replicated application Pods. "
            "It supports scaling replicas, rolling updates, "
            "and rolling back application versions."
        ),
        "keywords": ["deployment", "deployments", "replicas",
                     "scaling", "updates", "rollback"],
    },
    {
        "id": "doc-003",
        "title": "Kubernetes Services",
        "text": (
            "A Service exposes an application running in Pods over "
            "a stable network endpoint. It enables communication "
            "between applications and can distribute traffic "
            "to matching Pods."
        ),
        "keywords": ["service", "services", "endpoint",
                     "communication", "traffic", "network"],
    },
]


# --------------------------------------------------
# 3. Retrieval configuration
# --------------------------------------------------

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "what", "how", "why", "when", "where", "which",
    "do", "does", "did", "and", "or", "to", "of",
    "in", "on", "for", "with", "between", "from",
    "explain", "describe", "tell", "me", "about",
    "can", "could", "would", "should", "please",
    "it", "its", "this", "that", "these", "those",
    "be", "by", "as", "at", "into", "your",
    "kubernetes",
}


def tokenize(text):
    """Convert text into lowercase word tokens."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def meaningful_tokens(text):
    """Remove common words that do not help identify relevance."""
    return tokenize(text) - STOP_WORDS


# --------------------------------------------------
# 4. Retrieval with diagnostics
# --------------------------------------------------

@observe(name="retrieve-with-diagnostics")
def retrieve_documents(question, top_k=TOP_K):
    question_words = meaningful_tokens(question)
    diagnostics = []

    for doc in DOCUMENTS:
        document_text = (
            doc["title"] + " " +
            doc["text"] + " " +
            " ".join(doc["keywords"])
        )

        document_words = meaningful_tokens(document_text)
        matched_words = sorted(question_words & document_words)

        # Basic keyword-overlap score
        score = len(matched_words)

        diagnostics.append({
            "id": doc["id"],
            "title": doc["title"],
            "score": score,
            "matched_words": matched_words,
            "document": doc,
        })

    # Highest score first; preserve original order for ties
    ranked = sorted(
        diagnostics,
        key=lambda item: item["score"],
        reverse=True,
    )

    # Exclude documents with zero relevance
    selected = [
        item["document"]
        for item in ranked
        if item["score"] > 0
    ][:top_k]

    print("\nRetrieval diagnostics:")
    for item in ranked:
        print(
            f"- {item['id']} | {item['title']} | "
            f"score={item['score']} | "
            f"matched={item['matched_words']}"
        )

    print(f"\nSelected {len(selected)} document(s).")
    print(
        f"Retrieved {len(selected)} of "
        f"{len(DOCUMENTS)} candidate documents."
    )

    result = {
        "selected": selected,
        "candidate_count": len(DOCUMENTS),
        "retrieved_count": len(selected),
        "no_results": len(selected) == 0,
        "diagnostics": [
            {
                "id": item["id"],
                "title": item["title"],
                "score": item["score"],
                "matched_words": item["matched_words"],
            }
            for item in ranked
        ],
    }

    # Attach retrieval details to the Langfuse observation
    langfuse.update_current_span(
        input={"question": question},
        output={
            "candidate_count": result["candidate_count"],
            "retrieved_count": result["retrieved_count"],
            "no_results": result["no_results"],
            "diagnostics": result["diagnostics"],
        },
        metadata={
            "retrieval_method": "keyword-overlap",
            "top_k": top_k,
        },
    )

    return result


# --------------------------------------------------
# 5. Answer generation
# --------------------------------------------------

@observe(name="generate-answer-from-retrieval")
def generate_answer(question, documents):
    if not documents:
        answer = (
            "I could not find relevant information in the "
            "available Kubernetes knowledge base. "
            "This question may be outside the knowledge base's scope."
        )

        langfuse.update_current_span(
            input={"question": question, "documents": []},
            output={"answer": answer, "generation_skipped": True},
        )
        return answer

    context = "\n\n".join(
        f"Document: {doc['title']}\n{doc['text']}"
        for doc in documents
    )

    response = llm.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the question using only the supplied context. "
                    "If the context does not contain the answer, say so. "
                    "Be concise and accurate."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}\n\n"
                    "Give a direct answer in a few sentences."
                ),
            },
        ],
        max_tokens=MAX_TOKENS,
    )

    choice = response.choices[0]
    message = choice.message
    answer = message.content or ""

    usage = response.usage
    input_tokens = usage.prompt_tokens if usage else 0
    output_tokens = usage.completion_tokens if usage else 0

    if not answer.strip():
        # Some models may return reasoning separately from final content.
        reasoning = getattr(message, "reasoning", None)

        print("\nWarning: The model returned empty answer content.")
        print("Finish reason:", choice.finish_reason)

        if reasoning:
            print("A reasoning field was present, but no final answer was returned.")

        answer = (
            "The model returned an empty answer. "
            "Check the finish reason and response details before retrying."
        )

    langfuse.update_current_span(
        input={
            "question": question,
            "context_documents": [
                {"id": doc["id"], "title": doc["title"]}
                for doc in documents
            ],
        },
        output={
            "answer": answer,
            "finish_reason": choice.finish_reason,
        },
        metadata={
            "model": MODEL,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    )

    print(
        f"\nTokens: input={input_tokens}, "
        f"output={output_tokens}, "
        f"total={input_tokens + output_tokens}"
    )

    return answer


# --------------------------------------------------
# 6. RAG workflow with trace attributes
# --------------------------------------------------

@observe(name="rag-retrieval-diagnostics-workflow")
def run_rag(question):
    with propagate_attributes(
        user_id="learner_001",
        session_id="rag-diagnostics-session-001",
        tags=["learning-lab", "rag", "retrieval-diagnostics"],
        metadata={
            "experiment": "rag-02-retrieval-diagnostics",
            "retrieval_method": "keyword-overlap",
            "provider": "openrouter",
        },
    ):
        print("\n" + "=" * 65)
        print("Question:", question)

        retrieval = retrieve_documents(question)
        answer = generate_answer(question, retrieval["selected"])

        print("\nAnswer:\n")
        print(answer)

        return {
            "question": question,
            "retrieved_count": retrieval["retrieved_count"],
            "answer": answer,
        }


# --------------------------------------------------
# 7. Run experiments
# --------------------------------------------------

def main():
    questions = [
        "How do Kubernetes Deployments manage replicas and updates?",
        "What is a Kubernetes Service?",
        "Explain quantum computing.",
    ]

    try:
        for question in questions:
            try:
                run_rag(question)
            except Exception as exc:
                # Continue testing even if one API call fails
                print(f"\nExperiment failed for question: {question}")
                print(f"Error: {type(exc).__name__}: {exc}")
    finally:
        langfuse.flush()
        print("\nLangfuse flush requested.")


if __name__ == "__main__":
    main()
