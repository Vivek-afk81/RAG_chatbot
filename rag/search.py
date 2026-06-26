from huggingface_hub import InferenceClient
from rag.embedder import query_index


# Semantic search mode — returns top-k relevant passages with scores.
def semantic_search(question: str, index, chunks: list[str], k: int = 3) -> list[dict]:
    results, scores = query_index(question, index, chunks, k=k)
    return [
        {
            "rank":i+1,
            "score":round(scores[i], 4),
            "passage":results[i],
        }
        for i in range(len(results))
    ]


def answer_question(question: str, index, chunks: list[str], client: InferenceClient) -> str:
    context_chunks, _ = query_index(question, index, chunks, k=3)
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a helpful assistant.

Answer ONLY using the provided context.

If the answer is not present in the context, reply exactly:

"I cannot find that information in the document."

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
    )
    return response.choices[0].message.content