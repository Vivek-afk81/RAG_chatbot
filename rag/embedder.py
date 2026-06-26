import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Loading once globally
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def build_index(chunks: list[str]):
    embeddings = embedding_model.encode(
        chunks, normalize_embeddings=True
    )
    vectors = np.array(embeddings, dtype="float32")
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)
    return index


def query_index(question: str, index, chunks: list[str], k: int = 3):
    query_embedding = embedding_model.encode(
        [question], normalize_embeddings=True
    )
    query_vector = np.array(query_embedding, dtype="float32")

    D, I = index.search(query_vector, k=k)

    results = [chunks[idx] for idx in I[0]]
    scores = D[0].tolist()

    return results, scores