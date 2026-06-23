from .embeddings import model

def retrieve(question, chunks, index, k=3):

    query_embedding = model.encode([question])
    distances, indices = index.search(
        query_embedding,
        k
    )
    return [chunks[i] for i in indices[0]]