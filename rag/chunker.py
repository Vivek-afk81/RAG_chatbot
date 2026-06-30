CHUNK_PRESETS = {
    "small":  {"chunk_size": 300,  "overlap": 50},
    "medium": {"chunk_size": 1000, "overlap": 200},  # default
    "large":  {"chunk_size": 2000, "overlap": 300},
}

CHUNK_METHODS = ["Fixed", "Sentence", "Semantic"]
METHOD_MAP = {"Fixed": "fixed", "Sentence": "sentence", "Semantic": "semantic"}


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Your original fixed-size chunker — unchanged."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def sentence_chunk(text: str, sentences_per_chunk: int = 5, **_ignored) -> list[str]:
    """Groups N sentences per chunk using NLTK's sentence tokenizer."""
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    from nltk.tokenize import sent_tokenize

    sentences = sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i:i + sentences_per_chunk]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def semantic_chunk(text: str, similarity_threshold: float = 0.5,
                    max_chunk_size: int = 1500, **_ignored) -> list[str]:
    """Groups consecutive sentences while embedding similarity stays high;
    splits when topic shifts (similarity drop) or size cap is hit."""
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    from nltk.tokenize import sent_tokenize

    sentences = sent_tokenize(text)
    if not sentences:
        return []

    model = SentenceTransformer("all-MiniLM-L6-v2")  # same 384-dim space as your embedder
    embeddings = model.encode(sentences, show_progress_bar=False)

    chunks = []
    current = [sentences[0]]

    for i in range(1, len(sentences)):
        a, b = embeddings[i - 1], embeddings[i]
        sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
        current_len = sum(len(s) for s in current)

        if sim < similarity_threshold or current_len > max_chunk_size:
            chunks.append(" ".join(current).strip())
            current = [sentences[i]]
        else:
            current.append(sentences[i])

    if current:
        chunks.append(" ".join(current).strip())

    return [c for c in chunks if c]


def chunk_text_by_method(text: str, method: str = "fixed", **kwargs) -> list[str]:
    """Single entry point app.py calls — dispatches to the right strategy."""
    if method == "fixed":
        allowed = {"chunk_size", "overlap"}
        return chunk_text(text, **{k: v for k, v in kwargs.items() if k in allowed})
    elif method == "sentence":
        return sentence_chunk(text, **kwargs)
    elif method == "semantic":
        return semantic_chunk(text, **kwargs)
    raise ValueError(f"Unknown chunking method: {method}")


def compare_chunk_sizes(text: str) -> dict:
    """Unchanged — fixed-size preset comparison (size radio)."""
    results = {}
    for label, params in CHUNK_PRESETS.items():
        chunks = chunk_text(text, **params)
        avg_len = sum(len(c) for c in chunks) / len(chunks) if chunks else 0
        results[label] = {
            "chunk_size": params["chunk_size"],
            "overlap": params["overlap"],
            "total_chunks": len(chunks),
            "avg_chars": round(avg_len),
            "chunks": chunks,
        }
    return results


def compare_chunk_methods(text: str) -> dict:
    """New — compares Fixed vs Sentence vs Semantic on the same text."""
    results = {}
    for label, method in METHOD_MAP.items():
        chunks = chunk_text_by_method(text, method=method)
        avg_len = sum(len(c) for c in chunks) / len(chunks) if chunks else 0
        results[label] = {
            "method": method,
            "total_chunks": len(chunks),
            "avg_chars": round(avg_len),
            "chunks": chunks,
        }
    return results