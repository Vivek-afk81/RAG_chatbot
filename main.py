import os
import sys
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from rag.loader import load_pdf
from rag.chunker import chunk_text_by_method, compare_chunk_sizes,compare_chunk_methods, CHUNK_PRESETS,CHUNK_METHODS,METHOD_MAP
from rag.embedder import build_index, query_index
from rag.search import semantic_search, answer_question


# Setup

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(api_key=HF_TOKEN)

pdf_path = sys.argv[1] if len(sys.argv) > 1 else "data/book-of-short-stories.pdf"


# Load PDF
print(f"\nLoading: {pdf_path}")
text = load_pdf(pdf_path)
print(f"Extracted {len(text):,} characters")

size_map = {"1": "small", "2": "medium", "3": "large"}
method_map = {"1": "Fixed", "2": "Sentence", "3": "Semantic"}


def pick_method_and_chunk(text):
    """Prompts for method (and size, if Fixed) and returns (chunks, method_label, size_label)."""
    print("\n--- Chunking Method Comparison ---")
    method_comparison = compare_chunk_methods(text)
    for label, stats in method_comparison.items():
        print(
            f"  {label:<10} | chunks={stats['total_chunks']:>4} "
            f"| avg_chars={stats['avg_chars']:>5}"
        )

    print("\nWhich chunking method do you want to use?")
    print("  1. Fixed    (char window — pick a size next)")
    print("  2. Sentence (groups of N sentences via NLTK)")
    print("  3. Semantic (groups sentences by embedding similarity)")

    m_choice = input("\nEnter 1 / 2 / 3 [default: 1]: ").strip() or "1"
    method_label = method_map.get(m_choice, "Fixed")
    method = METHOD_MAP[method_label]

    if method == "fixed":
        print("\n--- Chunk Size Comparison ---")
        comparison = compare_chunk_sizes(text)
        for label, stats in comparison.items():
            print(
                f"  {label:<8} | chunk_size={stats['chunk_size']:>5} "
                f"| chunks={stats['total_chunks']:>4} "
                f"| avg_chars={stats['avg_chars']:>5}"
            )

        print("\nWhich chunk size do you want to use?")
        print("  1. small  (300 chars  — more chunks, finer retrieval)")
        print("  2. medium (1000 chars — your original default)")
        print("  3. large  (2000 chars — fewer chunks, broader context)")

        s_choice = input("\nEnter 1 / 2 / 3 [default: 2]: ").strip() or "2"
        size_label = size_map.get(s_choice, "medium")
        preset = CHUNK_PRESETS[size_label]
        chunks = chunk_text_by_method(text, method="fixed", **preset)
    else:
        size_label = "—"
        chunks = chunk_text_by_method(text, method=method)

    print(f"\nUsing '{method_label}'" + (f" / '{size_label}'" if method == "fixed" else "")
          + f" — {len(chunks)} total chunks")

    return chunks, method_label, size_label


chunks, current_method, current_size = pick_method_and_chunk(text)


# Build FAISS Index
print("Building FAISS index...")
index = build_index(chunks)
print(f"Vectors stored: {index.ntotal}")


# Chat Loop
print("\nReady!")
print("Commands: 'search' for semantic search mode | 'chunk' to re-pick size (Fixed only) "
      "| 'method' to re-pick chunking method | 'quit' to exit")

while True:
    question = input("\nAsk: ").strip()

    if not question:
        continue

    if question.lower() in ["quit", "exit"]:
        print("Goodbye!")
        break

    # Semantic Search Mode
    if question.lower() == "search":
        query = input("  Search query: ").strip()
        if not query:
            continue
        results = semantic_search(query, index, chunks, k=3)
        print(f"\nTop {len(results)} passages for: '{query}'\n")
        for r in results:
            print(f"  [{r['rank']}] Score: {r['score']}")
            print(f"  {r['passage'][:300].strip()}...")
            print()
        continue

    # Re-pick Chunking Method
    if question.lower() == "method":
        chunks, current_method, current_size = pick_method_and_chunk(text)
        index = build_index(chunks)
        print(f"  Rebuilt index — {len(chunks)} total chunks")
        continue

    # Re-pick Chunk Size (Fixed method only)
    if question.lower() == "chunk":
        if current_method != "Fixed":
            print(f"  Current method is '{current_method}' — size doesn't apply. "
                  f"Use 'method' to switch to Fixed first.")
            continue
        print("\n  1. small  2. medium  3. large")
        choice = input("  Pick: ").strip()
        current_size = size_map.get(choice, "medium")
        preset = CHUNK_PRESETS[current_size]
        chunks = chunk_text_by_method(text, method="fixed", **preset)
        index = build_index(chunks)
        print(f"  Rebuilt index with '{current_size}' chunks — {len(chunks)} total")
        continue

    # Answer Question
    try:
        answer = answer_question(question, index, chunks, client)
        print("\nAnswer:")
        print(answer)
    except Exception as e:
        print(f"Error: {e}")