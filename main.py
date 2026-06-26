import os
import sys
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from rag.loader import load_pdf
from rag.chunker import chunk_text, compare_chunk_sizes, CHUNK_PRESETS
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

# Chunk Size Comparison Mode 
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

choice = input("\nEnter 1 / 2 / 3 [default: 2]: ").strip() or "2"
size_map = {"1": "small", "2": "medium", "3": "large"}
chosen = size_map.get(choice, "medium")
preset = CHUNK_PRESETS[chosen]

chunks = chunk_text(text, **preset)
print(f"\nUsing '{chosen}' chunks — {len(chunks)} total chunks")


#  Build FAISS Index 
print("Building FAISS index...")
index = build_index(chunks)
print(f"Vectors stored: {index.ntotal}")


#  Chat Loop
print("\nReady!")
print("Commands: 'search' for semantic search mode | 'chunk' to re-pick chunk size | 'quit' to exit")

while True:
    question = input("\nAsk: ").strip()

    if not question:
        continue

    if question.lower() in ["quit", "exit"]:
        print("Goodbye!")
        break

    #  Semantic Search Mode 
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

    #  Re-pick Chunk Size 
    if question.lower() == "chunk":
        print("\n  1. small  2. medium  3. large")
        choice = input("  Pick: ").strip()
        chosen = size_map.get(choice, "medium")
        preset = CHUNK_PRESETS[chosen]
        chunks = chunk_text(text, **preset)
        index = build_index(chunks)
        print(f"  Rebuilt index with '{chosen}' chunks — {len(chunks)} total")
        continue

    # Answer Question 
    try:
        answer = answer_question(question, index, chunks, client)
        print("\nAnswer:")
        print(answer)
    except Exception as e:
        print(f"Error: {e}")