import os
import numpy as np
import faiss
import pymupdf

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer



# Load Environment Variables
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(
    api_key=HF_TOKEN
)



# Extract PDF Text
doc = pymupdf.open("data/book-of-short-stories.pdf")

text = ""

for page in doc:
    page_text = page.get_text()

    if page_text:
        text += page_text + "\n"

doc.close()


# Chunking

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []

    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks
chunks = chunk_text(text)
print(f"Total chunks: {len(chunks)}")



# Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = embedding_model.encode(
    chunks,normalize_embeddings=True
)



# FAISS Vector Database

vectors = np.array(embeddings,dtype="float32")
dimension = vectors.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(vectors)
print(f"Vectors stored: {index.ntotal}")



# Chat Loop
while True:

    question = input("\nAsk: ")

    if question.lower() in ["quit", "exit"]:
        print("Goodbye!")
        break

    # Query embedding
    query_embedding = embedding_model.encode(
        [question],normalize_embeddings=True)

    query_vector = np.array(
        query_embedding,
        dtype="float32"
    )

    # Search top chunks
    D, I = index.search(
        query_vector,
        k=5
    )

    context = "\n\n".join(
        [chunks[idx] for idx in I[0]]
    )

    # for debugging purpose
    # print("\nRetrieved Context:")
    # print(context[:1000])

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

    try:

        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=300,
        )
        answer = response.choices[0].message.content
        print("\nAnswer:")
        print(answer)
     #exception handling
    except Exception as e:
        print(f"Error: {e}")