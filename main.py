import pymupdf  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


doc = pymupdf.open("data/book-of-short-stories.pdf")
text = ""

for page in doc:
    text += page.get_text()
doc.close()


#chunking
def chunk_text(text,chunk_size=500):
    chunks=[]

    for i in range(0,len(text),chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

chunks=chunk_text(text)

#Embeddings

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings=embedding_model.encode(chunks)


#Faiss storing in vector db
vectors = np.array(
    embeddings
).astype("float32")

dimension = vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(vectors)

print(index.ntotal) 

#Retrieval

question = input("Ask: ")
query_embedding = embedding_model.encode(
    [question]
)

#search
D, I = index.search(
    np.array(query_embedding).astype("float32"),
    k=3
)

#show chunks
for idx in I[0]:
    print("\n")
    print(chunks[idx])