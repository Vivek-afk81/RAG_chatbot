CHUNK_PRESETS = {
    "small":  {"chunk_size": 300,  "overlap": 50},
    "medium": {"chunk_size": 1000, "overlap": 200},  # default
    "large":  {"chunk_size": 2000, "overlap": 300},
}

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:

    chunks=[]
    start=0
    while start < len(text):
        end =start+chunk_size
        chunks.append(text[start:end])
        start += chunk_size-overlap
    return chunks


   
# Run all 3 presets on the same text and return stats for each.Shows how chunk size affects the number and size of chunks.
    
def compare_chunk_sizes(text: str) -> dict:

    results = {}

    for label, params in CHUNK_PRESETS.items():
        chunks = chunk_text(text, **params)
        avg_len = sum(len(c) for c in chunks) / len(chunks) if chunks else 0

        results[label] = {
            "chunk_size":params["chunk_size"],
            "overlap":params["overlap"],
            "total_chunks":len(chunks),
            "avg_chars":round(avg_len),
            "chunks":chunks,
        }

    return results