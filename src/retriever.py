import numpy as np
import os
from typing import List, Dict
from src.embeddings import embed_query
from src.vector_store import load_index
from src.config import TOP_K


def retrieve(question: str) -> List[Dict]:
    """Semantic search — returns top-k chunks with citation metadata."""
    index, chunks = load_index()
    if index is None or len(chunks) == 0:
        return []

    query_vec = np.array([embed_query(question)], dtype="float32")
    k = min(TOP_K, len(chunks))
    _, indices = index.search(query_vec, k)

    results = []
    for i in indices[0]:
        if i == -1 or i >= len(chunks):
            continue
        chunk = chunks[i]
        results.append({
            "text": chunk["text"],
            "source": os.path.basename(chunk["source"]),
            "page": chunk["page"],
        })
    return results
