import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Tuple, Optional
from src.config import FAISS_INDEX_PATH

_cache: Tuple[Optional[faiss.Index], List[Dict]] = (None, [])


def _paths():
    return f"{FAISS_INDEX_PATH}/index.faiss", f"{FAISS_INDEX_PATH}/metadata.pkl"


def _invalidate_cache():
    global _cache
    _cache = (None, [])


def build_index(chunks: List[Dict], embeddings: List[List[float]]) -> None:
    """Add chunks to the FAISS index, creating it if it doesn't exist."""
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    vectors = np.array(embeddings, dtype="float32")

    index_path, meta_path = _paths()
    existing_index, existing_chunks = load_index()

    if existing_index is not None:
        existing_index.add(vectors)
        all_chunks = existing_chunks + chunks
        faiss.write_index(existing_index, index_path)
    else:
        dim = vectors.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(vectors)
        all_chunks = chunks
        faiss.write_index(index, index_path)

    with open(meta_path, "wb") as f:
        pickle.dump(all_chunks, f)

    _invalidate_cache()


def load_index() -> Tuple[Optional[faiss.Index], List[Dict]]:
    """Load FAISS index and metadata, using in-memory cache."""
    global _cache
    if _cache[0] is not None:
        return _cache

    index_path, meta_path = _paths()
    if not os.path.exists(index_path):
        return None, []

    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        chunks = pickle.load(f)

    _cache = (index, chunks)
    return _cache


def reset_index() -> None:
    """Delete persisted index and clear cache."""
    index_path, meta_path = _paths()
    for path in [index_path, meta_path]:
        if os.path.exists(path):
            os.remove(path)
    _invalidate_cache()


def index_size() -> int:
    _, chunks = load_index()
    return len(chunks)


def indexed_sources() -> set:
    """Return set of source filenames already in the index."""
    _, chunks = load_index()
    return {os.path.basename(c["source"]) for c in chunks}
