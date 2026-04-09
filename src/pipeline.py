import os
from typing import List, Dict
from src.loader import load_pdfs
from src.chunker import chunk_pages
from src.embeddings import embed_texts
from src.vector_store import build_index, reset_index, indexed_sources
from src.retriever import retrieve
from src.llm_service import generate_answer
from src.config import UPLOADS_PATH


def ingest(file_paths: List[str]) -> Dict:
    """Full ingestion pipeline: load → chunk → embed → index."""
    os.makedirs(UPLOADS_PATH, exist_ok=True)

    already_indexed = indexed_sources()
    duplicates = [p for p in file_paths if os.path.basename(p) in already_indexed]
    if duplicates:
        names = ", ".join(os.path.basename(p) for p in duplicates)
        raise ValueError(f"Already indexed: {names}. Use 'Clear All Documents' to re-upload.")

    pages = load_pdfs(file_paths)
    if not pages:
        raise ValueError("No text could be extracted from the uploaded PDFs.")

    chunks = chunk_pages(pages)
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    build_index(chunks, embeddings)

    for path in file_paths:
        try:
            os.remove(path)
        except OSError:
            pass

    return {
        "files_processed": [os.path.basename(p) for p in file_paths],
        "total_chunks": len(chunks),
    }


def ask(question: str) -> Dict:
    """RAG pipeline: retrieve → generate → return answer + citations."""
    chunks = retrieve(question)
    if not chunks:
        return {
            "answer": "No documents have been uploaded yet. Please upload a PDF first.",
            "citations": [],
        }

    answer = generate_answer(question, chunks)

    seen = set()
    citations = []
    for c in chunks:
        key = (c["source"], c["page"])
        if key not in seen:
            seen.add(key)
            citations.append({"source": c["source"], "page": c["page"]})

    return {"answer": answer, "citations": citations}
