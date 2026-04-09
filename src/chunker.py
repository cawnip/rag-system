from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_pages(pages: List[Dict]) -> List[Dict]:
    """Split pages into chunks, preserving source and page metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = []
    for page in pages:
        splits = splitter.split_text(page["text"])
        for split in splits:
            chunks.append({
                "text": split,
                "page": page["page"],
                "source": page["source"],
            })
    return chunks
