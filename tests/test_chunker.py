import pytest
from src.chunker import chunk_pages


def test_chunk_pages_basic():
    pages = [{"text": "Hello world. " * 50, "page": 1, "source": "doc.pdf"}]
    chunks = chunk_pages(pages)
    assert len(chunks) > 0
    for chunk in chunks:
        assert "text" in chunk
        assert chunk["page"] == 1
        assert chunk["source"] == "doc.pdf"


def test_chunk_pages_preserves_metadata():
    pages = [
        {"text": "Page one content. " * 40, "page": 1, "source": "a.pdf"},
        {"text": "Page two content. " * 40, "page": 2, "source": "a.pdf"},
    ]
    chunks = chunk_pages(pages)
    pages_seen = {c["page"] for c in chunks}
    assert 1 in pages_seen
    assert 2 in pages_seen


def test_chunk_pages_short_text_stays_single():
    pages = [{"text": "Short text.", "page": 1, "source": "doc.pdf"}]
    chunks = chunk_pages(pages)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Short text."


def test_chunk_pages_empty_input():
    assert chunk_pages([]) == []


def test_chunk_size_respected():
    long_text = "word " * 1000
    pages = [{"text": long_text, "page": 1, "source": "doc.pdf"}]
    chunks = chunk_pages(pages)
    for chunk in chunks:
        assert len(chunk["text"]) <= 600  # allow some slack for splitter boundaries
