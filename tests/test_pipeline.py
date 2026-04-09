import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# loader
# ---------------------------------------------------------------------------

class TestLoadPdf:
    def test_extracts_pages(self, tmp_path):
        fake_pdf = str(tmp_path / "test.pdf")
        # Create an empty file so pdfplumber.open() receives a valid path arg
        open(fake_pdf, "wb").close()

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "  Hello world  "

        with patch("pdfplumber.open") as mock_open:
            mock_open.return_value.__enter__.return_value.pages = [mock_page]
            from src.loader import load_pdf
            pages = load_pdf(fake_pdf)

        assert len(pages) == 1
        assert pages[0]["text"] == "Hello world"
        assert pages[0]["page"] == 1
        assert pages[0]["source"] == fake_pdf

    def test_raises_on_empty_pdf(self, tmp_path):
        fake_pdf = str(tmp_path / "empty.pdf")
        open(fake_pdf, "wb").close()

        mock_page = MagicMock()
        mock_page.extract_text.return_value = None

        with patch("pdfplumber.open") as mock_open:
            mock_open.return_value.__enter__.return_value.pages = [mock_page]
            from src.loader import load_pdf
            with pytest.raises(ValueError, match="No text could be extracted"):
                load_pdf(fake_pdf)

    def test_skips_blank_pages(self, tmp_path):
        fake_pdf = str(tmp_path / "test.pdf")
        open(fake_pdf, "wb").close()

        blank = MagicMock()
        blank.extract_text.return_value = "   "
        real = MagicMock()
        real.extract_text.return_value = "Real content"

        with patch("pdfplumber.open") as mock_open:
            mock_open.return_value.__enter__.return_value.pages = [blank, real]
            from src.loader import load_pdf
            pages = load_pdf(fake_pdf)

        assert len(pages) == 1
        assert pages[0]["page"] == 2


# ---------------------------------------------------------------------------
# vector_store
# ---------------------------------------------------------------------------

class TestVectorStore:
    def test_build_and_load(self, tmp_path):
        with patch("src.vector_store.FAISS_INDEX_PATH", str(tmp_path / "idx")):
            # Re-import so the patched path takes effect
            import importlib
            import src.vector_store as vs
            importlib.reload(vs)

            chunks = [{"text": "hello", "page": 1, "source": "a.pdf"}]
            embeddings = [[0.1] * 384]

            vs.build_index(chunks, embeddings)
            index, loaded_chunks = vs.load_index()

            assert index is not None
            assert len(loaded_chunks) == 1
            assert loaded_chunks[0]["text"] == "hello"

    def test_reset_clears_index(self, tmp_path):
        with patch("src.vector_store.FAISS_INDEX_PATH", str(tmp_path / "idx")):
            import importlib
            import src.vector_store as vs
            importlib.reload(vs)

            vs.build_index(
                [{"text": "data", "page": 1, "source": "x.pdf"}],
                [[0.2] * 384],
            )
            vs.reset_index()
            index, chunks = vs.load_index()

            assert index is None
            assert chunks == []

    def test_indexed_sources(self, tmp_path):
        with patch("src.vector_store.FAISS_INDEX_PATH", str(tmp_path / "idx")):
            import importlib
            import src.vector_store as vs
            importlib.reload(vs)

            vs.build_index(
                [{"text": "x", "page": 1, "source": "/uploads/report.pdf"}],
                [[0.3] * 384],
            )
            sources = vs.indexed_sources()
            assert "report.pdf" in sources


# ---------------------------------------------------------------------------
# pipeline — ingest
# ---------------------------------------------------------------------------

class TestIngest:
    def _run_ingest(self, tmp_path, fake_file):
        pages = [{"text": "Sample content", "page": 1, "source": str(fake_file)}]
        chunks = [{"text": "Sample content", "page": 1, "source": str(fake_file)}]
        embeddings = [[0.1] * 384]

        with (
            patch("src.pipeline.load_pdfs", return_value=pages),
            patch("src.pipeline.chunk_pages", return_value=chunks),
            patch("src.pipeline.embed_texts", return_value=embeddings),
            patch("src.pipeline.build_index"),
            patch("src.pipeline.indexed_sources", return_value=set()),
            patch("src.pipeline.UPLOADS_PATH", str(tmp_path)),
            patch("os.remove"),
        ):
            from src.pipeline import ingest
            return ingest([str(fake_file)])

    def test_ingest_returns_summary(self, tmp_path):
        fake_file = tmp_path / "report.pdf"
        fake_file.write_text("dummy")
        result = self._run_ingest(tmp_path, fake_file)

        assert "files_processed" in result
        assert "total_chunks" in result
        assert result["total_chunks"] == 1
        assert "report.pdf" in result["files_processed"]

    def test_ingest_rejects_duplicate(self, tmp_path):
        fake_file = tmp_path / "report.pdf"
        fake_file.write_text("dummy")

        with (
            patch("src.pipeline.indexed_sources", return_value={"report.pdf"}),
            patch("src.pipeline.UPLOADS_PATH", str(tmp_path)),
        ):
            from src.pipeline import ingest
            with pytest.raises(ValueError, match="Already indexed"):
                ingest([str(fake_file)])


# ---------------------------------------------------------------------------
# pipeline — ask
# ---------------------------------------------------------------------------

class TestAsk:
    def test_ask_returns_answer_and_citations(self):
        mock_chunks = [
            {"text": "FAISS is a vector search library.", "source": "doc.pdf", "page": 2},
        ]
        with (
            patch("src.pipeline.retrieve", return_value=mock_chunks),
            patch("src.pipeline.generate_answer", return_value="FAISS is made by Meta."),
        ):
            from src.pipeline import ask
            result = ask("What is FAISS?")

        assert result["answer"] == "FAISS is made by Meta."
        assert len(result["citations"]) == 1
        assert result["citations"][0]["page"] == 2

    def test_ask_deduplicates_citations(self):
        mock_chunks = [
            {"text": "Chunk A", "source": "doc.pdf", "page": 3},
            {"text": "Chunk B", "source": "doc.pdf", "page": 3},  # same page
        ]
        with (
            patch("src.pipeline.retrieve", return_value=mock_chunks),
            patch("src.pipeline.generate_answer", return_value="Answer"),
        ):
            from src.pipeline import ask
            result = ask("Question?")

        assert len(result["citations"]) == 1

    def test_ask_empty_index(self):
        with patch("src.pipeline.retrieve", return_value=[]):
            from src.pipeline import ask
            result = ask("Any question?")

        assert "No documents" in result["answer"]
        assert result["citations"] == []
