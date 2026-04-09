import pdfplumber
from typing import List, Dict


def load_pdf(file_path: str) -> List[Dict]:
    """Extract text from PDF, returns list of {text, page, source}."""
    pages = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append({
                    "text": text.strip(),
                    "page": i + 1,
                    "source": file_path,
                })

    if not pages:
        raise ValueError(
            f"No text could be extracted from '{file_path}'. "
            "The file may be a scanned image or contain only non-text content."
        )

    return pages


def load_pdfs(file_paths: List[str]) -> List[Dict]:
    """Load multiple PDFs. Raises ValueError listing all unreadable files."""
    all_pages = []
    failed = []

    for path in file_paths:
        try:
            all_pages.extend(load_pdf(path))
        except ValueError as e:
            failed.append(str(e))

    if failed:
        raise ValueError("\n".join(failed))

    return all_pages
