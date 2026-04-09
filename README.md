# RAG System — Multi-PDF Question Answering

A production-ready Retrieval-Augmented Generation (RAG) pipeline that lets you upload multiple PDF documents and ask natural language questions. Answers are grounded in the source material and include page-level citations.

---

## Demo

> Upload PDFs → Ask questions → Get cited answers

![RAG System UI](assets/demo.png)

**Live demo:** [huggingface.co/spaces/cawnip/rag-system](https://huggingface.co/spaces/cawnip/rag-system)

---

## Features

- Multi-PDF ingestion with automatic text extraction
- Semantic search using FAISS vector index (persisted to disk)
- Page-level source citations for every answer
- REST API (FastAPI) + interactive UI (Gradio)
- Duplicate detection and scanned-PDF detection
- In-memory index cache for fast repeated queries
- Answers in the same language as the question

---

## Architecture

```
PDF files
   │
   ▼
pdfplumber          ← text extraction (page by page)
   │
   ▼
RecursiveCharacterTextSplitter  ← chunk_size=512, overlap=50
   │
   ▼
all-MiniLM-L6-v2   ← sentence-transformers embeddings
   │
   ▼
FAISS Index         ← persisted to data/faiss_index/
   │
   ▼  (at query time: embed question → top-5 semantic search)
   │
   ▼
llama-3.3-70b-versatile (Groq)  ← grounded answer generation
   │
   ▼
Answer + Citations
```

---

## Tech Stack

| Component     | Technology                               |
|---------------|------------------------------------------|
| PDF Parsing   | pdfplumber                               |
| Chunking      | LangChain RecursiveCharacterTextSplitter |
| Embeddings    | sentence-transformers / all-MiniLM-L6-v2 |
| Vector Store  | FAISS (faiss-cpu)                        |
| LLM           | Groq — llama-3.3-70b-versatile           |
| Backend API   | FastAPI                                  |
| UI            | Gradio                                   |

---

## Project Structure

```
rag-system/
├── app.py                  # FastAPI application
├── ui/
│   └── gradio_app.py       # Gradio UI
├── src/
│   ├── config.py           # Environment variables and constants
│   ├── loader.py           # PDF text extraction
│   ├── chunker.py          # Text splitting
│   ├── embeddings.py       # Sentence-transformers wrapper
│   ├── vector_store.py     # FAISS index management
│   ├── retriever.py        # Semantic search
│   ├── llm_service.py      # Groq API integration
│   ├── pipeline.py         # Ingest and ask pipelines
│   └── schemas.py          # Pydantic request/response models
├── tests/
│   ├── test_chunker.py
│   └── test_pipeline.py
├── notebooks/
│   └── evaluation.ipynb     # RAGAS evaluation
├── assets/
│   └── demo.png
├── requirements.txt
├── Makefile
└── .env.example
```

---

## Getting Started

### 1. Clone and install

```bash
git clone https://github.com/your-username/rag-system.git
cd rag-system
pip install -r requirements.txt
```

### 2. Set up environment

```bash
cp .env.example .env
```

Edit `.env` and add your [Groq API key](https://console.groq.com):

```
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Run the Gradio UI

```bash
python ui/gradio_app.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860)

### 4. Run the FastAPI backend

```bash
make run-api
```

API docs available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## API Reference

### `POST /upload`
Upload and index one or more PDF files.

```bash
curl -X POST http://localhost:8000/upload \
  -F "files=@document.pdf"
```

### `POST /ask`
Ask a question against the indexed documents.

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main finding?"}'
```

Response:
```json
{
  "answer": "The main finding is ...",
  "citations": [
    {"source": "document.pdf", "page": 3},
    {"source": "document.pdf", "page": 7}
  ]
}
```

### `GET /health`
Returns the number of indexed chunks.

### `DELETE /reset`
Clears the entire index.

---

## Configuration

All settings are in `src/config.py` and can be overridden via environment variables:

| Variable          | Default                   | Description                  |
|-------------------|---------------------------|------------------------------|
| `GROQ_API_KEY`    | —                         | Groq API key (required)      |
| `GROQ_MODEL`      | llama-3.3-70b-versatile   | LLM model                    |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2          | Sentence-transformers model  |
| `CHUNK_SIZE`      | 512                       | Tokens per chunk             |
| `CHUNK_OVERLAP`   | 50                        | Overlap between chunks       |
| `TOP_K`           | 5                         | Retrieved chunks per query   |

---

## License

MIT
