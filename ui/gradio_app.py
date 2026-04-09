import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import shutil
from src.pipeline import ingest, ask
from src.vector_store import reset_index, index_size
from src.config import UPLOADS_PATH


def upload_pdfs(files):
    if not files:
        return "No files selected.", get_status()
    os.makedirs(UPLOADS_PATH, exist_ok=True)
    saved = []
    for file in files:
        dest = os.path.join(UPLOADS_PATH, os.path.basename(file.name))
        shutil.copy(file.name, dest)
        saved.append(dest)
    try:
        result = ingest(saved)
        names = ", ".join(result["files_processed"])
        msg = f"Processed {result['total_chunks']} chunks from: {names}"
        return msg, get_status()
    except Exception as e:
        return f"Error: {str(e)}", get_status()


def reset():
    reset_index()
    return "Index cleared. You can now upload new documents.", get_status()


def get_status():
    n = index_size()
    if n == 0:
        return "No documents loaded."
    return f"{n} chunks ready for querying."


def chat(message, history):
    if not message.strip():
        return history, ""
    if index_size() == 0:
        history.append((message, "No documents have been uploaded yet. Please upload a PDF first."))
        return history, ""
    try:
        result = ask(message)
        citations = result.get("citations", [])
        answer = result["answer"]
        if citations:
            sources = "\n".join(
                f"  {c['source']}  —  Page {c['page']}" for c in citations
            )
            full_response = f"{answer}\n\nSources:\n{sources}"
        else:
            full_response = answer
        history.append((message, full_response))
    except Exception as e:
        history.append((message, f"Error: {str(e)}"))
    return history, ""


CSS = """
footer { display: none !important; }
.gr-button { font-weight: 500; }
.status-text { font-size: 13px; color: #6b7280; }
"""

with gr.Blocks(title="RAG System", theme=gr.themes.Default(), css=CSS) as demo:

    gr.Markdown("""
# RAG System
Ask questions across multiple PDF documents and get answers with source citations.
""")

    with gr.Tab("Documents"):
        gr.Markdown("Upload one or more PDF files. They will be processed and made available for querying.")

        with gr.Row():
            with gr.Column(scale=3):
                file_input = gr.File(
                    file_count="multiple",
                    file_types=[".pdf"],
                    label="Select PDF files",
                )
            with gr.Column(scale=1):
                status_box = gr.Textbox(
                    value=get_status(),
                    label="Status",
                    interactive=False,
                    lines=2,
                )
                upload_btn = gr.Button("Upload & Process", variant="primary")
                reset_btn = gr.Button("Clear All Documents", variant="stop")

        upload_msg = gr.Textbox(label="Result", interactive=False, visible=True)

        upload_btn.click(
            upload_pdfs,
            inputs=file_input,
            outputs=[upload_msg, status_box]
        )
        reset_btn.click(
            reset,
            outputs=[upload_msg, status_box]
        )

    with gr.Tab("Ask"):
        gr.Markdown("Type a question about your uploaded documents. Answers include page-level source citations.")

        chatbot = gr.Chatbot(
            label="",
            height=400,
            bubble_full_width=False,
            show_label=False,
        )

        with gr.Row():
            question_input = gr.Textbox(
                placeholder="Ask a question about your documents...",
                show_label=False,
                scale=5,
                container=False,
            )
            ask_btn = gr.Button("Send", variant="primary", scale=1)

        gr.Examples(
            examples=[
                "What is the main topic of the document?",
                "Summarize the key findings.",
                "What methodology is described?",
            ],
            inputs=question_input,
            label="Example questions",
        )

        ask_btn.click(
            chat,
            inputs=[question_input, chatbot],
            outputs=[chatbot, question_input]
        )
        question_input.submit(
            chat,
            inputs=[question_input, chatbot],
            outputs=[chatbot, question_input]
        )

    with gr.Tab("How It Works"):
        gr.Markdown("""
## Pipeline

1. **Document Parsing** — PDF files are parsed page by page using `pdfplumber`
2. **Chunking** — Text is split into overlapping segments (512 tokens, 50-token overlap) using LangChain
3. **Embedding** — Each chunk is embedded with `sentence-transformers/all-MiniLM-L6-v2`
4. **Indexing** — Embeddings are stored in a FAISS vector index, persisted to disk
5. **Retrieval** — At query time, the top-5 most semantically relevant chunks are retrieved
6. **Generation** — Groq (`llama-3.3-70b-versatile`) generates a grounded answer with citations

## Stack

| Component     | Technology                                      |
|---------------|-------------------------------------------------|
| PDF Parsing   | pdfplumber                                      |
| Chunking      | LangChain RecursiveCharacterTextSplitter        |
| Embeddings    | sentence-transformers / all-MiniLM-L6-v2        |
| Vector Store  | FAISS                                           |
| LLM           | Groq — llama-3.3-70b-versatile                  |
| Backend API   | FastAPI                                         |
| UI            | Gradio                                          |
""")


if __name__ == "__main__":
    demo.launch()
