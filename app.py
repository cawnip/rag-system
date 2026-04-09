import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from src.schemas import UploadResponse, AskRequest, AskResponse, HealthResponse, Citation
from src.pipeline import ingest, ask
from src.vector_store import reset_index, index_size
from src.config import UPLOADS_PATH

app = FastAPI(
    title="RAG System",
    description="Multi-PDF Retrieval-Augmented Generation API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok", "indexed_chunks": index_size()}


@app.post("/upload", response_model=UploadResponse)
async def upload(files: List[UploadFile] = File(...)):
    os.makedirs(UPLOADS_PATH, exist_ok=True)
    saved_paths = []

    for file in files:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{file.filename} is not a PDF.")
        dest = os.path.join(UPLOADS_PATH, file.filename)
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        saved_paths.append(dest)

    result = ingest(saved_paths)
    return UploadResponse(message="PDFs indexed successfully.", **result)


@app.post("/ask", response_model=AskResponse)
def ask_question(body: AskRequest):
    result = ask(body.question)
    return AskResponse(
        answer=result["answer"],
        citations=[Citation(**c) for c in result["citations"]],
    )


@app.delete("/reset")
def reset():
    reset_index()
    return {"message": "Index reset successfully."}
