import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 5

FAISS_INDEX_PATH = "data/faiss_index"
UPLOADS_PATH = "data/uploads"
