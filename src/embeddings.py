from sentence_transformers import SentenceTransformer
from typing import List
from src.config import EMBEDDING_MODEL

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_model()
    return model.encode(texts, show_progress_bar=True).tolist()


def embed_query(text: str) -> List[float]:
    model = get_model()
    return model.encode(text).tolist()
