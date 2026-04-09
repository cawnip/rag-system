from groq import Groq
from typing import List, Dict
from src.config import GROQ_API_KEY, GROQ_MODEL

_client = None


def get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def generate_answer(question: str, context_chunks: List[Dict]) -> str:
    """Send question + retrieved context to Groq LLM and return answer."""
    context = "\n\n".join(
        f"[{c['source']} — Page {c['page']}]\n{c['text']}"
        for c in context_chunks
    )

    prompt = f"""You are a helpful assistant that answers questions based strictly on the provided context.
Answer in the same language as the question.
If the answer cannot be found in the context, say "I couldn't find relevant information in the uploaded documents."

Context:
{context}

Question: {question}

Answer:"""

    response = get_client().chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()
