from functools import lru_cache

from google import genai
from google.genai import types

from app.core.config import settings


@lru_cache
def get_gemini_client():
    return genai.Client(api_key=settings.gemini_api_key)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """
    Create document embeddings for stored passages.
    """
    if not texts:
        return []

    client = get_gemini_client()

    response = client.models.embed_content(
        model=settings.gemini_embedding_model,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=settings.gemini_embedding_dimensions,
        ),
    )

    embeddings = [
        item.values
        for item in response.embeddings
    ]

    if len(embeddings) != len(texts):
        raise RuntimeError("Gemini returned an unexpected number of embeddings")

    for vector in embeddings:
        if len(vector) != settings.gemini_embedding_dimensions:
            raise RuntimeError("Embedding dimension does not match database")

    return embeddings


def embed_query(question: str) -> list[float]:
    """
    Create an embedding for a student's question.
    """
    client = get_gemini_client()

    response = client.models.embed_content(
        model=settings.gemini_embedding_model,
        contents=question,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=settings.gemini_embedding_dimensions,
        ),
    )

    vector = response.embeddings[0].values

    if len(vector) != settings.gemini_embedding_dimensions:
        raise RuntimeError("Query embedding dimension mismatch")

    return vector


def generate_answer(prompt: str) -> str:
    """
    Generate an answer from retrieved course material.
    """
    client = get_gemini_client()

    response = client.models.generate_content(
        model=settings.gemini_chat_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=1200,
        ),
    )

    answer = response.text

    if not answer:
        raise RuntimeError("Gemini returned an empty response")

    return answer.strip()