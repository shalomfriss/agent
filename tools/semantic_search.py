
from __future__ import annotations

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from langchain_core.tools import tool


QDRANT_URL = "http://127.0.0.1:6333"
COLLECTION_NAME = "source_code"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

_client = QdrantClient(url=QDRANT_URL)
_model = SentenceTransformer(EMBEDDING_MODEL)


@tool
def semantic_code_search(
    question: str,
    max_results: int = 8,
) -> str:
    """
    Search the codebase by meaning rather than exact text.

    Use this when the user describes behavior or concepts without knowing
    the exact class, method, file, or symbol name.

    Examples:
    - Where is authentication handled?
    - What code retries failed network requests?
    - How is offline caching implemented?
    - Where are API responses converted into view models?

    Args:
        question: Conceptual description of the code being sought.
        max_results: Number of code chunks to return, from 1 to 20.
    """
    max_results = max(1, min(max_results, 20))

    query_vector = _model.encode(
        question,
        normalize_embeddings=True,
    ).tolist()

    response = _client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=max_results,
        with_payload=True,
    )

    if not response.points:
        return "No semantic code matches found."

    formatted: list[str] = []

    for index, point in enumerate(response.points, start=1):
        payload = point.payload or {}

        formatted.append(
            "\n".join(
                [
                    f"Result {index}",
                    f"Score: {point.score:.4f}",
                    f"File: {payload.get('path', '')}",
                    (
                        f"Lines: {payload.get('start_line', '')}-"
                        f"{payload.get('end_line', '')}"
                    ),
                    f"Language: {payload.get('language', '')}",
                    "",
                    str(payload.get("content", "")),
                ]
            )
        )

    return "\n\n---\n\n".join(formatted)