from __future__ import annotations

from typing import Any

import httpx
from langchain_core.tools import tool


SEARXNG_BASE_URL = "http://127.0.0.1:16000"


@tool
def search_web(query: str, max_results: int = 5) -> str:
    """
    Search the public web using the local SearXNG server.

    Args:
        query: Focused web search query.
        max_results: Number of results to return, from 1 to 10.
    """
    max_results = max(1, min(max_results, 10))

    try:
        response = httpx.get(
            f"{SEARXNG_BASE_URL}/search",
            params={
                "q": query,
                "format": "json",
                "language": "en-US",
                "safesearch": 1,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
    except httpx.HTTPError as exc:
        return f"SearXNG request failed: {exc}"
    except ValueError:
        return "SearXNG returned invalid JSON."

    results = payload.get("results", [])[:max_results]

    if not results:
        return "No web results found."

    output: list[str] = []

    for index, result in enumerate(results, start=1):
        output.append(
            "\n".join(
                [
                    f"{index}. {result.get('title', 'Untitled')}",
                    f"URL: {result.get('url', '')}",
                    f"Summary: {result.get('content', '')}",
                ]
            )
        )

    return "\n\n".join(output)