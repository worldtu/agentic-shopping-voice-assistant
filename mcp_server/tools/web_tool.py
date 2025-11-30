import os
import json
from typing import Any, Dict, List, Optional

import aiohttp
import asyncio

from mcp_server.util import SimpleTTLCache, log_request_response
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env.local")


SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
SERPER_URL = "https://google.serper.dev/search"

# Cache web.search responses for a short time
_cache = SimpleTTLCache(ttl_seconds=300)


async def web_search(
    query: str,
    max_results: int = 5,
    site_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """
    MCP tool: web.search

    Args:
        query: User query text.
        max_results: Maximum number of organic results to return.
        site_filter: Optional site restriction (e.g. "site:amazon.com").

    Returns:
        Dict with shape: {"results": [ {title, url, snippet, price?, availability?, source}, ... ]}
    """

    if not SERPER_API_KEY:
        # Return an explicit error payload instead of raising
        result = {
            "error": "SERPER_API_KEY is not set in environment.",
            "results": [],
        }
        log_request_response("web.search", {"query": query}, result)
        return result

    # Apply simple site filter into query if provided
    final_query = f"{query} {site_filter}".strip() if site_filter else query

    cache_key = json.dumps(
        {"q": final_query, "max_results": max_results},
        sort_keys=True,
        ensure_ascii=False,
    )
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {"q": final_query}

    results_out: List[Dict[str, Any]] = []

    async with aiohttp.ClientSession() as session:
        async with session.post(SERPER_URL, json=payload, headers=headers) as resp:
            data = await resp.json()
            print("[SERPER RAW RESPONSE]", data)

    organic = data.get("organic", [])[:max_results]

    for item in organic:
        results_out.append(
            {
                "title": item.get("title"),
                "url": item.get("link"),
                "snippet": item.get("snippet"),
                # Price / availability are often not provided by generic search,
                # so we leave them as None; this still satisfies the schema.
                "price": None,
                "availability": None,
                "source": "serper",
            }
        )

    result = {"results": results_out}
    _cache.set(cache_key, result)
    log_request_response(
        "web.search",
        {"query": final_query, "max_results": max_results},
        result,
    )
    return result


def web_search_sync(
    query: str,
    max_results: int = 5,
    site_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Synchronous wrapper so that LangGraph nodes can call web.search
    without managing an event loop.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(
        web_search(query=query, max_results=max_results, site_filter=site_filter)
    )

