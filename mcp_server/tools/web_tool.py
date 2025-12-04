import os
import json
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
import threading

import aiohttp
import asyncio

from mcp_server.util import SimpleTTLCache, log_request_response
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env.local")


SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
SERPER_SEARCH_URL = "https://google.serper.dev/search"
SERPER_SHOPPING_URL = "https://google.serper.dev/shopping"

# Cache web.search responses for a short time
_cache = SimpleTTLCache(ttl_seconds=300)


async def web_search(
    query: str,
    max_results: int = 5,
    site_filter: Optional[str] = None,
    search_type: str = "search",  # "search" or "shopping"
) -> Dict[str, Any]:
    """
    MCP tool: web.search

    Args:
        query: User query text.
        max_results: Maximum number of results to return.
        site_filter: Optional site restriction (e.g. "site:amazon.com").
        search_type: "search" for general search, "shopping" for product search.

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
        {"q": final_query, "max_results": max_results, "type": search_type},
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
    payload = {"q": final_query, "num": max_results}

    results_out: List[Dict[str, Any]] = []

    # Choose endpoint based on search type
    endpoint = SERPER_SHOPPING_URL if search_type == "shopping" else SERPER_SEARCH_URL

    async with aiohttp.ClientSession() as session:
        async with session.post(endpoint, json=payload, headers=headers) as resp:
            data = await resp.json()
            print(f"[SERPER {search_type.upper()} RAW RESPONSE]", json.dumps(data, indent=2))

    # Handle shopping results
    if search_type == "shopping":
        shopping_results = data.get("shopping", [])[:max_results]
        
        for item in shopping_results:
            # Shopping results have rich product data
            price_str = item.get("price")
            price = _parse_price(price_str) if price_str else None
            
            results_out.append({
                "title": item.get("title"),
                "url": item.get("link"),
                "snippet": item.get("snippet") or item.get("description", ""),
                "price": price,
                "availability": _extract_availability(item),
                "source": "serper_shopping",
                "rating": item.get("rating"),
                "reviews": item.get("reviews"),
                "image_url": item.get("imageUrl"),
            })
    else:
        # Handle regular organic search results
        organic = data.get("organic", [])[:max_results]
        
        for item in organic:
            results_out.append({
                "title": item.get("title"),
                "url": item.get("link"),
                "snippet": item.get("snippet"),
                "price": None,
                "availability": None,
                "source": "serper",
            })

    # Fallback: Serper shopping can return zero items if the query format
    # isn't supported (e.g., multiple site filters). Retry with organic search
    if search_type == "shopping" and not results_out:
        print("[WEB SEARCH] Shopping endpoint returned 0 results. Falling back to general search.")
        return await web_search(
            query=query,
            max_results=max_results,
            site_filter=site_filter,
            search_type="search",
        )

    result = {"results": results_out}
    _cache.set(cache_key, result)
    
    # Log with source URL for robots.txt tracking
    source_url = endpoint if results_out else None
    log_request_response(
        "web.search",
        {"query": final_query, "max_results": max_results, "search_type": search_type},
        result,
        source_url=source_url
    )
    return result


def _parse_price(price_str: str) -> Optional[float]:
    """
    Parse price string to float.
    Handles formats like: "$14.99", "$14", "14.99", "14,999.99"
    """
    import re
    
    if not price_str:
        return None
    
    # Remove currency symbols and whitespace
    cleaned = re.sub(r'[$€£¥,\s]', '', str(price_str))
    
    # Extract first number (handles cases like "$14.99 - $19.99")
    match = re.search(r'(\d+\.?\d*)', cleaned)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    
    return None


def _extract_availability(item: Dict) -> Optional[str]:
    """
    Extract availability from shopping result item.
    """
    # Check delivery info
    delivery = item.get("delivery", "").lower()
    if "in stock" in delivery or "available" in delivery:
        return "in_stock"
    elif "out of stock" in delivery:
        return "out_of_stock"
    
    # If there's a price and link, assume available
    if item.get("price") and item.get("link"):
        return "in_stock"
    
    return None


def web_search_sync(
    query: str,
    max_results: int = 5,
    site_filter: Optional[str] = None,
    search_type: str = "search",
) -> Dict[str, Any]:
    """
    Synchronous wrapper so that LangGraph nodes can call web.search
    without managing an event loop.
    
    This function runs the async web_search in a separate thread with its own
    event loop to avoid conflicts with already-running event loops (e.g., FastAPI).
    """
    def run_in_thread():
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                web_search(
                    query=query, 
                    max_results=max_results, 
                    site_filter=site_filter,
                    search_type=search_type
                )
            )
        finally:
            loop.close()
    
    # Run the async function in a separate thread
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_in_thread)
        return future.result()

