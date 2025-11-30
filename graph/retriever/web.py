"""
Web search retrieval logic (MCP integration point).

This module now calls the MCP tool `web.search` 
"""

from typing import Dict, List
import logging

from graph.tools import call_mcp_tool

logger = logging.getLogger(__name__)


def retrieve_from_web(
    query: str,
    filters: Dict,
    k: int = 5,
) -> List[Dict]:
    """
    Retrieve products from live web search via MCP.

    Args:
        query: Search query text.
        filters: Dict with planner filters (category, min_price, max_price, brand, material).
        k: Number of results.

    Returns:
        List of product dicts with standard format:
        {
            "title": ...,
            "url": ...,
            "snippet": ...,
            "price": ... or None,
            "availability": ... or None,
            "source": "serper"
        }
    """

    logger.info("[WEB] Calling MCP tool web.search")

    # You can optionally pass site_filter or other params here.
    result = call_mcp_tool(
        "web.search",
        {
            "query": query,
            "max_results": k,
            # "site_filter": "site:amazon.com",  # uncomment if you want to restrict
        },
    )

    if "error" in result:
        logger.warning(f"[WEB] MCP web.search error: {result['error']}")
        return []

    docs = result.get("results", [])
    logger.info(f"[WEB] Retrieved {len(docs)} results from MCP web.search")
    return docs[:k]
