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

    # Reformulate query for product shopping
    product_query = _reformulate_for_shopping(query, filters)
    
    logger.info(f"[WEB] Reformulated query: {product_query}")

    # Call MCP with shopping-optimized parameters
    result = call_mcp_tool(
        "web.search",
        {
            "query": product_query,
            "max_results": k * 2,  # Request more to filter out non-products
            "search_type": "shopping",  # Request shopping results
        },
    )

    if "error" in result:
        logger.warning(f"[WEB] MCP web.search error: {result['error']}")
        return []

    docs = result.get("results", [])
    logger.info(f"[WEB] Retrieved {len(docs)} results from MCP web.search")
    
    # Filter out non-product results and extract prices
    filtered_docs = _filter_and_enhance_results(docs, query)
    
    return filtered_docs[:k]


def _reformulate_for_shopping(query: str, filters: Dict) -> str:
    """
    Reformulate query to be more product-shopping focused.
    
    Adds context keywords and filters to avoid generic/stock results.
    """
    import re
    
    # Start with the base query
    reformulated = query.lower()
    
    # Remove common query prefixes and phrases (order matters!)
    # Remove these patterns first to avoid leaving orphan words
    reformulated = re.sub(r'\b(find|show me|search for|looking for|get me)\b', '', reformulated)
    reformulated = re.sub(r'\b(current price for|price for|price of|cost of)\b', '', reformulated)
    reformulated = re.sub(r'\b(the current|current|latest)\b', '', reformulated)
    reformulated = re.sub(r'\b(what is|what\'s|whats)\b', '', reformulated)
    
    # Clean up multiple spaces
    reformulated = re.sub(r'\s+', ' ', reformulated).strip()
    
    # Remove leading articles if they start the query
    reformulated = re.sub(r'^\b(a|an|the)\b\s+', '', reformulated)
    
    # Add shopping context if not already present
    shopping_keywords = ["buy", "shop", "purchase", "product", "price", "store"]
    if not any(kw in reformulated for kw in shopping_keywords):
        reformulated = f"buy {reformulated}"
    
    # Add price range if available
    if filters.get("max_price") and filters.get("min_price"):
        reformulated += f" under ${filters['max_price']}"
    elif filters.get("max_price"):
        reformulated += f" under ${filters['max_price']}"
    
    # Add brand filter
    if filters.get("brand"):
        reformulated += f" {filters['brand']}"
    
    # Add material filter
    if filters.get("material"):
        reformulated += f" {filters['material']}"
    
    # NOTE: Do not append multiple site filters for Serper shopping queries.
    # The shopping endpoint doesn't support Google-style "OR" clauses and
    # returns zero results if we force them, so we let Serper decide sites.
    
    return reformulated.strip()


def _filter_and_enhance_results(docs: List[Dict], original_query: str) -> List[Dict]:
    """
    Filter out non-product results and extract prices from snippets.
    
    Args:
        docs: Raw web search results
        original_query: Original query for relevance checking
    
    Returns:
        Filtered and enhanced results
    """
    import re
    
    filtered = []
    
    for doc in docs:
        url = doc.get("url", "").lower()
        snippet = doc.get("snippet", "")
        
        # Filter out non-product pages
        # Skip: stock tickers, news, Wikipedia, generic info pages
        if any(x in url for x in ["finance.yahoo.com", "/stock", "/quote", "wikipedia.org", 
                                   "investopedia.com", "bloomberg.com", "reuters.com"]):
            logger.debug(f"[WEB] Filtering out non-product URL: {url}")
            continue
        
        # Prefer shopping sites
        is_shopping_site = any(x in url for x in ["amazon.com", "walmart.com", "target.com", 
                                                    "ebay.com", "etsy.com", "/product", "/p/", "/dp/"])
        
        # Try to extract price from snippet
        price = _extract_price_from_snippet(snippet)
        if price:
            doc["price"] = price
        
        # Try to extract availability
        availability = _extract_availability_from_snippet(snippet)
        if availability:
            doc["availability"] = availability
        
        # Add relevance boost for shopping sites
        doc["is_shopping_site"] = is_shopping_site
        
        filtered.append(doc)
    
    # Sort by shopping site preference
    filtered.sort(key=lambda x: x.get("is_shopping_site", False), reverse=True)
    
    return filtered


def _extract_price_from_snippet(snippet: str) -> float:
    """Extract price from snippet text using regex."""
    import re
    
    # Pattern: $XX.XX or $XX or XX.XX USD
    patterns = [
        r'\$(\d+\.?\d*)',  # $14.99 or $14
        r'(\d+\.?\d*)\s*USD',  # 14.99 USD
        r'Price:\s*\$?(\d+\.?\d*)',  # Price: $14.99
    ]
    
    for pattern in patterns:
        match = re.search(pattern, snippet)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, IndexError):
                continue
    
    return None


def _extract_availability_from_snippet(snippet: str) -> str:
    """Extract availability status from snippet."""
    snippet_lower = snippet.lower()
    
    if any(x in snippet_lower for x in ["in stock", "available now", "available"]):
        return "in_stock"
    elif any(x in snippet_lower for x in ["out of stock", "unavailable", "sold out"]):
        return "out_of_stock"
    elif any(x in snippet_lower for x in ["pre-order", "coming soon"]):
        return "pre_order"
    
    return None
