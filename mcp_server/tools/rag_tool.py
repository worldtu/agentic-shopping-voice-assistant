from typing import Any, Dict, Optional

from mcp_server.util import log_request_response
from graph.retriever.rag1 import rag_search as rag_backend_search


async def rag_search(
    query: str,
    k: int = 10,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    MCP tool: rag.search (async version)

    Args:
        query: User query text.
        k: Number of products to retrieve.
        filters: Optional structured filters from planner
                 (e.g. {"max_price": 15, "category": "cleaner"}).

    Returns:
        Dict with shape:
        {
          "results": [
             {
               "doc_id": ...,
               "title": ...,
               "price": ...,
               "category": ...,
               "brand": ...,
               "material": ...,
               "ingredients": ...,
               "rating": ...,
               "content": ...,
               "score": ...,
               "source": "rag"
             },
             ...
          ]
        }
    """
    filters = filters or {}

    # Call your FAISS-based private RAG implementation
    products = rag_backend_search(query=query, filters=filters, k=k)

    result = {"results": products}
    log_request_response(
        "rag.search",
        {"query": query, "k": k, "filters": filters},
        result,
    )
    return result


def rag_search_sync(
    query: str,
    k: int = 10,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Synchronous shim used by LangGraph nodes.

    This keeps the graph logic simple:
    - Graph just calls rag_search_sync(...)
    - Under the hood we still share the same implementation and logging.
    """
    filters = filters or {}
    products = rag_backend_search(query=query, filters=filters, k=k)
    result = {"results": products}
    log_request_response(
        "rag.search",
        {"query": query, "k": k, "filters": filters},
        result,
    )
    return result
