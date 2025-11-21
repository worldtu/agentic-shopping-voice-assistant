# graph/retriever/__init__.py
"""
Unified retriever interface
Maintains backward compatibility with v1
"""

from graph.retriever.rag1 import retrieve_from_rag, get_vector_store,rag_with_auto_filter
from graph.retriever.web import retrieve_from_web
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

# Re-export for backward compatibility
__all__ = ['retrieve_products', 'retrieve_from_rag', 'retrieve_from_web', 'get_vector_store', 'rag_with_auto_filter']


def retrieve_products(
    query: str,
    filters: Dict,
    k: int = 5
) -> List[Dict]:
    """
    Unified retriever (v2) — automatically extracts filters via Groq API.
    Args:
        query: Search query text
        filters: Ignored (auto-handled by LLM)
        k: Number of results
    Returns:
        List of product dicts
    """
    return rag_with_auto_filter(query, k)
