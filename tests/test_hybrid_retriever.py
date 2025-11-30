# tests/test_hybrid_retriever.py

import sys
from pathlib import Path
from unittest.mock import patch
import pytest

# Ensure we can import the graph package
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from graph.nodes import hybrid_retriever_node


def test_hybrid():
    """
    Unit test for hybrid_retriever_node:
    - should call BOTH retrieve_from_rag and retrieve_from_web
    - should merge results into retrieved_docs
    - should append correct step_log entries
    """

    # ---- Construct minimal valid state ----
    state = {
        "query": "shampoo cheap now",
        "task": "product_search",
        "constraints": {},
        "plan": {
            # Directly specify both sources (hybrid).
            # This test does NOT rely on router or planner LLM outputs.
            "sources": ["private_rag", "web_search"],
            "retrieval_fields": ["title", "brand", "price", "rating"],
            "comparison_criteria": ["price", "rating"],
            "filters": {},
        },
        "retrieved_docs": [],
        "answer": "",
        "citations": [],
        "step_log": [],
    }

    # Fake RAG and Web docs
    fake_rag_docs = [
        {"title": "rag-doc-1", "price": 9.99, "source": "private_rag"},
    ]
    fake_web_docs = [
        {"title": "web-doc-1", "price": 7.99, "source": "web_search"},
    ]

    # ⚠️ IMPORTANT:
    # Patch retrieve_from_rag / retrieve_from_web inside graph.nodes,
    # because hybrid_retriever_node imports them there.
    with patch(
        "graph.nodes.retrieve_from_rag",
        return_value=fake_rag_docs,
    ) as mock_rag, patch(
        "graph.nodes.retrieve_from_web",
        return_value=fake_web_docs,
    ) as mock_web:

        new_state = hybrid_retriever_node(state)

    # ---- Assert: both retrievers were called ----
    mock_rag.assert_called_once()
    mock_web.assert_called_once()

    # ---- Assert: results merged into retrieved_docs ----
    docs = new_state["retrieved_docs"]
    assert len(docs) == 2
    titles = {d["title"] for d in docs}
    assert {"rag-doc-1", "web-doc-1"} <= titles

    # ---- Assert: step_log contains hybrid_retriever entry ----
    hybrid_logs = [
        log for log in new_state["step_log"]
        if log["node"] == "hybrid_retriever"
    ]

    assert len(hybrid_logs) == 1
    assert hybrid_logs[0]["success"] is True
    assert hybrid_logs[0]["output"]["rag_docs"] == 1
    assert hybrid_logs[0]["output"]["web_docs"] == 1

    print("✓ hybrid_retriever_node correctly called RAG + Web and merged results.")
