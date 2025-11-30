import json
import sys
import importlib
from unittest.mock import patch
from langchain_core.runnables import RunnableLambda


# --- Fake LLM that forces router to output web_search ---
def fake_llm():
    def run(query):
        return json.dumps({
            "task": "product_search",
            "constraints": {},
            "safety_flags": [],
            "sources": ["web_search"]      
        })
    return RunnableLambda(run)


@patch("graph.router.get_llm", side_effect=fake_llm)          # PATCH #1
@patch("graph.models.llm.get_llm", side_effect=fake_llm)      # PATCH #2
@patch("graph.retriever.rag.retrieve_from_rag", return_value=[])
def test_web_trigger(mock_rag, mock_llm2, mock_llm1):

    for module in list(sys.modules.keys()):
        if (
            module.startswith("graph.router")
            or module.startswith("graph.models.llm")
            or module.startswith("graph.nodes")
            or module.startswith("graph.graph")
        ):
            del sys.modules[module]

    import graph.graph
    importlib.reload(graph.graph)

    create_graph = graph.graph.create_graph
    graph = create_graph()

    result = graph.invoke({
        "query": "is shampoo available now?",
        "step_log": []
    })

    assert "web_search" in result["plan"]["sources"], \
        f"Expected web_search, got {result['plan']['sources']}"
